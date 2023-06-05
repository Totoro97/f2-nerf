//
// Created by ppwang on 2022/5/7.
//

#include "Renderer.h"
#include "../Common.h"
#include "../Utils/Utils.h"
#include "../Utils/StopWatch.h"
#include "../Utils/CustomOps/CustomOps.h"
#include "../Utils/CustomOps/FlexOps.h"
#include "../Utils/CustomOps/Scatter.h"

using Tensor = torch::Tensor;
namespace F = torch::nn::functional;

TORCH_LIBRARY(volume_render, m)
{
  std::cout << "register volume render info" << std::endl;
  m.class_<VolumeRenderInfo>("VolumeRenderInfo").def(torch::init());
}

Renderer::Renderer(GlobalDataPool* global_data_pool, int n_images) {
  global_data_pool_ = global_data_pool;
  // global_data_pool_->renderer_ = std::reinterpret_cast<void*>(this);
  global_data_pool_->renderer_ = reinterpret_cast<void*>(this);
  auto conf = global_data_pool->config_["renderer"];

  pts_sampler_ = ConstructPtsSampler(global_data_pool);
  RegisterSubPipe(pts_sampler_.get());

  scene_field_ = ConstructField(global_data_pool);
  RegisterSubPipe(scene_field_.get());

  shader_ = ConstructShader(global_data_pool);
  RegisterSubPipe(shader_.get());

  use_app_emb_ = conf["use_app_emb"].as<bool>();
  // WARNING: Hard code here.
  app_emb_ = torch::randn({ n_images, 16 }, CUDAFloat) * .1f;
  app_emb_.requires_grad_(true);

  auto bg_color = conf["bg_color"].as<std::string>();
  if (bg_color == "white")
    bg_color_type_ = BGColorType::white;
  else if (bg_color == "black")
    bg_color_type_ = BGColorType::black;
  else
    bg_color_type_ = BGColorType::rand_noise;
}


RenderResult Renderer::Render(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds, const Tensor& emb_idx) {
#ifdef PROFILE
  ScopeWatch watch(__func__);
#endif
  int n_rays = rays_o.sizes()[0];
  sample_result_ = pts_sampler_->GetSamples(rays_o, rays_d, bounds);
  int n_all_pts = sample_result_.pts.sizes()[0];
  float sampled_pts_per_ray = float(n_all_pts) / float(n_rays);
  if (global_data_pool_->mode_ == RunningMode::TRAIN) {
    global_data_pool_->sampled_pts_per_ray_ =
        global_data_pool_->sampled_pts_per_ray_ * 0.9f + sampled_pts_per_ray * 0.1f;
  }
  CHECK(sample_result_.pts_idx_bounds.max().item<int>() <= n_all_pts);
  CHECK(sample_result_.pts_idx_bounds.min().item<int>() >= 0);

  Tensor bg_color;
  if (bg_color_type_ == BGColorType::white) {
    bg_color = torch::ones({n_rays, 3}, CUDAFloat);
  }
  else if (bg_color_type_ == BGColorType::rand_noise) {
    if (global_data_pool_->mode_ == RunningMode::TRAIN) {
      bg_color = torch::rand({n_rays, 3}, CUDAFloat);
    }
    else {
      bg_color = torch::ones({n_rays, 3}, CUDAFloat) * .5f;
    }
  }
  else {
    bg_color = torch::zeros({n_rays, 3}, CUDAFloat);
  }

  if (n_all_pts <= 0) {
    Tensor colors = bg_color;
    if (global_data_pool_->mode_ == RunningMode::TRAIN) {
      global_data_pool_->meaningful_sampled_pts_per_ray_ = global_data_pool_->meaningful_sampled_pts_per_ray_ * 0.9f;
    }
    return {
        colors,
        torch::zeros({ n_rays, 1 }, CUDAFloat),
        torch::zeros({ n_rays }, CUDAFloat),
        Tensor(),
        torch::full({ n_rays }, 512.f, CUDAFloat),
        Tensor(),
        Tensor()
    };
  }
  CHECK_EQ(rays_o.sizes()[0], sample_result_.pts_idx_bounds.sizes()[0]);

  auto DensityAct = [](Tensor x) -> Tensor {
    const float shift = 3.f;
    return torch::autograd::TruncExp::apply(x - shift)[0];
  };

  // First, inference without gradients - early stop
  SampleResultFlex sample_result_early_stop;
  {
    torch::NoGradGuard no_grad_guard;

    Tensor pts  = sample_result_.pts;
    Tensor dirs = sample_result_.dirs;
    Tensor anchors = sample_result_.anchors.index({"...", 0}).contiguous();

    Tensor scene_feat = scene_field_->AnchoredQuery(pts, anchors);
    Tensor sampled_density = DensityAct(scene_feat.index({ Slc(), Slc(0, 1) }));

    Tensor sampled_dt = sample_result_.dt;
    Tensor sampled_t = (sample_result_.t + 1e-2f).contiguous();
    Tensor sec_density = sampled_density.index({Slc(), 0}) * sampled_dt;
    Tensor alphas = 1.f - torch::exp(-sec_density);
    Tensor idx_start_end = sample_result_.pts_idx_bounds;
    Tensor acc_density = FlexOps::AccumulateSum(sec_density, idx_start_end, false);
    Tensor trans = torch::exp(-acc_density);
    Tensor weights = trans * alphas;
    Tensor mask = trans > 1e-4f;
    Tensor mask_idx = torch::where(mask)[0];

    sample_result_early_stop.pts = sample_result_.pts.index({mask_idx}).contiguous();
    sample_result_early_stop.dirs = sample_result_.dirs.index({mask_idx}).contiguous();
    sample_result_early_stop.dt = sample_result_.dt.index({mask_idx}).contiguous();
    sample_result_early_stop.t = sample_result_.t.index({mask_idx}).contiguous();
    sample_result_early_stop.anchors = sample_result_.anchors.index({mask_idx}).contiguous();

    sample_result_early_stop.first_oct_dis = sample_result_.first_oct_dis.clone();
    sample_result_early_stop.pts_idx_bounds = FilterIdxBounds(sample_result_.pts_idx_bounds, mask);

    CHECK_EQ(sample_result_early_stop.pts_idx_bounds.max().item<int>(), sample_result_early_stop.pts.size(0));


    if (global_data_pool_->mode_ == RunningMode::TRAIN) {
      pts_sampler_->UpdateOctNodes(sample_result_,
                                   weights.detach(),
                                   alphas.detach());

      float meaningful_per_ray = mask.to(torch::kFloat32).sum().item<float>();
      meaningful_per_ray /= n_rays;
      global_data_pool_->meaningful_sampled_pts_per_ray_ =
          global_data_pool_->meaningful_sampled_pts_per_ray_ * 0.9f + meaningful_per_ray * 0.1f;
    }
  }

  Tensor scene_feat, edge_feat;
  Tensor pts  = sample_result_early_stop.pts;
  Tensor dirs = sample_result_early_stop.dirs;
  Tensor anchors = sample_result_early_stop.anchors.index({"...", 0}).contiguous();
  n_all_pts = pts.size(0);

  // Feature variation loss.
  if (global_data_pool_->mode_ == RunningMode::TRAIN) {
    const int n_edge_pts = 8192;
    auto [ edge_pts, edge_anchors ] = pts_sampler_->GetEdgeSamples(n_edge_pts);
    edge_pts = edge_pts.reshape({ n_edge_pts * 2, 3 }).contiguous();
    edge_anchors = edge_anchors.reshape({ n_edge_pts * 2 }).contiguous();

    Tensor query_pts = torch::cat({ pts, edge_pts }, 0);
    Tensor query_anchors = torch::cat({ anchors, edge_anchors }, 0);
    Tensor all_feat = scene_field_->AnchoredQuery(query_pts, query_anchors);
    scene_feat = all_feat.slice(0, 0, n_all_pts);
    edge_feat = all_feat.slice(0, n_all_pts, n_all_pts + n_edge_pts * 2).reshape({ n_edge_pts, 2, -1 });
  }
  else {
    // Query density &gra color
    scene_feat = scene_field_->AnchoredQuery(pts, anchors);  // [n_pts, feat_dim];
  }


  Tensor idx_start_end = sample_result_early_stop.pts_idx_bounds;

  Tensor sampled_density = DensityAct(scene_feat.index({ Slc(), Slc(0, 1) }));

  Tensor shading_feat = torch::cat({torch::ones_like(scene_feat.index({Slc(), Slc(0, 1)}), CUDAFloat),
                                    scene_feat.index({Slc(), Slc(1, None)})}, 1);

  if (global_data_pool_->mode_ == RunningMode::TRAIN && use_app_emb_) {
    Tensor all_emb_idx = CustomOps::ScatterIdx(n_all_pts, sample_result_early_stop.pts_idx_bounds, emb_idx);
    shading_feat = CustomOps::ScatterAdd(app_emb_, all_emb_idx, shading_feat);
  }

  Tensor sampled_colors = shader_->Query(shading_feat, dirs);
  if (global_data_pool_->gradient_scaling_progress_ < 1.) {
    sampled_density = CustomOps::GradientScaling(sampled_density, idx_start_end,
                                                 global_data_pool_->gradient_scaling_progress_);
    sampled_colors = CustomOps::GradientScaling(sampled_colors, idx_start_end,
                                                global_data_pool_->gradient_scaling_progress_);
  }
  Tensor sampled_dt = sample_result_early_stop.dt;
  Tensor sampled_t = (sample_result_early_stop.t + 1e-2f).contiguous();
  Tensor sec_density = sampled_density.index({Slc(), 0}) * sampled_dt;
  Tensor alphas = 1.f - torch::exp(-sec_density);
  Tensor acc_density = FlexOps::AccumulateSum(sec_density, idx_start_end, false);
  Tensor trans = torch::exp(-acc_density);
  Tensor weights = trans * alphas;

  Tensor last_trans = torch::exp(-FlexOps::Sum(sec_density, idx_start_end));
  Tensor colors = FlexOps::Sum(weights.unsqueeze(-1) * sampled_colors, idx_start_end);
  colors = colors + last_trans.unsqueeze(-1) * bg_color;
  Tensor disparity = FlexOps::Sum(weights / sampled_t, idx_start_end);
  Tensor depth = FlexOps::Sum(weights * sampled_t, idx_start_end) / (1.f - last_trans + 1e-4f);

  CHECK_NOT_NAN(colors);

  return { colors, sample_result_early_stop.first_oct_dis, disparity, edge_feat, depth, weights, idx_start_end };
}


int Renderer::LoadStates(const std::vector<Tensor>& states, int idx) {
  for (auto pipe : sub_pipes_) {
    idx = pipe->LoadStates(states, idx);
  }

  app_emb_.data().copy_(states[idx++].clone().to(torch::kCUDA).contiguous());

  return idx;
}

std::vector<Tensor> Renderer::States() {
  std::vector<Tensor> ret;
  for (auto pipe : sub_pipes_) {
    auto cur_states = pipe->States();
    ret.insert(ret.end(), cur_states.begin(), cur_states.end());
  }

  ret.push_back(app_emb_.data());

  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> Renderer::OptimParamGroups() {
  std::vector<torch::optim::OptimizerParamGroup> ret;
  for (auto pipe : sub_pipes_) {
    auto cur_params = pipe->OptimParamGroups();
    for (const auto& para_group : cur_params) {
      ret.emplace_back(para_group);
    }
  }

  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(global_data_pool_->learning_rate_);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;
    opt->weight_decay() = 1e-6;

    std::vector<Tensor> params;
    params.push_back(app_emb_);
    ret.emplace_back(std::move(params), std::move(opt));
  }
  return ret;
}
