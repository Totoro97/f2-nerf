//
// Created by ppwang on 2022/5/6.
//

#include "ExpRunner.h"
#include "Utils/CustomOps/CustomOps.h"
#include "Utils/StopWatch.h"
#include "Utils/Utils.h"
#include "Utils/cnpy.h"
#include <experimental/filesystem> // GCC 7.5?
#include <fmt/core.h>

namespace fs = std::experimental::filesystem::v1;
using Tensor = torch::Tensor;

ExpRunner::ExpRunner(const std::string &conf_path) {
  global_data_pool_ = std::make_unique<GlobalDataPool>(conf_path);
  const auto &config = global_data_pool_->config_;
  base_dir_ = config["base_dir"].as<std::string>();

  base_exp_dir_ = config["base_exp_dir"].as<std::string>();
  global_data_pool_->base_exp_dir_ = base_exp_dir_;

  fs::create_directories(base_exp_dir_);

  pts_batch_size_ = config["train"]["pts_batch_size"].as<int>();
  end_iter_ = config["train"]["end_iter"].as<int>();
  vis_freq_ = config["train"]["vis_freq"].as<int>();
  report_freq_ = config["train"]["report_freq"].as<int>();
  stats_freq_ = config["train"]["stats_freq"].as<int>();
  save_freq_ = config["train"]["save_freq"].as<int>();
  learning_rate_ = config["train"]["learning_rate"].as<float>();
  learning_rate_alpha_ = config["train"]["learning_rate_alpha"].as<float>();
  learning_rate_warm_up_end_iter_ =
      config["train"]["learning_rate_warm_up_end_iter"].as<int>();
  ray_march_init_fineness_ =
      config["train"]["ray_march_init_fineness"].as<float>();
  ray_march_fineness_decay_end_iter_ =
      config["train"]["ray_march_fineness_decay_end_iter"].as<int>();
  tv_loss_weight_ = config["train"]["tv_loss_weight"].as<float>();
  disp_loss_weight_ = config["train"]["disp_loss_weight"].as<float>();
  var_loss_weight_ = config["train"]["var_loss_weight"].as<float>();
  var_loss_start_ = config["train"]["var_loss_start"].as<int>();
  var_loss_end_ = config["train"]["var_loss_end"].as<int>();
  gradient_scaling_start_ = config["train"]["gradient_scaling_start"].as<int>();
  gradient_scaling_end_ = config["train"]["gradient_scaling_end"].as<int>();

  // Dataset
  dataset_ = std::make_unique<Dataset>(global_data_pool_.get());

  // Renderer
  renderer_ =
      std::make_unique<Renderer>(global_data_pool_.get(), dataset_->n_images_);

  // Optimizer
  optimizer_ =
      std::make_unique<torch::optim::Adam>(renderer_->OptimParamGroups());

  if (config["is_continue"].as<bool>()) {
    LoadCheckpoint(base_exp_dir_ + "/checkpoints/latest");
  }

  if (config["reset"] && config["reset"].as<bool>()) {
    renderer_->Reset();
  }
}

void ExpRunner::Train() {
  global_data_pool_->mode_ = RunningMode::TRAIN;

  std::string log_dir = base_exp_dir_ + "/logs";
  fs::create_directories(log_dir);

  std::vector<float> mse_records;
  float time_per_iter = 0.f;
  StopWatch clock;

  float psnr_smooth = -1.0;
  UpdateAdaParams();

  {
    StopWatch watch;
    global_data_pool_->iter_step_ = iter_step_;
    for (; iter_step_ < end_iter_;) {
      global_data_pool_->backward_nan_ = false;
      // global_data_pool_->drop_out_prob_ = 1.f - std::min(1.f,
      // float(iter_step_) / 1000.f); global_data_pool_->drop_out_prob_ = 0.f;

      int cur_batch_size =
          int(pts_batch_size_ /
              global_data_pool_->meaningful_sampled_pts_per_ray_) >>
          4 << 4;
      auto [train_rays, gt_colors, emb_idx] =
          dataset_->RandRaysData(cur_batch_size, DATA_TRAIN_SET);

      Tensor &rays_o = train_rays.origins;
      Tensor &rays_d = train_rays.dirs;
      Tensor &bounds = train_rays.bounds;

      auto render_result = renderer_->Render(rays_o, rays_d, bounds, emb_idx);
      Tensor pred_colors = render_result.colors.index({Slc(0, cur_batch_size)});
      Tensor disparity = render_result.disparity;
      Tensor color_loss =
          torch::sqrt((pred_colors - gt_colors).square() + 1e-4f).mean();

      Tensor disparity_loss = disparity.square().mean();

      Tensor edge_feats = render_result.edge_feats;
      Tensor tv_loss =
          (edge_feats.index({Slc(), 0}) - edge_feats.index({Slc(), 1}))
              .square()
              .mean();

      Tensor sampled_weights = render_result.weights;
      Tensor idx_start_end = render_result.idx_start_end;
      Tensor sampled_var = CustomOps::WeightVar(sampled_weights, idx_start_end);
      Tensor var_loss = (sampled_var + 1e-2).sqrt().mean();

      float var_loss_weight = 0.f;
      if (iter_step_ > var_loss_end_) {
        var_loss_weight = var_loss_weight_;
      } else if (iter_step_ > var_loss_start_) {
        var_loss_weight = float(iter_step_ - var_loss_start_) /
                          float(var_loss_end_ - var_loss_start_) *
                          var_loss_weight_;
      }

      Tensor loss = color_loss + var_loss * var_loss_weight +
                    disparity_loss * disp_loss_weight_ +
                    tv_loss * tv_loss_weight_;

      float mse = (pred_colors - gt_colors).square().mean().item<float>();
      float psnr = 20.f * std::log10(1 / std::sqrt(mse));
      psnr_smooth = psnr_smooth < 0.f ? psnr : psnr * .1f + psnr_smooth * .9f;
      CHECK(!std::isnan(pred_colors.mean().item<float>()));
      CHECK(!std::isnan(gt_colors.mean().item<float>()));
      CHECK(!std::isnan(mse));

      // There can be some cases that the output colors have no grad due to the
      // occupancy grid.
      if (loss.requires_grad()) {
        optimizer_->zero_grad();
        loss.backward();
        if (global_data_pool_->backward_nan_) {
          std::cout << "Nan!" << std::endl;
          continue;
        } else {
          optimizer_->step();
        }
      }

      mse_records.push_back(mse);

      iter_step_++;
      global_data_pool_->iter_step_ = iter_step_;

      if (iter_step_ % stats_freq_ == 0) {
        cnpy::npy_save(base_exp_dir_ + "/stats.npy", mse_records.data(),
                       {mse_records.size()});
      }

      if (iter_step_ % vis_freq_ == 0) {
        int t = iter_step_ / vis_freq_;
        int vis_idx;
        vis_idx = (iter_step_ / vis_freq_) % dataset_->test_set_.size();
        vis_idx = dataset_->test_set_[vis_idx];
        VisualizeImage(vis_idx);
      }

      if (iter_step_ % save_freq_ == 0) {
        SaveCheckpoint();
      }
      time_per_iter = time_per_iter * 0.6f + clock.TimeDuration() * 0.4f;

      if (iter_step_ % report_freq_ == 0) {
        std::cout << fmt::format(
                         "Iter: {:>6d} PSNR: {:.2f} NRays: {:>5d} OctSamples: "
                         "{:.1f} Samples: {:.1f} MeaningfulSamples: {:.1f} "
                         "IPS: {:.1f} LR: {:.4f}",
                         iter_step_, psnr_smooth, cur_batch_size,
                         global_data_pool_->sampled_oct_per_ray_,
                         global_data_pool_->sampled_pts_per_ray_,
                         global_data_pool_->meaningful_sampled_pts_per_ray_,
                         1.f / time_per_iter,
                         optimizer_->param_groups()[0].options().get_lr())
                  << std::endl;
      }
      UpdateAdaParams();
    }
    YAML::Node info_data;

    std::ofstream info_fout(base_exp_dir_ + "/train_info.txt");
    info_fout << watch.TimeDuration() << std::endl;
    info_fout.close();
  }

  std::cout << "Train done, test." << std::endl;
  TestImages();
}

void ExpRunner::LoadCheckpoint(const std::string &path) {
  {
    Tensor scalars;
    torch::load(scalars, path + "/scalars.pt");
    iter_step_ = std::round(scalars[0].item<float>());
    UpdateAdaParams();
  }

  {
    std::vector<Tensor> scene_states;
    torch::load(scene_states, path + "/renderer.pt");
    renderer_->LoadStates(scene_states, 0);
  }
}

void ExpRunner::SaveCheckpoint() {
  std::string output_dir =
      base_exp_dir_ + fmt::format("/checkpoints/{:0>8d}", iter_step_);
  fs::create_directories(output_dir);

  fs::remove_all(base_exp_dir_ + "/checkpoints/latest");
  fs::create_directory(base_exp_dir_ + "/checkpoints/latest");
  // scene
  torch::save(renderer_->States(), output_dir + "/renderer.pt");
  fs::create_symlink(output_dir + "/renderer.pt",
                     base_exp_dir_ + "/checkpoints/latest/renderer.pt");
  // optimizer
  // torch::save(*(optimizer_), output_dir + "/optimizer.pt");
  // other scalars
  Tensor scalars = torch::empty({1}, CPUFloat);
  scalars.index_put_({0}, float(iter_step_));
  torch::save(scalars, output_dir + "/scalars.pt");
  fs::create_symlink(output_dir + "/scalars.pt",
                     base_exp_dir_ + "/checkpoints/latest/scalars.pt");
}

void ExpRunner::UpdateAdaParams() {
  // Update ray march fineness
  if (iter_step_ >= ray_march_fineness_decay_end_iter_) {
    global_data_pool_->ray_march_fineness_ = 1.f;
  } else {
    float progress =
        float(iter_step_) / float(ray_march_fineness_decay_end_iter_);
    global_data_pool_->ray_march_fineness_ =
        std::exp(std::log(1.f) * progress +
                 std::log(ray_march_init_fineness_) * (1.f - progress));
  }
  // Update learning rate
  float lr_factor;
  if (iter_step_ >= learning_rate_warm_up_end_iter_) {
    float progress = float(iter_step_ - learning_rate_warm_up_end_iter_) /
                     float(end_iter_ - learning_rate_warm_up_end_iter_);
    lr_factor = (1.f - learning_rate_alpha_) *
                    (std::cos(progress * float(M_PI)) * .5f + .5f) +
                learning_rate_alpha_;
  } else {
    lr_factor = float(iter_step_) / float(learning_rate_warm_up_end_iter_);
  }
  float lr = learning_rate_ * lr_factor;
  for (auto &g : optimizer_->param_groups()) {
    g.options().set_lr(lr);
  }

  // Update gradient scaling ratio
  {
    float progress = 1.f;
    if (iter_step_ < gradient_scaling_end_) {
      progress = std::max(
          0.f, (float(iter_step_) - gradient_scaling_start_) /
                   (gradient_scaling_end_ - gradient_scaling_start_ + 1e-9f));
    }
    global_data_pool_->gradient_scaling_progress_ = progress;
  }
}

std::tuple<Tensor, Tensor, Tensor>
ExpRunner::RenderWholeImage(Tensor rays_o, Tensor rays_d, Tensor bounds) {
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor first_oct_disp = torch::full({n_rays, 1}, 1.f, CPUFloat);
  Tensor pred_disp = torch::zeros({n_rays, 1}, CPUFloat);

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o =
        rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d =
        rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_bounds =
        bounds.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result =
        renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor());
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
    if (!render_result.first_oct_dis.sizes().empty()) {
      Tensor &ret_first_oct_dis = render_result.first_oct_dis;
      if (ret_first_oct_dis.has_storage()) {
        Tensor cur_first_oct_dis =
            render_result.first_oct_dis.detach().to(torch::kCPU);
        first_oct_disp.index_put_({Slc(i, i_high)}, cur_first_oct_dis);
      }
    }
  }
  pred_disp = pred_disp / pred_disp.max();
  first_oct_disp = first_oct_disp.min() / first_oct_disp;

  return {pred_colors, first_oct_disp, pred_disp};
}

std::tuple<Tensor, Tensor> ExpRunner::RenderWholeImageForMesh(Tensor rays_o,
                                                              Tensor rays_d,
                                                              Tensor bounds) {
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor pred_depths = torch::zeros({n_rays, 1}, CPUFloat);

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o =
        rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d =
        rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_bounds =
        bounds.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result =
        renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor());
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    Tensor depths = render_result.depth.detach().to(torch::kCPU).squeeze();
    // Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_depths.index_put_({Slc(i, i_high)}, depths.unsqueeze(-1));
    // pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
    // if (!render_result.first_oct_dis.sizes().empty()) {
    //   Tensor &ret_first_oct_dis = render_result.first_oct_dis;
    //   if (ret_first_oct_dis.has_storage()) {
    //     Tensor cur_first_oct_dis =
    //         render_result.first_oct_dis.detach().to(torch::kCPU);
    //     first_oct_disp.index_put_({Slc(i, i_high)}, cur_first_oct_dis);
    //   }
    // }
  }
  // pred_disp = pred_disp / pred_disp.max();
  // first_oct_disp = first_oct_disp.min() / first_oct_disp;

  return {pred_colors, pred_depths};
}

void ExpRunner::RenderAllImages() {
  for (int idx = 0; idx < dataset_->n_images_; idx++) {
    VisualizeImage(idx);
  }
}

void ExpRunner::VisualizeImage(int idx) {
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  auto [rays_o, rays_d, bounds] = dataset_->RaysOfCamera(idx);
  auto [pred_colors, first_oct_dis, pred_disps] =
      RenderWholeImage(rays_o, rays_d, bounds);

  int H = dataset_->height_;
  int W = dataset_->width_;

  Tensor img_tensor = torch::cat(
      {dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 3}),
       pred_colors.reshape({H, W, 3}),
       first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
       pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})},
      1);
  fs::create_directories(base_exp_dir_ + "/images");
  Utils::WriteImageTensor(base_exp_dir_ + "/images/" +
                              fmt::format("{}_{}.png", iter_step_, idx),
                          img_tensor);

  global_data_pool_->mode_ = prev_mode;
}

void ExpRunner::RenderPath() {
  torch::NoGradGuard no_grad_guard;
  int n_images = dataset_->render_poses_.size(0);
  global_data_pool_->mode_ = RunningMode::VALIDATE;
  int res_level = 1;
  for (int i = 0; i < n_images; i++) {
    std::cout << i << std::endl;
    auto [rays_o, rays_d, bounds] =
        dataset_->RaysFromPose(dataset_->render_poses_[i], res_level);
    auto [pred_colors, first_oct_dis, pred_disps] =
        RenderWholeImage(rays_o, rays_d, bounds);
    int H = dataset_->height_ / res_level;
    int W = dataset_->width_ / res_level;

    Tensor img_tensor =
        torch::cat({pred_colors.reshape({H, W, 3}),
                    first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
                    pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})},
                   1);

    fs::create_directories(base_exp_dir_ + "/novel_images");
    Utils::WriteImageTensor(base_exp_dir_ + "/novel_images/" +
                                fmt::format("{}_{:0>3d}.png", iter_step_, i),
                            img_tensor);
  }
}

void ExpRunner::TestImages() {
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  float psnr_sum = 0.f;
  float cnt = 0.f;
  YAML::Node out_info;
  {
    fs::create_directories(base_exp_dir_ + "/test_images");
    for (int i : dataset_->test_set_) {
      auto [rays_o, rays_d, bounds] = dataset_->RaysOfCamera(i);
      auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(
          rays_o, rays_d, bounds); // At this stage, the returned number is

      int H = dataset_->height_;
      int W = dataset_->width_;

      auto quantify = [](const Tensor &x) {
        return (x.clip(0.f, 1.f) * 255.f)
                   .to(torch::kUInt8)
                   .to(torch::kFloat32) /
               255.f;
      };
      pred_disps = pred_disps.reshape({H, W, 1});
      first_oct_dis = first_oct_dis.reshape({H, W, 1});
      pred_colors = pred_colors.reshape({H, W, 3});
      pred_colors = quantify(pred_colors);
      float mse =
          (pred_colors.reshape({H, W, 3}) -
           dataset_->image_tensors_[i].to(torch::kCPU).reshape({H, W, 3}))
              .square()
              .mean()
              .item<float>();
      float psnr = 20.f * std::log10(1 / std::sqrt(mse));
      out_info[fmt::format("{}", i)] = psnr;
      std::cout << fmt::format("{}: {}", i, psnr) << std::endl;
      psnr_sum += psnr;
      cnt += 1.f;
      Utils::WriteImageTensor(
          base_exp_dir_ + "/test_images/" +
              fmt::format("color_{}_{:0>3d}.png", iter_step_, i),
          pred_colors);
      Utils::WriteImageTensor(
          base_exp_dir_ + "/test_images/" +
              fmt::format("depth_{}_{:0>3d}.png", iter_step_, i),
          pred_disps.repeat({1, 1, 3}));
      Utils::WriteImageTensor(
          base_exp_dir_ + "/test_images/" +
              fmt::format("oct_depth_{}_{:0>3d}.png", iter_step_, i),
          first_oct_dis.repeat({1, 1, 3}));
    }
  }
  float mean_psnr = psnr_sum / cnt;
  std::cout << fmt::format("Mean psnr: {}", mean_psnr) << std::endl;
  out_info["mean_psnr"] = mean_psnr;

  std::ofstream info_fout(base_exp_dir_ + "/test_images/info.yaml");
  info_fout << out_info;

  global_data_pool_->mode_ = prev_mode;
}
void ExpRunner::OutputMeshMeta() {

  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;
  std::string mesh_meta_path = base_exp_dir_ + "/mesh_meta";
  {
    fs::create_directories(mesh_meta_path);
    int counter = 0;
    for (int i : dataset_->train_set_) {
      auto [rays_o, rays_d, bounds] = dataset_->RaysOfCamera(i);
      auto [pred_colors, pred_depths] = RenderWholeImageForMesh(
          rays_o, rays_d, bounds); // At this stage, the returned number is

      int H = dataset_->height_;
      int W = dataset_->width_;

      auto quantify = [](const Tensor &x) {
        return (x.clip(0.f, 1.f) * 255.f)
                   .to(torch::kUInt8)
                   .to(torch::kFloat32) /
               255.f;
      };
      pred_colors = pred_colors.reshape({H, W, 3});
      pred_colors = quantify(pred_colors);
      pred_depths = pred_depths.reshape({H, W, 1});
      Tensor c2w = dataset_->c2w_[i].detach().to(torch::kCPU).contiguous();
      Tensor intri = dataset_->intri_[i].detach().to(torch::kCPU).contiguous();

      cnpy::npy_save(mesh_meta_path + "/" +
                         fmt::format("color_{}_{:0>3d}.npy", iter_step_, i),
                     (float *)pred_colors.data_ptr(),
                     {(unsigned long)H, (unsigned long)W, 3});

      cnpy::npy_save(mesh_meta_path + "/" +
                         fmt::format("depth_{}_{:0>3d}.npy", iter_step_, i),
                     (float *)pred_depths.data_ptr(),
                     {(unsigned long)H, (unsigned long)W, 1});
      cnpy::npy_save(mesh_meta_path + "/" +
                         fmt::format("c2w_{}_{:0>3d}.npy", iter_step_, i),
                     (float *)c2w.data_ptr(), {3, 4});
      cnpy::npy_save(mesh_meta_path + "/" +
                         fmt::format("intri_{}_{:0>3d}.npy", iter_step_, i),
                     (float *)intri.data_ptr(), {3, 3});
      counter += 1;
      if (counter % 20 == 0) {
        std::cout << "process: " << counter << "/"
                  << dataset_->train_set_.size() << std::endl;
      }
    }
  }
}
void ExpRunner::Execute() {
  std::string mode = global_data_pool_->config_["mode"].as<std::string>();
  std::cout << "===============> mode: " << mode.c_str() << std::endl;
  if (mode == "train") {
    Train();
  } else if (mode == "render_path") {
    RenderPath();
  } else if (mode == "test") {
    TestImages();
  } else if (mode == "render_all") {
    RenderAllImages();
  } else if (mode == "mesh_mata") {
    OutputMeshMeta();
  }
}
