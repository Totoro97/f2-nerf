//
// Created by ppwang on 2022/5/7.
//

#ifndef SANR_RENDERER_H
#define SANR_RENDERER_H

#pragma once
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "../Utils/Pipe.h"
#include "../Utils/GlobalDataPool.h"
#include "../Field/FieldFactory.h"
#include "../Shader/ShaderFactory.h"
#include "../PtsSampler/PtsSamplerFactory.h"

struct RenderResult {
  using Tensor = torch::Tensor;
  Tensor colors;
  Tensor first_oct_dis;
  Tensor disparity;
  Tensor edge_feats;
  Tensor depth;
  Tensor weights;
  Tensor idx_start_end;
};
/*
struct VolumeRenderInfoPool {
  using Tensor = torch::Tensor;
  // Volume renderer input data
  Tensor sampled_density;
  Tensor sampled_color;
  Tensor bg_color;
  float distortion_weight_;
  // Volume renderer side product
  Tensor cum_density;
  Tensor trans;
  Tensor weight;
};*/

class VolumeRenderInfo : public torch::CustomClassHolder {
public:
  SampleResultFlex* sample_result;
};

class Renderer : public Pipe {
  using Tensor = torch::Tensor;

  enum BGColorType { white, black, rand_noise };
public:
  Renderer(GlobalDataPool* global_data_pool, int n_images);
  RenderResult Render(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds, const Tensor& emb_idx);

  int LoadStates(const std::vector<Tensor>& states, int idx) override;
  std::vector<Tensor> States() override ;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;

  GlobalDataPool* global_data_pool_;
  std::unique_ptr<PtsSampler> pts_sampler_;
  std::unique_ptr<Field> scene_field_;
  std::unique_ptr<Shader> shader_;

  bool use_app_emb_;
  Tensor app_emb_;

  BGColorType bg_color_type_ = BGColorType::rand_noise;

  SampleResultFlex sample_result_;
};

torch::Tensor FilterIdxBounds(const torch::Tensor& idx_bounds,
                              const torch::Tensor& mask);


#endif //SANR_RENDERER_H
