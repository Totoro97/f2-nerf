//
// Created by ppwang on 2022/6/20.
//

#pragma once
#include <memory>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include "../Utils/Pipe.h"
#include "../Utils/GlobalDataPool.h"
#include "../Common.h"

struct SampleResultFlex {
  using Tensor = torch::Tensor;
  Tensor pts;                           // [ n_all_pts, 3 ]
  Tensor dirs;                          // [ n_all_pts, 3 ]
  Tensor dt;                            // [ n_all_pts, 1 ]
  Tensor t;                             // [ n_all_pts, 1 ]
  Tensor anchors;                       // [ n_all_pts, 3 ]
  Tensor pts_idx_bounds;                // [ n_rays, 2 ] // start, end
  Tensor first_oct_dis;                 // [ n_rays, 1 ]
};

class PtsSampler : public Pipe {
  using Tensor = torch::Tensor;
public:
  PtsSampler() = default;
  virtual SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
  }
  virtual SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
  }
  virtual std::tuple<Tensor, Tensor> GetEdgeSamples(int n_pts) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor() };
  }

  virtual void UpdateOctNodes(const SampleResultFlex& sample_result,
                              const Tensor& sampled_weights,
                              const Tensor& sampled_alpha) {
    CHECK(false) << "Not implemented";
  }

  GlobalDataPool* global_data_pool_ = nullptr;
};