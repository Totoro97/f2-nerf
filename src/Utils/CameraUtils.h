//
// Created by ppwang on 2023/3/15.
//

#ifndef SANR_CAMERAUTILS_H
#define SANR_CAMERAUTILS_H

#include <torch/torch.h>

struct alignas(32) Rays {
  torch::Tensor origins;
  torch::Tensor dirs;
};

struct alignas(32) BoundedRays {
  torch::Tensor origins;
  torch::Tensor dirs;
  torch::Tensor bounds;  // near, far
};

torch::Tensor PoseInterpolate(const torch::Tensor& pose_a, const torch::Tensor& pose_b, float alpha);

#endif //SANR_CAMERAUTILS_H

