//
// Created by ppwang on 2023/3/15.
//

#include "CameraUtils.h"
#include <Eigen/Eigen>
#include "../Common.h"

using Tensor = torch::Tensor;

Tensor PoseInterpolate(const Tensor& pose_a, const Tensor& pose_b, float alpha) {
  Tensor a = pose_a.to(torch::kCPU);
  Tensor b = pose_b.to(torch::kCPU);

  auto ToRotMat = [](Tensor ps){
    Eigen::Matrix3f rot;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        rot(i, j) = ps.index({i, j}).item<float>();
      }
    }
    return rot;
  };

  Eigen::Quaternionf rot_a(ToRotMat(a));
  Eigen::Quaternionf rot_b(ToRotMat(b));

  Eigen::Quaternionf rot = rot_a.slerp(alpha, rot_b);

  Eigen::Matrix3f rot_mat = rot.normalized().toRotationMatrix();
  Tensor ret = torch::eye(4, CPUFloat);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ret.index_put_({i, j}, rot_mat(i, j));
    }
  }

  ret.index_put_({Slc(0, 3), 3}, (a * (1.f - alpha) + b * alpha).index({Slc(0, 3), 3}));

  return ret.index({Slc(0, 3), Slc(0, 4)}).contiguous().to(pose_a.device());
}