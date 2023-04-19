//
// Created by ppwang on 2022/5/7.
//

#ifndef SANR_DATASET_H
#define SANR_DATASET_H

#pragma once
#include <string>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "../Common.h"
#include "../Utils/GlobalDataPool.h"
#include "../Utils/CameraUtils.h"

#define DATA_TRAIN_SET 1
#define DATA_TEST_SET 2
#define DATA_VAL_SET 4

class Dataset {
  using Tensor = torch::Tensor;
public:
  enum RaySampleMode {
    SINGLE_IMAGE, ALL_IMAGES,
  };
  Dataset(GlobalDataPool* global_data_pool);

  void NormalizeScene();
  void PrepareTrainData(float factor);

  GlobalDataPool* global_data_pool_;

  // Rays
  BoundedRays RaysOfCamera(int idx, int reso_level = 1);
  BoundedRays RaysFromPose(const Tensor& pose, int reso_level = 1);
  BoundedRays RandRaysFromPose(int batch_size, const Tensor& pose);
  BoundedRays RaysInterpolate(int idx_0, int idx_1, float alpha, int reso_level = 1);
  BoundedRays RandRaysWholeSpace(int batch_size);
  std::tuple<BoundedRays, Tensor, Tensor> RandRaysDataOfCamera(int idx, int batch_size);
  std::tuple<BoundedRays, Tensor, Tensor> RandRaysDataOfTrainSet(int batch_size); // Deprecated.
  std::tuple<BoundedRays, Tensor, Tensor> RandRaysData(int batch_size, int sets);

  // Others
  Rays Img2WorldRay(int cam_idx, const Tensor& ij);
  Rays Img2WorldRay(const Tensor& pose, const Tensor& intri, const Tensor& dist_params, const Tensor& ij);
  Rays Img2WorldRayFlex(const Tensor& cam_indices, const Tensor& ij);
  Tensor CameraUndistort(const Tensor& cam_xy, const Tensor& dist_params);

  int n_images_ = 0;
  Tensor poses_, c2w_, w2c_, intri_, dist_params_, bounds_;
  Tensor c2w_train_, w2c_train_, intri_train_, bounds_train_;
  Tensor render_poses_;
  Tensor center_;
  float radius_;

  int height_, width_;
  std::vector<int> train_set_, test_set_, val_set_, split_info_;
  Tensor image_tensors_;
  RaySampleMode ray_sample_mode_;
};

#endif //SANR_DATASET_H