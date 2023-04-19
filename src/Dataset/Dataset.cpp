//
// Created by ppwang on 2022/5/7.
//
#include "Dataset.h"
#include <iostream>
#include <fmt/core.h>
#include <experimental/filesystem>
#include "../Utils/cnpy.h"
#include "../Utils/Utils.h"
#include "../Utils/StopWatch.h"

using Tensor = torch::Tensor;

namespace fs = std::experimental::filesystem::v1;

Dataset::Dataset(GlobalDataPool* global_data_pool) {
  ScopeWatch dataset_watch("Dataset::Dataset");
  global_data_pool_ = global_data_pool;
  global_data_pool_->dataset_ = reinterpret_cast<void*>(this);

  const auto& config = global_data_pool_->config_["dataset"];
  const auto data_path = config["data_path"].as<std::string>();
  const auto factor = config["factor"].as<float>();
  const auto ray_sample_mode = config["ray_sample_mode"].as<std::string>();
  if (ray_sample_mode == "single_image") {
    ray_sample_mode_ = RaySampleMode::SINGLE_IMAGE;
  }
  else {
    ray_sample_mode_ = RaySampleMode::ALL_IMAGES;
  }

  // Load camera pose
  CHECK(fs::exists(data_path + "/cams_meta.npy"));
  {
    cnpy::NpyArray arr = cnpy::npy_load(data_path + "/cams_meta.npy");
    auto options = torch::TensorOptions().dtype(torch::kFloat64);  // WARN: Float64 Here!!!!!
    Tensor cam_data = torch::from_blob(arr.data<double>(), arr.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);

    n_images_ = arr.shape[0];
    cam_data = cam_data.reshape({n_images_, 27});
    Tensor poses = cam_data.slice(1, 0, 12).reshape({-1, 3, 4}).contiguous();

    Tensor intri = cam_data.slice(1, 12, 21).reshape({-1, 3, 3}).contiguous();
    intri.index_put_({Slc(), Slc(0, 2), Slc(0, 3)}, intri.index({Slc(), Slc(0, 2), Slc(0, 3)}) / factor);

    Tensor dist_params = cam_data.slice(1, 21, 25).reshape({-1, 4}).contiguous();   // [k1, k2, p1, p2]
    Tensor bounds = cam_data.slice(1, 25, 27).reshape({-1, 2}).contiguous();

    poses_ = poses.to(torch::kCUDA).contiguous();
    intri_ = intri.to(torch::kCUDA).contiguous();
    dist_params_ = dist_params.to(torch::kCUDA).contiguous();
    bounds_ = bounds.to(torch::kCUDA).contiguous();
  }

  NormalizeScene();

  // Load render camera poses
  if (fs::exists(data_path + "/poses_render.npy")) {
    cnpy::NpyArray arr = cnpy::npy_load(data_path + "/poses_render.npy");
    auto options = torch::TensorOptions().dtype(torch::kFloat64);  // WARN: Float64 Here!!!!!
    Tensor cam_data = torch::from_blob(arr.data<double>(), arr.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);

    int n_render_poses = arr.shape[0];
    cam_data = cam_data.reshape({-1, 3, 4});
    cam_data = cam_data.index({Slc(0, n_render_poses)});
    Tensor poses = cam_data;
    render_poses_ = poses;   // [n, 3, 4]
    render_poses_.index_put_({Slc(), Slc(0, 3), 3}, (render_poses_.index({Slc(), Slc(0, 3), 3}) - center_.unsqueeze(0)) / radius_);
    std::cout << "Load render poses" << std::endl;
  }

  // Relax bounds
  auto bounds_factor = config["bounds_factor"].as<std::vector<float>>();
  bounds_ = torch::stack( { bounds_.index({"...", 0}) * bounds_factor[0],
                                 { bounds_.index({"...", 1}) * bounds_factor[1]}}, -1).contiguous();
  bounds_.clamp_(1e-2f, 1e9f);

  global_data_pool_->near_ = bounds_.min().item<float>();
  std::vector<Tensor> images;
  // Load images
  {
    ScopeWatch watch("LoadImages");
    std::ifstream image_list(data_path + "/image_list.txt");
    for (int i = 0; i < n_images_; i++) {
      std::string image_path;
      std::getline(image_list, image_path);
      images.push_back(Utils::ReadImageTensor(image_path).to(torch::kCPU));
    }
  }

  // Load train/test/val split info
  try {
    cnpy::NpyArray sp_arr = cnpy::npy_load(data_path + "/split.npy");
    CHECK_EQ(sp_arr.shape[0], n_images_);

    auto sp_arr_ptr = sp_arr.data<unsigned char>();
    for (int i = 0; i < n_images_; i++) {
      int st = sp_arr_ptr[i];
      split_info_.push_back(st);
      if ((st & 1) == 1) train_set_.push_back(i);
      if ((st & 2) == 2) test_set_.push_back(i);
      if ((st & 4) == 4) val_set_.push_back(i);
    }
  }
  catch (...) {
    for (int i = 0; i < n_images_; i++) {
      if (i % 8 == 0) test_set_.push_back(i);
      else train_set_.push_back(i);
    }
  }
  std::cout << fmt::format("Number of train/test/val images: {}/{}/{}",
                           train_set_.size(), test_set_.size(), val_set_.size()) << std::endl;

  // Get training camera poses
  Tensor train_idx_ts = torch::from_blob(train_set_.data(), { int(train_set_.size()) }, CPUInt).to(torch::kCUDA).to(torch::kLong);
  c2w_train_ = c2w_.index({train_idx_ts}).contiguous();
  w2c_train_ = w2c_.index({train_idx_ts}).contiguous();
  intri_train_ = intri_.index({train_idx_ts}).contiguous();
  bounds_train_ = bounds_.index({train_idx_ts}).contiguous();

  // Prepare training images
  height_ = images[0].size(0);
  width_  = images[0].size(1);
  image_tensors_ = torch::stack(images, 0).to(torch::kCUDA).contiguous();
}

void Dataset::NormalizeScene() {
  // Given poses_ & bounds_, Gen new poses_, c2w_, w2c_, bounds_.
  const auto& config = global_data_pool_->config_["dataset"];
  Tensor cam_pos = poses_.index({Slc(), Slc(0, 3), 3}).clone();
  center_ = cam_pos.mean(0, false);
  Tensor bias = cam_pos - center_.unsqueeze(0);
  radius_ = torch::linalg_norm(bias, 2, -1, false).max().item<float>();
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  poses_.index_put_({Slc(), Slc(0, 3), 3}, cam_pos);

  poses_ = poses_.contiguous();
  c2w_ = poses_.clone();
  w2c_ = torch::eye(4, CUDAFloat).unsqueeze(0).repeat({n_images_, 1, 1}).contiguous();
  w2c_.index_put_({Slc(), Slc(0, 3), Slc()}, c2w_.clone());
  w2c_ = torch::linalg_inv(w2c_);
  w2c_ = w2c_.index({Slc(), Slc(0, 3), Slc()}).contiguous();
  bounds_ = (bounds_ / radius_).contiguous();

  Utils::TensorExportPCD(global_data_pool_->base_exp_dir_ + "/cam_pos.ply", poses_.index({Slc(), Slc(0, 3), 3}));
}

Rays Dataset::Img2WorldRay(int cam_idx, const Tensor &ij) {
  return Img2WorldRay(poses_[cam_idx], intri_[cam_idx], dist_params_[cam_idx], ij);
}

Rays Dataset::Img2WorldRay(const Tensor& pose,
                           const Tensor& intri,
                           const Tensor& dist_param,
                           const Tensor& ij) {
  int n_pts = ij.sizes()[0];
  Tensor i = ij.index({"...", 0}).to(torch::kFloat32) + .5f;
  Tensor j = ij.index({"...", 1}).to(torch::kFloat32) + .5f; // Shift half pixel;

  float cx = intri.index({ 0, 2 }).item<float>();
  float cy = intri.index({ 1, 2 }).item<float>();
  float fx = intri.index({ 0, 0 }).item<float>();
  float fy = intri.index({ 1, 1 }).item<float>();

  Tensor uv = torch::stack( { (j - cx) / fx, -(i - cy) / fy }, -1);
  Tensor params = dist_param.unsqueeze(0).repeat({ n_pts, 1 }).contiguous();
  uv = CameraUndistort(uv, params);
  Tensor dirs = torch::cat({ uv, -torch::ones({n_pts, 1}, CUDAFloat)}, -1);
  Tensor rays_d = torch::matmul(pose.index({ None, Slc(0, 3), Slc(0, 3)}), dirs.index({"...", None}));

  rays_d = rays_d.index({"...", 0});
  Tensor rays_o = pose.index({ None, Slc(0, 3), 3}).repeat({ n_pts, 1 });

  return { rays_o, rays_d };
}

BoundedRays Dataset::RaysOfCamera(int idx, int reso_level) {
  int H = height_;
  int W = width_;
  Tensor ii = torch::linspace(0.f, H - 1.f, H, CUDAFloat);
  Tensor jj = torch::linspace(0.f, W - 1.f, W, CUDAFloat);
  auto ij = torch::meshgrid({ ii, jj }, "ij");
  Tensor i = ij[0].reshape({-1});
  Tensor j = ij[1].reshape({-1});

  float near = bounds_.index({idx, 0}).item<float>();
  float far  = bounds_.index({idx, 1}).item<float>();

  Tensor bounds = torch::stack({
                                   torch::full({ H * W }, near, CUDAFloat),
                                   torch::full({ H * W }, far,  CUDAFloat)
                               }, -1).contiguous();

  auto [ rays_o, rays_d ] = Img2WorldRay(idx, torch::stack({ i, j }, -1));
  return { rays_o, rays_d, bounds };
}

BoundedRays Dataset::RaysFromPose(const Tensor &pose, int reso_level) {
  int H = height_ / reso_level;
  int W = width_ / reso_level;
  Tensor ii = torch::linspace(0.f, height_ - 1.f, H, CUDAFloat);
  Tensor jj = torch::linspace(0.f, width_ - 1.f, W, CUDAFloat);
  auto ij = torch::meshgrid({ ii, jj }, "ij");
  Tensor i = ij[0].reshape({-1});
  Tensor j = ij[1].reshape({-1});

  auto [ rays_o, rays_d ] = Img2WorldRay(pose, intri_[0], dist_params_[0], torch::stack({ i, j }, -1));
  // TODO....
  float near = bounds_.index({Slc(), 0}).min().item<float>();
  float far  = bounds_.index({Slc(), 1}).max().item<float>();

  Tensor bounds = torch::stack({
                                   torch::full({ H * W }, near, CUDAFloat),
                                   torch::full({ H * W }, far,  CUDAFloat)
                               }, -1).contiguous();

  return { rays_o, rays_d, bounds };
}

BoundedRays Dataset::RandRaysFromPose(int batch_size, const Tensor& pose) {
  int H = height_;
  int W = width_;
  Tensor i = torch::randint(0, H, batch_size, CUDALong);
  Tensor j = torch::randint(0, W, batch_size, CUDALong);
  auto [ rays_o, rays_d ] = Img2WorldRay(pose, intri_[0], dist_params_[0], torch::stack({ i, j }, -1).to(torch::kFloat32));
  float near = bounds_.index({Slc(), 0}).min().item<float>();
  float far  = bounds_.index({Slc(), 1}).max().item<float>();

  Tensor bounds = torch::stack({
                                   torch::full({ batch_size }, near, CUDAFloat),
                                   torch::full({ batch_size }, far,  CUDAFloat)
                               }, -1).contiguous();
  return { rays_o, rays_d, bounds };
}


BoundedRays Dataset::RaysInterpolate(int idx_0, int idx_1, float alpha, int reso_level) {
  Tensor pose_0 = poses_[idx_0];
  Tensor pose_1 = poses_[idx_1];
  Tensor pose = PoseInterpolate(pose_0, pose_1, alpha);

  return RaysFromPose(pose, reso_level);
}

BoundedRays Dataset::RandRaysWholeSpace(int batch_size) {
  const int window_size = 10;
  Tensor weights = torch::rand({3}, CPUFloat) + 1e-7f;
  Tensor indices = torch::randint(0, window_size, {3}, CPUInt) + torch::randint(0, n_images_ - window_size, {1}, CPUInt);
  int a = indices[0].item<int>(), b = indices[1].item<int>(), c = indices[2].item<int>();
  float wa = weights[0].item<float>(), wb = weights[1].item<float>(), wc = weights[2].item<float>();
  Tensor pose = PoseInterpolate(poses_[a], poses_[b], wb / (wb + wa));
  pose = PoseInterpolate(pose, poses_[c], wc / (wa + wb + wc));

  return RandRaysFromPose(batch_size, pose);
}

std::tuple<BoundedRays, Tensor, Tensor> Dataset::RandRaysDataOfCamera(int idx, int batch_size) {
  int H = height_;
  int W = width_;
  Tensor i = torch::randint(0, H, batch_size, CUDALong);
  Tensor j = torch::randint(0, W, batch_size, CUDALong);
  auto [ rays_o, rays_d ] = Img2WorldRay(idx, torch::stack({ i, j }, -1).to(torch::kFloat32));
  Tensor gt_colors = image_tensors_[idx].view({-1, 3}).index({ (i * W + j) }).to(torch::kCUDA).contiguous();
  float near = bounds_.index({idx, 0}).item<float>();
  float far  = bounds_.index({idx, 1}).item<float>();

  Tensor bounds = torch::stack({
    torch::full({ H * W }, near, CUDAFloat),
    torch::full({ H * W }, far,  CUDAFloat)
  }, -1).contiguous();
  return { { rays_o, rays_d, bounds }, gt_colors, torch::full({ batch_size }, idx, CUDAInt) };
}


std::tuple<BoundedRays, Tensor, Tensor> Dataset::RandRaysData(int batch_size, int sets) {
  std::vector<int> img_idx;
  if ((sets & DATA_TRAIN_SET) != 0) {
    img_idx.insert(img_idx.end(), train_set_.begin(), train_set_.end());
  }
  if ((sets & DATA_VAL_SET) != 0) {
    img_idx.insert(img_idx.end(), val_set_.begin(), val_set_.end());
  }
  if ((sets & DATA_TEST_SET) != 0) {
    img_idx.insert(img_idx.end(), test_set_.begin(), test_set_.end());
  }
  Tensor cur_set = torch::from_blob(img_idx.data(), { int(img_idx.size())}, CPUInt).to(torch::kCUDA);
  Tensor cam_indices = torch::randint(int(img_idx.size()), { batch_size }, CUDALong); // Torch index need "long long" type
  cam_indices = cur_set.index({cam_indices}).contiguous();
  Tensor i = torch::randint(0, height_, batch_size, CUDALong);
  Tensor j = torch::randint(0, width_, batch_size, CUDALong);
  Tensor ij = torch::stack({i, j}, -1).contiguous();

  Tensor gt_colors = image_tensors_.view({-1, 3}).index({ (cam_indices * height_ * width_ + i * width_ + j).to(torch::kLong) }).contiguous();
  auto [ rays_o, rays_d ] = Img2WorldRayFlex(cam_indices.to(torch::kInt32), ij.to(torch::kInt32));
  Tensor bounds = bounds_.index({cam_indices.to(torch::kLong)}).contiguous();

  return { { rays_o, rays_d, bounds }, gt_colors, cam_indices.to(torch::kInt32).contiguous() };
}

std::tuple<BoundedRays, Tensor, Tensor> Dataset::RandRaysDataOfTrainSet(int batch_size) {
  if (ray_sample_mode_ == RaySampleMode::SINGLE_IMAGE) {
    int idx = torch::randint(int(train_set_.size()), {1}).item<int>();
    idx = train_set_[idx];
    return RandRaysDataOfCamera(idx, batch_size);
  }
  else {
    return RandRaysData(batch_size, DATA_TRAIN_SET);
  }
}
