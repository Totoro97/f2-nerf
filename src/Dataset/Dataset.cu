//
// Created by ppwang on 2022/9/21.
//

#include "Dataset.h"
#include <torch/torch.h>
#include <Eigen/Eigen>
#include "../Common.h"


using Tensor = torch::Tensor;

template <typename T>
__device__ __host__ inline void apply_camera_distortion(const T* extra_params, const T u, const T v, T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];
  const T p1 = extra_params[2];
  const T p2 = extra_params[3];

  const T u2 = u * u;
  const T uv = u * v;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T radial = k1 * r2 + k2 * r2 * r2;
  *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
  *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

// This implementation is from instant-ngp
template <typename T>
__device__ __host__ inline void iterative_camera_undistortion(const T* params, T* u, T* v) {
  // Parameters for Newton iteration using numerical differentiation with
  // central differences, 100 iterations should be enough even for complex
  // camera models with higher order terms.
  const uint32_t kNumIterations = 100;
  const float kMaxStepNorm = 1e-10f;
  const float kRelStepSize = 1e-6f;

  Eigen::Matrix2f J;
  const Eigen::Vector2f x0(*u, *v);
  Eigen::Vector2f x(*u, *v);
  Eigen::Vector2f dx;
  Eigen::Vector2f dx_0b;
  Eigen::Vector2f dx_0f;
  Eigen::Vector2f dx_1b;
  Eigen::Vector2f dx_1f;

  for (uint32_t i = 0; i < kNumIterations; ++i) {
    const float step0 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(0)));
    const float step1 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(1)));
    apply_camera_distortion(params, x(0), x(1), &dx(0), &dx(1));
    apply_camera_distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    apply_camera_distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    apply_camera_distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    apply_camera_distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
    J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
    const Eigen::Vector2f step_x = J.inverse() * (x + dx - x0);
    x -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }

  *u = x(0);
  *v = x(1);
}

__global__ void CameraUndistortKernel(int n_pixels, const float* params, float* u, float* v) {
  int pix_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pix_idx >= n_pixels) { return; }

  iterative_camera_undistortion<float>(params + pix_idx * 4, u + pix_idx, v + pix_idx);
}

// Input:
//    cam_xy: the xy coordinates in camera sapce **in OpenGL style**
//    dist_params: distortion parameters: k1, k2, p1, p2
Tensor Dataset::CameraUndistort(const Tensor& cam_xy, const Tensor& dist_params) {
  int n_pixels = cam_xy.sizes()[0];
  CHECK_EQ(n_pixels, dist_params.sizes()[0]);
  Tensor u = cam_xy.index({Slc(), 0}).contiguous();
  Tensor v = -cam_xy.index({Slc(), 1}).contiguous();   // OpenGL -> OpenCV

  CK_CONT(dist_params); CK_CONT(u); CK_CONT(v);
  dim3 grid_dim = LIN_GRID_DIM(n_pixels);
  dim3 block_dim = LIN_BLOCK_DIM(n_pixels);
  CameraUndistortKernel<<<grid_dim, block_dim>>>(n_pixels,
                                                 dist_params.data_ptr<float>(),
                                                 u.data_ptr<float>(),
                                                 v.data_ptr<float>());
  return torch::stack({ u, -v }, -1).contiguous();
}


__global__ void Img2WorldRayKernel(int n_rays,
                                   Watrix34f* poses,
                                   Watrix33f* intri,
                                   Wec4f* dist_params,
                                   int* cam_indices,
                                   Wec2f* ij,
                                   Wec3f* out_rays_o,
                                   Wec3f* out_rays_d) {
  int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= n_rays) { return; }

  int cam_idx = cam_indices[ray_idx];
  float i = static_cast<float>(ij[ray_idx][0]);
  float j = static_cast<float>(ij[ray_idx][1]);
  float cx = intri[cam_idx](0, 2);
  float cy = intri[cam_idx](1, 2);
  float fx = intri[cam_idx](0, 0);
  float fy = intri[cam_idx](1, 1);

  float u = (j - cx) / fx;
  float v = (i - cy) / fy;  // OpenCV style
  iterative_camera_undistortion<float>((float*) (dist_params + cam_idx), &u, &v);
  Wec3f dir = {u, -v, -1.f };  // OpenGL style
  out_rays_d[ray_idx] = poses[cam_idx].block<3, 3>(0, 0) * dir;
  out_rays_o[ray_idx] = poses[cam_idx].block<3, 1>(0, 3);
}

Rays Dataset::Img2WorldRayFlex(const Tensor& cam_indices, const Tensor& ij) {
  Tensor ij_shift = (ij + .5f).contiguous();
  CK_CONT(cam_indices);
  CK_CONT(ij_shift);
  CK_CONT(poses_);
  CK_CONT(intri_);
  CK_CONT(dist_params_);
  CHECK_EQ(poses_.sizes()[0], intri_.sizes()[0]);
  CHECK_EQ(cam_indices.sizes()[0], ij.sizes()[0]);

  const int n_rays = cam_indices.sizes()[0];
  dim3 block_dim = LIN_BLOCK_DIM(n_rays);
  dim3 grid_dim  = LIN_GRID_DIM(n_rays);

  Tensor rays_o = torch::zeros({n_rays, 3}, CUDAFloat).contiguous();
  Tensor rays_d = torch::zeros({n_rays, 3}, CUDAFloat).contiguous();

  Img2WorldRayKernel<<<grid_dim, block_dim>>>(n_rays,
                                              RE_INTER(Watrix34f *, poses_.data_ptr()),
                                              RE_INTER(Watrix33f *, intri_.data_ptr()),
                                              RE_INTER(Wec4f*, dist_params_.data_ptr()),
                                              cam_indices.data_ptr<int>(),
                                              RE_INTER(Wec2f*, ij_shift.data_ptr()),
                                              RE_INTER(Wec3f*, rays_o.data_ptr()),
                                              RE_INTER(Wec3f*, rays_d.data_ptr()));

  return { rays_o, rays_d };
}
