#include "Renderer.h"
#include <torch/torch.h>
#include <Eigen/Eigen>
#include "../Common.h"

using Tensor = torch::Tensor;

__global__ void CountValidPts(int n_rays, int* idx_bounds, int* mask, int* out_cnt) {
  int ray_idx = LINEAR_IDX();
  if (ray_idx >= n_rays) return;
  int idx_start = idx_bounds[ray_idx * 2];
  int idx_end   = idx_bounds[ray_idx * 2 + 1];
  int cnt = 0;
  for (int i = idx_start; i < idx_end; i++) {
    cnt += mask[i];
  }
  out_cnt[ray_idx] = cnt;
}

torch::Tensor FilterIdxBounds(const torch::Tensor& idx_bounds,
                              const torch::Tensor& mask) {
  int n_rays = idx_bounds.size(0);
  // Tensor sorted_idx_bounds, sorted_idx;
  // std::tie(sorted_idx_bounds, sorted_idx) = torch::sort(idx_bounds, true, 0, false);
  // CHECK(torch::equal(sorted_idx.index({Slc(), 0}), sorted_idx.index({Slc(), 1})));
  // sorted_idx = sorted_idx.index({Slc(), 0}).contiguous();
  // Tensor inv_idx = torch::zeros({ n_rays }, CUDALong);
  // {
  //   dim3 grid_dim = LIN_GRID_DIM(n_rays);
  //   dim3 block_dim = LIN_BLOCK_DIM(n_rays);
  //   InvPermutation<<<grid_dim, block_dim>>>(n_rays, inv_idx.data_ptr<int64_t>(), sorted_idx.data_ptr<int64_t>());
  // }

  // Tensor sorted_idx_bounds = idx_bounds.clone().contiguous();
  Tensor mask_int = mask.to(torch::kInt32).contiguous();

  Tensor valid_cnt = torch::zeros({ n_rays }, CUDAInt);
  dim3 grid_dim = LIN_GRID_DIM(n_rays);
  dim3 block_dim = LIN_BLOCK_DIM(n_rays);
  CountValidPts<<<grid_dim, block_dim>>>(n_rays,
                                         idx_bounds.data_ptr<int>(),
                                         mask_int.data_ptr<int>(),
                                         valid_cnt.data_ptr<int>());
  valid_cnt = torch::cumsum(valid_cnt, 0);
  Tensor new_idx_bounds = torch::zeros({ n_rays, 2 }, CUDAInt);
  new_idx_bounds.index_put_({Slc(), 1}, valid_cnt);
  new_idx_bounds.index_put_({Slc(1, None), 0}, valid_cnt.index({Slc(0, -1)}));
  return new_idx_bounds;
  // return new_idx_bounds.index({inv_idx}).contiguous();
}
