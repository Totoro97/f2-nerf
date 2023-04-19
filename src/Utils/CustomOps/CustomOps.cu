//
// Created by ppwang on 2023/3/17.
//

#include "CustomOps.h"
#include "../../Common.h"

#define SCALE (16.f)

using Tensor = torch::Tensor;

__global__ void WeightVarLossForwardKernel(int n_outs, float* weights, int* idx_start_end, float* out_vars) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  if (idx_start >= idx_end) {
    out_vars[idx] = 0.f;
    return;
  }
  float mean = 0.f;
  float weight_sum = 1e-6f;
  float len = SCALE;
  for (int i = 0; i + idx_start < idx_end; i++) {
    mean += weights[i + idx_start] * (float(i) / len);
    weight_sum += weights[i + idx_start];
  }
  mean /= weight_sum;
  float variance = 0.f;
  for (int i = 0; i + idx_start < idx_end; i++) {
    float bias = float(i) / len - mean;
    variance += weights[i + idx_start] * bias * bias;
  }
  out_vars[idx] = variance;
}


__global__ void WeightVarLossBackwardKernel(int n_outs, float* weights, int* idx_start_end, float* dl_dvars, float* dl_dw) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  if (idx_start >= idx_end) {
    return;
  }
  float mean = 0.f;
  float weight_sum = 1e-6f;
  float len = SCALE;
  for (int i = 0; i + idx_start < idx_end; i++) {
    mean += weights[i + idx_start] * (float(i) / len);
    weight_sum += weights[i + idx_start];
  }
  mean /= weight_sum;
  float variance = 0.f;
  float tmp = 0.f;
  for (int i = 0; i + idx_start < idx_end; i++) {
    float bias = float(i) / len - mean;
    variance += weights[i + idx_start] * bias * bias;
    tmp += weights[i + idx_start] * 2.f * bias;
  }
  for (int i = 0; i + idx_start < idx_end; i++) {
    float bias = float(i) / len - mean;
    float grad = (bias * bias + tmp * -(float(i) / len) / weight_sum);
    dl_dw[i + idx_start] = dl_dvars[idx] * grad;
  }
}

__global__ void GradientDoorBackwardKernel(int n_rays, int c, float ratio, float* rand_val, int* idx_start_end, float* out_vals) {
  int idx = LINEAR_IDX();
  if (idx >= n_rays) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  for (int i = 0; i + idx_start < idx_end; i++) {
    float cur_ratio = ratio + (1.f - ratio) * (float(i) + .5f) / float(idx_end - idx_start);
    for (int j = 0; j < c; j++) {
      // out_vals[(i + idx_start) * c + j] *= cur_ratio;
      if (rand_val[(i + idx_start) * c + j] > cur_ratio) {
        out_vals[(i + idx_start) * c + j] *= 0.f;
      }
    }
  }
}

namespace torch::autograd {

class WeightVarLoss : public Function<WeightVarLoss> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor weights,
                               Tensor idx_start_end) {
    CK_CONT(weights);
    CK_CONT(idx_start_end);
    int n_outs = idx_start_end.size(0);
    Tensor out_vars = torch::empty({ n_outs }, CUDAFloat);
    dim3 grid_dim  = LIN_GRID_DIM(n_outs);
    dim3 block_dim = LIN_BLOCK_DIM(n_outs);
    WeightVarLossForwardKernel<<<grid_dim, block_dim>>>(n_outs,
                                                        weights.data_ptr<float>(),
                                                        idx_start_end.data_ptr<int>(),
                                                        out_vars.data_ptr<float>());
    ctx->save_for_backward({ weights, idx_start_end });
    return { out_vars };
  }

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_output) {
    Tensor dl_dvar = grad_output[0].contiguous();
    auto saved_tensors = ctx->get_saved_variables();
    Tensor& weights = saved_tensors[0];
    Tensor& idx_start_end = saved_tensors[1];

    int n_outs = idx_start_end.size(0);
    int n_all  = weights.size(0);

    Tensor dl_dw = torch::empty({ n_all }, CUDAFloat);
    dim3 grid_dim  = LIN_GRID_DIM(n_outs);
    dim3 block_dim = LIN_BLOCK_DIM(n_outs);

    WeightVarLossBackwardKernel<<<grid_dim, block_dim>>>(n_outs,
                                                         weights.data_ptr<float>(),
                                                         idx_start_end.data_ptr<int>(),
                                                         dl_dvar.data_ptr<float>(),
                                                         dl_dw.data_ptr<float>());

    return { dl_dw, Tensor() };
  }
};

class GradientDoor : public Function<GradientDoor> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor weights,
                               Tensor idx_start_end,
                               Tensor ratio_ts) {
    CK_CONT(weights);
    CK_CONT(idx_start_end);
    ctx->save_for_backward({ idx_start_end,  ratio_ts });
    return { weights.clone() };
  }

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_output) {
    Tensor dl_dw = grad_output[0].contiguous();
    auto saved_tensors = ctx->get_saved_variables();
    Tensor& idx_start_end = saved_tensors[0];
    float ratio = saved_tensors[1].item<float>();

    int c = 1;
    if (dl_dw.sizes().size() > 1) {
      c = dl_dw.size(1);
    }

    int n_rays = idx_start_end.size(0);

    Tensor out_dl_dw = dl_dw.clone().contiguous();
    Tensor rand_val = torch::rand_like(out_dl_dw, CUDAFloat);

    dim3 grid_dim  = LIN_GRID_DIM(n_rays);
    dim3 block_dim = LIN_BLOCK_DIM(n_rays);

    GradientDoorBackwardKernel<<<grid_dim, block_dim>>>(n_rays, c, ratio,
                                                        rand_val.data_ptr<float>(),
                                                        idx_start_end.data_ptr<int>(),
                                                        out_dl_dw.data_ptr<float>());

    return { out_dl_dw, Tensor(), Tensor() };
  }
};

}

Tensor CustomOps::WeightVar(Tensor weights, Tensor idx_start_end) {
  return torch::autograd::WeightVarLoss::apply(weights.contiguous(), idx_start_end.contiguous())[0];
}

Tensor CustomOps::GradientDoor(torch::Tensor weights, torch::Tensor idx_start_end, float ratio) {
  Tensor ratio_ts = torch::full({1}, ratio, CUDAFloat);
  return torch::autograd::GradientDoor::apply(weights.contiguous(), idx_start_end.contiguous(), ratio_ts)[0];
}

__global__ void DropoutMaskKernel(int n_rays, float alpha, int* idx_start_end, float* rand_vals, float* mask) {
  int ray_idx = LINEAR_IDX();
  if (ray_idx >= n_rays) { return; }
  int idx_start = idx_start_end[ray_idx * 2];
  int idx_end   = idx_start_end[ray_idx * 2 + 1];
  if (idx_start >= idx_end) { return; }
  float len = idx_end - idx_start;
  for (int i = 0; i + idx_start < idx_end; i++) {
    float prob = (.5f + float(i)) / len * alpha + 1.f * (1.f - alpha);
    if (prob + 1e-5f > rand_vals[i + idx_start]) {
      mask[i + idx_start] = 1.f;
    }
    else {
      mask[i + idx_start] = 0.f;
    }
  }
}

Tensor CustomOps::DropoutMask(int n_pts, torch::Tensor idx_start_end, float alpha) {
  int n_rays = idx_start_end.size(0);
  CK_CONT(idx_start_end);

  Tensor rand_vals = torch::rand({ n_pts }, CUDAFloat);
  Tensor mask = torch::rand({ n_pts }, CUDAFloat);

  dim3 grid_dim  = LIN_GRID_DIM(n_rays);
  dim3 block_dim = LIN_BLOCK_DIM(n_rays);

  DropoutMaskKernel<<<grid_dim, block_dim>>>(n_rays, alpha,
                                             idx_start_end.data_ptr<int>(),
                                             rand_vals.data_ptr<float>(),
                                             mask.data_ptr<float>());
  return mask;
}
