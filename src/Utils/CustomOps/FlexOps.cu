#include "FlexOps.h"

using Tensor = torch::Tensor;

__global__ void FlexSumForwardKernel(int n_outs, float* val, int* idx_start_end, float* sum) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  float out_val = 0.f;
  for (int i = idx_start; i < idx_end; i++) {
    out_val += val[i];
  }
  sum[idx] = out_val;
}

__global__ void FlexSumBackwardKernel(int n_outs, float* dl_dsum, int* idx_start_end, float* dl_dval) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  float fill_val = dl_dsum[idx];
  for (int i = idx_start; i < idx_end; i++) {
    dl_dval[i] = fill_val;
  }
}

__global__ void FlexSumVecForwardKernel(int n_outs, int vec_size, float* val, int* idx_start_end, float* sum) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  for (int j = 0; j < vec_size; j++) {
    float out_val = 0.f;
    for (int i = idx_start; i < idx_end; i++) {
      out_val += val[i * vec_size + j];
    }
    sum[idx * vec_size + j] = out_val;
  }
}

__global__ void FlexSumVecBackwardKernel(int n_outs, int vec_size, float* dl_dsum, int* idx_start_end, float* dl_dval) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  for (int j = 0; j < vec_size; j++) {
    float fill_val = dl_dsum[idx * vec_size + j];
    for (int i = idx_start; i < idx_end; i++) {
      dl_dval[i * vec_size + j] = fill_val;
    }
  }
}

__global__ void FlexAccumulateSumForwardKernel(int n_outs, bool include_this, float* val, int* idx_start_end, float* sum) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  float out_val = 0.f;
  if (include_this) {
    for (int i = idx_start; i < idx_end; i++) {
      out_val += val[i];
      sum[i] = out_val;
    }
  }
  else {
    for (int i = idx_start; i < idx_end; i++) {
      sum[i] = out_val;
      out_val += val[i];
    }
  }
}

__global__ void FlexAccumulateSumBackwardKernel(int n_outs, bool include_this, float* dl_dsum, int* idx_start_end, float* dl_dval) {
  int idx = LINEAR_IDX();
  if (idx >= n_outs) return;
  int idx_start = idx_start_end[idx * 2];
  int idx_end   = idx_start_end[idx * 2 + 1];
  float wp = 0.f;
  if (include_this) {
    for (int i = idx_end - 1; i >= idx_start; i--) {
      wp += dl_dsum[i];
      dl_dval[i] = wp;
    }
  }
  else {
    for (int i = idx_end - 1; i >= idx_start; i--) {
      dl_dval[i] = wp;
      wp += dl_dsum[i];
    }
  }
}

namespace torch::autograd {

class FlexSum : public Function<FlexSum> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor val,
                               Tensor idx_start_end) {
    CK_CONT(val);
    CK_CONT(idx_start_end);
    int n_outs = idx_start_end.size(0);
    Tensor sum;
    dim3 grid_dim  = LIN_GRID_DIM(n_outs);
    dim3 block_dim = LIN_BLOCK_DIM(n_outs);

    if (val.sizes().size() == 1) {
      sum = torch::empty({ n_outs }, CUDAFloat);
      FlexSumForwardKernel<<<grid_dim, block_dim>>>(
          n_outs, val.data_ptr<float>(), idx_start_end.data_ptr<int>(), sum.data_ptr<float>());
    }
    else {
      int vec_size = val.size(1);
      sum = torch::empty({ n_outs, vec_size }, CUDAFloat);
      FlexSumVecForwardKernel<<<grid_dim, block_dim>>>(
          n_outs, vec_size,
          val.data_ptr<float>(), idx_start_end.data_ptr<int>(), sum.data_ptr<float>());
    }
    ctx->save_for_backward({ val, idx_start_end });
    return { sum };
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
    Tensor dl_dsum = grad_output[0].contiguous();
    auto saved_tensors = ctx->get_saved_variables();
    Tensor& val = saved_tensors[0];
    Tensor& idx_start_end = saved_tensors[1];
    int n_outs = idx_start_end.size(0);
    int n_all  = val.size(0);

    Tensor dl_dval;
    dim3 grid_dim  = LIN_GRID_DIM(n_outs);
    dim3 block_dim = LIN_BLOCK_DIM(n_outs);

    if (val.sizes().size() == 1) {
      dl_dval = torch::empty({ n_all }, CUDAFloat);
      FlexSumBackwardKernel<<<grid_dim, block_dim>>>(
          n_outs, dl_dsum.data_ptr<float>(), idx_start_end.data_ptr<int>(), dl_dval.data_ptr<float>());

    }
    else {
      int vec_size = val.size(1);
      dl_dval = torch::empty({ n_all, vec_size }, CUDAFloat);
      FlexSumVecBackwardKernel<<<grid_dim, block_dim>>>(
          n_outs, vec_size,
          dl_dsum.data_ptr<float>(), idx_start_end.data_ptr<int>(), dl_dval.data_ptr<float>());
    }
    return { dl_dval, Tensor() };
  }
};

class FlexAccumulateSum : public Function<FlexAccumulateSum> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor val,
                               Tensor idx_start_end,
                               torch::IValue include_this_ivalue) {
    CK_CONT(val);
    CK_CONT(idx_start_end);
    bool include_this = include_this_ivalue.toBool();
    int n_all = val.size(0);
    int n_outs = idx_start_end.size(0);
    Tensor sum = torch::empty({ n_all }, CUDAFloat);
    dim3 grid_dim  = LIN_GRID_DIM(n_outs);
    dim3 block_dim = LIN_BLOCK_DIM(n_outs);

    FlexAccumulateSumForwardKernel<<<grid_dim, block_dim>>>(
        n_outs, include_this,
        val.data_ptr<float>(), idx_start_end.data_ptr<int>(), sum.data_ptr<float>());

    ctx->save_for_backward({ val, idx_start_end });
    ctx->saved_data["include_this"] = include_this_ivalue;
    return { sum };
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
    Tensor dl_dsum = grad_output[0].contiguous();

    auto saved_tensors = ctx->get_saved_variables();
    bool include_this = ctx->saved_data["include_this"].toBool();
    Tensor& val = saved_tensors[0];
    Tensor& idx_start_end = saved_tensors[1];
    int n_outs = idx_start_end.size(0);
    int n_all  = val.size(0);

    Tensor dl_dval = torch::empty({ n_all }, CUDAFloat);
    dim3 grid_dim  = LIN_GRID_DIM(n_outs);
    dim3 block_dim = LIN_BLOCK_DIM(n_outs);

    FlexAccumulateSumBackwardKernel<<<grid_dim, block_dim>>>(
        n_outs, include_this,
        dl_dsum.data_ptr<float>(), idx_start_end.data_ptr<int>(), dl_dval.data_ptr<float>());

    return { dl_dval, Tensor(), Tensor() };
  }
};

}


namespace FlexOps {

Tensor Sum(Tensor val, Tensor idx_start_end) {
  return torch::autograd::FlexSum::apply(val.contiguous(), idx_start_end.contiguous())[0];
}

Tensor AccumulateSum(Tensor val, Tensor idx_start_end, bool include_this) {
  return torch::autograd::FlexAccumulateSum::apply(val.contiguous(),
                                                   idx_start_end.contiguous(),
                                                   torch::IValue(include_this))[0];
}

}