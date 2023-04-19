/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   torch_bindings.cu
 *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
 */

#include "TCNNWP.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)
#else
#include <torch/torch.h>
#endif

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>
#include <tiny-cuda-nn/cpp_api.h>
#include <iostream>

#include "../Common.h"

#define CHECK_TS(x) CHECK(x.device().is_cuda()); CHECK(x.is_contiguous())

using Tensor = torch::Tensor;

void* void_data_ptr(torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32: return tensor.data_ptr<float>();
    case torch::kHalf: return tensor.data_ptr<torch::Half>();
    default: throw std::runtime_error{"Unknown precision torch->void"};
  }
}

c10::ScalarType torch_type(tcnn::cpp::EPrecision precision) {
  switch (precision) {
    case tcnn::cpp::EPrecision::Fp32: return torch::kFloat32;
    case tcnn::cpp::EPrecision::Fp16: return torch::kHalf;
    default: throw std::runtime_error{"Unknown precision tcnn->torch"};
  }
}

TORCH_LIBRARY(tcnn_wp, m)
{
  std::cout << "register TCNNWPInfo" << std::endl;
  m.class_<TCNNWPInfo>("TCNNWPInfo").def(torch::init());
}

TCNNWP::TCNNWP(GlobalDataPool* global_data_pool, int d_in, int d_out, int d_hidden, int n_hidden_layers) {
  global_data_pool_ = global_data_pool;
  d_in_ = d_in;
  d_out_ = d_out;
  d_hidden_ = d_hidden;
  n_hidden_layers_ = n_hidden_layers;

  nlohmann::json config = {
      {"otype", "FullyFusedMLP"},
      {"activation", "ReLU"},
      {"output_activation", "None"},
      {"n_neurons", d_hidden},
      {"n_hidden_layers", n_hidden_layers},
  };

  module_ = std::unique_ptr<tcnn::cpp::Module>(tcnn::cpp::create_network(d_in_, d_out_, config));
  Tensor params = torch::zeros({ int(module_->n_params()) }, CUDAFloat);
  size_t seed = 19970826;
  module_->initialize_params(seed, params.data_ptr<float>());
  params_ = params.to(torch::kFloat32);
  params_.requires_grad_(true);
}

Tensor TCNNWP::Query(const Tensor& pts) {
  auto info = torch::make_intrusive<TCNNWPInfo>();

  int batch_size = pts.size(0);
  int batch_size_al = (batch_size + 127) / 128 * 128;
  auto pad_opt = torch::nn::functional::PadFuncOptions({ 0LL, 0LL, 0LL, (long long) (batch_size_al - batch_size)});
  Tensor input = torch::nn::functional::pad(pts, pad_opt);
  tcnn_ctx_.ctx.reset();
  info->tcnn_ = this;
  Tensor feat = torch::autograd::TCNNWPFunction::apply(input, params_.to(torch::kFloat16), torch::IValue(info))[0];
  return feat.index({ Slc(0, batch_size), Slc(0, d_out_)}).to(torch::kFloat32).contiguous();
}

namespace torch::autograd {

variable_list TCNNWPFunction::forward(AutogradContext *ctx,
                                      Tensor input,
                                      Tensor params,
                                      IValue tcnn_info) {
  ctx->set_materialize_grads(false);
  auto info_ptr = tcnn_info.toCustomClass<TCNNWPInfo>();
  ctx->saved_data["tcnn_info"] = tcnn_info;
  auto tcnnwp = info_ptr->tcnn_;

  CHECK_TS(input);
  CHECK_TS(params);

  CHECK_EQ(input.scalar_type(), torch::kFloat32);
  CHECK_EQ(params.scalar_type(), torch_type(tcnnwp->module_->param_precision()));

  CHECK_EQ(input.size(1), tcnnwp->module_->n_input_dims());
  CHECK_EQ(params.size(0), tcnnwp->module_->n_params());

  at::Device device = input.device();
  CHECK_EQ(input.device(), params.device());

  const at::cuda::CUDAGuard device_guard{device};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  uint32_t batch_size = input.size(0);

  torch::Tensor output = torch::rand({ batch_size, tcnnwp->module_->n_output_dims() }, torch::TensorOptions().dtype(
      torch_type(tcnnwp->module_->output_precision())).device(device));

  // CHECK(input.requires_grad());

  tcnn::cpp::Context tcnn_ctx;
  if (!input.requires_grad() && !params.requires_grad()) {
    tcnnwp->module_->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
  }
  else {
    tcnn_ctx = tcnnwp->module_->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params),
                             input.requires_grad());
  }
  // torch::cuda::synchronize();

  CHECK_EQ(output.scalar_type(), torch_type(tcnnwp->module_->output_precision()));
  tcnnwp->query_output_ = output;
  tcnnwp->query_pts_ = input;
  tcnnwp->tcnn_ctx_ = std::move(tcnn_ctx);
  return { output };
}

variable_list TCNNWPFunction::backward(AutogradContext *ctx, variable_list grad_output) {
  auto info_ptr = ctx->saved_data["tcnn_info"].toCustomClass<TCNNWPInfo>();
  auto tcnn_wp = info_ptr->tcnn_;
  float scale = tcnn_wp->loss_scale_;

  if (!tcnn_wp->tcnn_ctx_.ctx) {
    throw std::runtime_error{"Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
  }

  Tensor dL_doutput = grad_output[0] * scale;
  Tensor& input = tcnn_wp->query_pts_;
  Tensor& output = tcnn_wp->query_output_;
  Tensor params = tcnn_wp->params_.to(torch::kFloat16);

  CHECK_TS(input);
  CHECK_TS(params);
  CHECK_TS(output);
  CHECK_TS(dL_doutput);

  CHECK_EQ(input.scalar_type(), torch::kFloat32);
  CHECK_EQ(params.scalar_type(), torch_type(tcnn_wp->module_->param_precision()));
  CHECK_EQ(output.scalar_type(), torch_type(tcnn_wp->module_->output_precision()));
  CHECK_EQ(dL_doutput.scalar_type(), torch_type(tcnn_wp->module_->output_precision()));

  CHECK_EQ(input.size(1), tcnn_wp->module_->n_input_dims());
  CHECK_EQ(output.size(1), tcnn_wp->module_->n_output_dims());
  CHECK_EQ(params.size(0), tcnn_wp->module_->n_params());
  CHECK_EQ(output.size(0), input.size(0));
  CHECK_EQ(dL_doutput.size(0), input.size(0));

  // Device
  at::Device device = input.device();
  CHECK_EQ(device, params.device());
  CHECK_EQ(device, output.device());
  CHECK_EQ(device, dL_doutput.device());

  const at::cuda::CUDAGuard device_guard{device};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  uint32_t batch_size = input.size(0);

  torch::Tensor dL_dinput;
  CHECK(input.requires_grad());
  if (input.requires_grad()) {
    dL_dinput = torch::empty( { batch_size, input.size(1) },
                              torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }

  torch::Tensor dL_dparams;
  dL_dparams = torch::empty({ int(tcnn_wp->module_->n_params()) },
                            torch::TensorOptions().dtype(torch_type(tcnn_wp->module_->param_precision())).device(device));

  tcnn_wp->module_->backward(
      stream,
      tcnn_wp->tcnn_ctx_,
      batch_size,
      input.requires_grad() ?dL_dinput.data_ptr<float>() : nullptr,
      void_data_ptr(dL_doutput),
      void_data_ptr(dL_dparams),
      input.data_ptr<float>(),
      void_data_ptr(output),
      void_data_ptr(params)
  );
  // torch::cuda::synchronize();

  // return { dL_dinput / 128., (dL_dparams).to(torch::kFloat32) / 128., Tensor() };
  dL_dinput = dL_dinput / scale;
  dL_dparams = (dL_dparams).to(torch::kFloat32) / scale;

  if (!torch::all(torch::isfinite(dL_dinput)).item<bool>() ||
      !torch::all(torch::isfinite(dL_dparams)).item<bool>()) {
    // dL_dinput = torch::zeros_like(dL_dinput);
    // dL_dparams = torch::zeros_like(dL_dparams);
    tcnn_wp->global_data_pool_->backward_nan_ = true;
    tcnn_wp->loss_scale_ = std::max(tcnn_wp->loss_scale_ / 2.f, 1.f);
  }

  return { dL_dinput, dL_dparams, Tensor() };
}

};

void TCNNWP::InitParams() {
  size_t seed = 19970826;
  module_->initialize_params(seed, params_.data_ptr<float>());
}

int TCNNWP::LoadStates(const std::vector<Tensor>& states, int idx) {
  CHECK(false) << "This should be handled by the parent module";
  return idx;
}

std::vector<Tensor> TCNNWP::States() {
  CHECK(false) << "This should be handled by the parent module";
  return {};
}

std::vector<torch::optim::OptimizerParamGroup> TCNNWP::OptimParamGroups() {
  CHECK(false) << "This should be handled by the parent module";
  return {};
}

void TCNNWP::Reset() {
  CHECK(false) << "This should be handled by the parent module";
}