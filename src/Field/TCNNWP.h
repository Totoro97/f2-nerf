//
// Created by ppwang on 2022/10/4.
//

#ifndef SANR_TCNNWP_H
#define SANR_TCNNWP_H

#include "Field.h"
#include <tiny-cuda-nn/cpp_api.h>

class TCNNWP : public Field {
  using Tensor = torch::Tensor;
public:
  TCNNWP(GlobalDataPool* global_data_pool, int d_in, int d_out, int d_hidden, int n_hidden_layers);

  Tensor Query(const Tensor& pts) override;

  int LoadStates(const std::vector<Tensor>& states, int idx) override;
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
  void Reset() override;
  void InitParams();

  int d_in_, d_out_, d_hidden_, n_hidden_layers_;
  std::unique_ptr<tcnn::cpp::Module> module_;

  Tensor params_;
  Tensor query_pts_, query_output_;

  tcnn::cpp::Context tcnn_ctx_;

  float loss_scale_ = 128.f;
};

class TCNNWPInfo : public torch::CustomClassHolder {
public:
  TCNNWP* tcnn_ = nullptr;
};

namespace torch::autograd {

class TCNNWPFunction : public Function<TCNNWPFunction> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor input,
                               Tensor params,
                               IValue TCNNWPInfo);

  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}

#endif //SANR_TCNNWP_H
