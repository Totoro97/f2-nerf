//
// Created by ppwang on 2022/10/5.
//

#include "CustomOps.h"

namespace torch::autograd {

variable_list TruncExp::forward(AutogradContext *ctx,
                                Tensor input) {
  ctx->save_for_backward( { input });
  return { torch::exp(input) };
}

variable_list TruncExp::backward(AutogradContext *ctx, variable_list grad_output) {
  Tensor x = ctx->get_saved_variables()[0];
  return { grad_output[0] * torch::exp(x.clamp(-100.f, 5.f)) };
}

}