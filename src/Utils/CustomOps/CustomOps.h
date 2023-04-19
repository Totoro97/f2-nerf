//
// Created by ppwang on 2022/10/5.
//

#ifndef SANR_CUSTOMOPS_H
#define SANR_CUSTOMOPS_H

#include <torch/torch.h>

namespace torch::autograd {

class TruncExp : public Function<TruncExp> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor input);

  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}

namespace CustomOps {

torch::Tensor WeightVar(torch::Tensor weights, torch::Tensor idx_start_end);
torch::Tensor DropoutMask(int n_rays, torch::Tensor idx_start_end, float alpha);
torch::Tensor GradientDoor(torch::Tensor weights, torch::Tensor idx_start_end, float ratio);

}

#endif //SANR_CUSTOMOPS_H
