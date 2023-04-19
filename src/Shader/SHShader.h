//
// Created by ppwang on 2022/10/8.
//

#ifndef SANR_SHSHADER_H
#define SANR_SHSHADER_H

#include "Shader.h"
#include "../Field/TCNNWP.h"

class SHShader : public Shader {
  using Tensor = torch::Tensor;
public:
  SHShader(GlobalDataPool* global_data_pool);
  Tensor Query(const Tensor& feats, const Tensor& dirs) override;
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
  int LoadStates(const std::vector<Tensor>& states, int) override;
  void Reset() override;

  Tensor SHEncode(const Tensor& dirs);

  std::unique_ptr<TCNNWP> mlp_;

  int d_hidden_, n_hiddens_;
  int degree_;
};


#endif //SANR_SHSHADER_H
