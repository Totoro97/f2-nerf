//
// Created by ppwang on 2022/9/18.
//
#pragma once
#include <vector>
#include <torch/torch.h>

class Pipe {
  using Tensor = torch::Tensor;

public:
  virtual int LoadStates(const std::vector<Tensor>& states, int idx);
  virtual std::vector<Tensor> States();
  virtual std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups();
  virtual void Reset();
  void RegisterSubPipe(Pipe* sub_pipe);

  std::vector<Pipe*> sub_pipes_;
};
