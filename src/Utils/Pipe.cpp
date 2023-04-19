//
// Created by ppwang on 2022/9/18.
//

#include "Pipe.h"

using Tensor = torch::Tensor;

int Pipe::LoadStates(const std::vector<Tensor>& states, int idx) {
  for (auto pipe : sub_pipes_) {
    idx = pipe->LoadStates(states, idx);
  }
  return idx;
}

std::vector<Tensor> Pipe::States() {
  std::vector<Tensor> ret;
  for (auto pipe : sub_pipes_) {
    auto cur_states = pipe->States();
    ret.insert(ret.end(), cur_states.begin(), cur_states.end());
  }
  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> Pipe::OptimParamGroups() {
  std::vector<torch::optim::OptimizerParamGroup> ret;
  for (auto pipe : sub_pipes_) {
    auto cur_params = pipe->OptimParamGroups();
    for (const auto& para_group : cur_params) {
      ret.emplace_back(para_group);
    }
  }
  return ret;
}

void Pipe::RegisterSubPipe(Pipe* sub_pipe) {
  sub_pipes_.push_back(sub_pipe);
}

void Pipe::Reset() {
  for (auto pipe : sub_pipes_) {
    pipe->Reset();
  }
}