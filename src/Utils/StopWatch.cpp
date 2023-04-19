//
// Created by ppwang on 2022/5/18.
//

#include "StopWatch.h"
#include <iostream>
#include <torch/torch.h>

StopWatch::StopWatch() {
  t_point_ = std::chrono::steady_clock::now();
}

double StopWatch::TimeDuration() {
  std::chrono::steady_clock::time_point new_point = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(new_point - t_point_);
  t_point_ = new_point;
  return time_span.count();
}

ScopeWatch::ScopeWatch(const std::string& scope_name) : scope_name_(scope_name){
  torch::cuda::synchronize();
  t_point_ = std::chrono::steady_clock::now();
  std::cout << "[" << scope_name_ << "] begin" << std::endl;
}

ScopeWatch::~ScopeWatch() {
  torch::cuda::synchronize();
  std::chrono::steady_clock::time_point new_point = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(new_point - t_point_);
  std::cout << "[" << scope_name_ << "] end in " << time_span.count() << " seconds" << std::endl;
}