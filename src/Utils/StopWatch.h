//
// Created by ppwang on 2022/5/18.
//
#pragma once
#include <chrono>
#include <string>

class StopWatch {
public:
  StopWatch();
  ~StopWatch() = default;
  double TimeDuration();
  std::chrono::steady_clock::time_point t_point_;
};

class ScopeWatch {
public:
  ScopeWatch(const std::string& scope_name);
  ~ScopeWatch();
  std::chrono::steady_clock::time_point t_point_;
  std::string scope_name_;
};
