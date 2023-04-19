//
// Created by ppwang on 2022/9/16.
//

#include "GlobalDataPool.h"

GlobalDataPool::GlobalDataPool(const std::string &config_path) {
  config_ = YAML::LoadFile(config_path);
  learning_rate_ = config_["train"]["learning_rate"].as<float>();
}