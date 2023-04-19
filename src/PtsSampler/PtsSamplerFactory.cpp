//
// Created by ppwang on 2022/9/19.
//

#include "PersSampler.h"

std::unique_ptr<PtsSampler> ConstructPtsSampler(GlobalDataPool* global_data_pool) {
  auto type = global_data_pool->config_["pts_sampler"]["type"].as<std::string>();
  if (type == "PersSampler") {
    return std::make_unique<PersSampler>(global_data_pool);
  }
  CHECK(false) << "No such pts sampler.";
}