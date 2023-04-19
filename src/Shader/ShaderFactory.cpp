//
// Created by ppwang on 2022/9/16.
//

#include "ShaderFactory.h"
#include "SHShader.h"

std::unique_ptr<Shader> ConstructShader(GlobalDataPool* global_data_pool) {
  auto type = global_data_pool->config_["shader"]["type"].as<std::string>();
  if (type == "SHShader") {
    return std::make_unique<SHShader>(global_data_pool);
  }
  else {
    CHECK(false) << "There is no such shader type.";
    return {};
  }
}
