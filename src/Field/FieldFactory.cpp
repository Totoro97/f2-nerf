//
// Created by ppwang on 2022/9/16.
//
#include "FieldFactory.h"
#include "Hash3DAnchored.h"

std::unique_ptr<Field> ConstructField(GlobalDataPool* global_data_pool) {
  auto type = global_data_pool->config_["field"]["type"].as<std::string>();
  if (type == "Hash3DAnchored") {
    return std::make_unique<Hash3DAnchored>(global_data_pool);
  }
  else {
    CHECK(false) << "There is no such field type.";
    return {};
  }
}