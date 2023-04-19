//
// Created by ppwang on 2022/9/16.
//
#pragma once
#include <torch/torch.h>
#include "../Utils/Pipe.h"
#include "../Utils/GlobalDataPool.h"


class Field : public Pipe {
using Tensor = torch::Tensor;

public:
  virtual Tensor Query(const Tensor& coords) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor Query(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor AnchoredQuery(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  GlobalDataPool* global_data_pool_;
};
