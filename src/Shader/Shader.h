//
// Created by ppwang on 2022/9/16.
//

#pragma once
#include <torch/torch.h>
#include "../Utils/Pipe.h"
#include "../Utils/GlobalDataPool.h"


class Shader : public Pipe {
  using Tensor = torch::Tensor;
public:
  virtual Tensor Query(const Tensor& feats, const Tensor& dirs) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  GlobalDataPool* global_data_pool_;

  int d_in_, d_out_;
};
