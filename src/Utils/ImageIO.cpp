//
// Created by ppwang on 2023/4/4.
//

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <torch/torch.h>
#include "../Common.h"
#include "Utils.h"

using Tensor = torch::Tensor;

Tensor Utils::ReadImageTensor(const std::string& path) {
  int w, h, n;
  unsigned char *idata = stbi_load(path.c_str(), &w, &h, &n, 0);

  Tensor img = torch::empty({ h, w, n }, CPUUInt8);
  std::memcpy(img.data_ptr(), idata, w * h * n);
  stbi_image_free(idata);

  img = img.to(torch::kFloat32).to(torch::kCPU) / 255.f;
  return img;
}

bool Utils::WriteImageTensor(const std::string &path, Tensor img) {
  Tensor out_img = (img * 255.f).clip(0.f, 255.f).to(torch::kUInt8).to(torch::kCPU).contiguous();
  stbi_write_png(path.c_str(), out_img.size(1), out_img.size(0), out_img.size(2), out_img.data_ptr(), 0);
  return true;
}