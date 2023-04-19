//
// Created by ppwang on 2022/5/11.
//

#include <string>
#include <torch/torch.h>

namespace Utils {
using Tensor = torch::Tensor;

void TensorExportPCD(const std::string& path, Tensor verts);
void TensorExportPCD(const std::string& path, Tensor verts, Tensor vert_colors);
int SaveVectorAsNpy(const std::string& path, std::vector<float> data);

Tensor ReadImageTensor(const std::string& path);
bool WriteImageTensor(const std::string& path, Tensor img);

}