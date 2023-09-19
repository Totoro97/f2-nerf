
#include "Utils.h"
#include "happly.h"
#include "../Common.h"
#include "cnpy.h"

using Tensor = torch::Tensor;

void Utils::TensorExportPCD(const std::string &path, Tensor verts)
{
  // Suppose these hold your data
  unsigned n_points = verts.sizes()[0];

  std::vector<std::array<double, 3>> meshVertexPositions(n_points);
  std::vector<std::array<double, 3>> meshVertexColors(n_points);
  std::vector<std::vector<size_t>> meshFaceIndices;

  Tensor verts_cpu = verts.contiguous().to(torch::kF64).to(torch::kCPU);
  double *data_ptr = verts_cpu.data_ptr<double>();

  for (unsigned i = 0; i < n_points; i++)
  {
    meshVertexPositions[i] = {data_ptr[i * 3], data_ptr[i * 3 + 1], data_ptr[i * 3 + 2]};
    meshVertexColors[i] = {0., 0., 0.};
  }

  // Create an empty object
  happly::PLYData plyOut;

  // Add mesh data (elements are created automatically)
  plyOut.addVertexPositions(meshVertexPositions);
  plyOut.addVertexColors(meshVertexColors);
  plyOut.addFaceIndices(meshFaceIndices);

  // Write the object to file
  plyOut.write(path, happly::DataFormat::ASCII);
}

void Utils::TensorExportPCD(const std::string &path,
                            Tensor verts,
                            Tensor vert_colors)
{
  // Suppose these hold your data
  unsigned n_points = verts.sizes()[0];
  CHECK_EQ(n_points, vert_colors.sizes()[0]);
  std::vector<std::array<double, 3>> meshVertexPositions(n_points);
  std::vector<std::array<double, 3>> meshVertexColors(n_points);
  std::vector<std::vector<size_t>> meshFaceIndices;

  Tensor verts_cpu = verts.contiguous().to(torch::kF64).to(torch::kCPU);
  double *data_ptr = verts_cpu.data_ptr<double>();
  Tensor vert_color_cpu = vert_colors.contiguous().to(torch::kF64).to(torch::kCPU);
  double *vert_data_ptr = vert_color_cpu.data_ptr<double>();

  for (unsigned i = 0; i < n_points; i++)
  {
    meshVertexPositions[i] = {data_ptr[i * 3], data_ptr[i * 3 + 1], data_ptr[i * 3 + 2]};
    meshVertexColors[i] = {vert_data_ptr[i * 3], vert_data_ptr[i * 3 + 1], vert_data_ptr[i * 3 + 2]};
  }

  // Create an empty object
  happly::PLYData plyOut;

  // Add mesh data (elements are created automatically)
  plyOut.addVertexPositions(meshVertexPositions);
  plyOut.addVertexColors(meshVertexColors);
  plyOut.addFaceIndices(meshFaceIndices);

  // Write the object to file
  plyOut.write(path, happly::DataFormat::ASCII);
}