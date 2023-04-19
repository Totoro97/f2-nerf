//
// Created by ppwang on 2022/5/8.
//
#pragma once

#define None torch::indexing::None
#define Slc torch::indexing::Slice

#define CUDAHalf torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
#define CUDAFloat torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
#define CUDALong torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)
#define CUDAInt torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)
#define CUDAUInt8 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)
#define CPUHalf torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU)
#define CPUFloat torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
#define CPULong torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU)
#define CPUInt torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU)
#define CPUUInt8 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)

#ifdef HALF_PRECISION
#define CUDAFlex CUDAHalf
#define CPUFlex CPUHalf
#define FlexType __half
#else
#define CUDAFlex CUDAFloat
#define CPUFlex CPUFloat
#define FlexType float
#endif

#define CHECK_NOT_NAN(x) CHECK(std::isfinite((x).mean().item<float>()))
#define CK_CONT(x) CHECK(x.is_contiguous())

#define PRINT_VAL(x) do { std::cout << #x << " is value " << (x) << std::endl; } while (false)

#define DivUp(x, y)  (((x) + (y) - 1) / (y))
#define ToI64(x) static_cast<long long>(x)
#define THREAD_CAP 512u
#define LIN_BLOCK_DIM(x) { THREAD_CAP, 1, 1 }
#define LIN_GRID_DIM(x) { unsigned(DivUp((x), THREAD_CAP)), 1, 1 }
#define LINEAR_IDX() (blockIdx.x * blockDim.x + threadIdx.x)
#define RE_INTER(x, y) reinterpret_cast<x>(y)

// Use "W" instead of "V" to avoid conflict of OpenCV :-).
#define Wec4d Eigen::Vector4d
#define Wec4f Eigen::Vector4f
#define Wec4i Eigen::Vector4i
#define Wec3d Eigen::Vector3d
#define Wec3f Eigen::Vector3f
#define Wec3i Eigen::Vector3i
#define Wec2d Eigen::Vector2d
#define Wec2f Eigen::Vector2f
#define Wec2i Eigen::Vector2i

#define Watrix22f Eigen::Matrix<float, 2, 2, Eigen::RowMajor>
#define Watrix33f Eigen::Matrix<float, 3, 3, Eigen::RowMajor>
#define Watrix34f Eigen::Matrix<float, 3, 4, Eigen::RowMajor>
#define Watrix43f Eigen::Matrix<float, 4, 3, Eigen::RowMajor>
#define Watrix44f Eigen::Matrix<float, 4, 4, Eigen::RowMajor>

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
