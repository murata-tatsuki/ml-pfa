// Derived from cms-pepr/pytorch_cmspepr e94c49b; BSD-3-Clause, see ../LICENSE.
#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <climits>

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_fn(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
    );

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_interface(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
    ){
  CHECK_INPUT(coords);
  CHECK_INPUT(row_splits);
  CHECK_INPUT(mask);
  TORCH_CHECK(coords.dim() == 2 && coords.size(1) > 0, "coords must be N x D");
  TORCH_CHECK(coords.scalar_type() == torch::kFloat32 || coords.scalar_type() == torch::kFloat64,
              "coords must be FP32 or FP64");
  TORCH_CHECK(row_splits.dim() == 1 && row_splits.numel() >= 2 && row_splits.numel() <= 65536,
              "row_splits must describe 1..65535 events");
  TORCH_CHECK(row_splits.scalar_type() == torch::kInt32 && mask.scalar_type() == torch::kInt32,
              "row_splits and mask must be int32");
  TORCH_CHECK(coords.device() == row_splits.device() && coords.device() == mask.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(mask.numel() == coords.size(0) && n_neighbours > 0, "invalid mask size or k");
  TORCH_CHECK(coords.size(0) <= INT32_MAX && n_neighbours <= INT32_MAX / std::max<int64_t>(1, coords.size(0)),
              "indices exceed int32 capacity");
  return select_knn_cuda_fn(
    coords, row_splits, mask, n_neighbours, max_radius, mask_mode
    );
}

TORCH_LIBRARY(pfa_knn_event_parallel, m) {
  m.def("select_knn_cuda", select_knn_cuda_interface);
}