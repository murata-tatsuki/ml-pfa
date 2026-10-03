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

std::tuple<torch::Tensor, torch::Tensor> select_knn_planned_cuda_fn(
    torch::Tensor coords, torch::Tensor row_splits, torch::Tensor mask,
    int64_t n_neighbours, double max_radius, int64_t mask_mode,
    torch::Tensor tasks, int64_t block_size, int64_t max_blocks,
    bool packed, bool zero_init);

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


std::tuple<torch::Tensor, torch::Tensor> select_knn_planned_cuda_interface(
    torch::Tensor coords, torch::Tensor row_splits, torch::Tensor mask,
    int64_t n_neighbours, double max_radius, int64_t mask_mode,
    torch::Tensor tasks, int64_t block_size, int64_t max_blocks,
    bool packed, bool zero_init) {
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

  CHECK_INPUT(tasks);
  TORCH_CHECK(tasks.device() == coords.device() && tasks.scalar_type() == torch::kInt32,
              "tasks must be int32 on the coordinate device");
  TORCH_CHECK(tasks.dim() == 2 && tasks.size(0) == 2 && tasks.size(1) <= INT32_MAX,
              "tasks must be 2 x nblocks");
  TORCH_CHECK(block_size == 128 || block_size == 256 || block_size == 512 || block_size == 1024,
              "block_size must be 128, 256, 512 or 1024");
  TORCH_CHECK(max_blocks >= 0 && max_blocks <= INT32_MAX,
              "invalid maximum block count");
  TORCH_CHECK(coords.size(0) == 0 || (packed ? tasks.size(1) > 0 : max_blocks > 0),
              "nonempty coordinates require a nonempty launch plan");
  // Tensor contents are validated on CPU by prepare_plan, before upload.
  return select_knn_planned_cuda_fn(coords, row_splits, mask, n_neighbours,
      max_radius, mask_mode, tasks, block_size, max_blocks, packed, zero_init);
}

TORCH_LIBRARY(pfa_knn_event_parallel, m) {
  m.def("select_knn_cuda", select_knn_cuda_interface);
  m.def("select_knn_planned_cuda", select_knn_planned_cuda_interface);
}