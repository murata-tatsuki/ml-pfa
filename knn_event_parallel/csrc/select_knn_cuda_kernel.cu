// Derived from cms-pepr/pytorch_cmspepr e94c49b; BSD-3-Clause, see ../LICENSE.
// Scheduling and allocation variants share the unchanged per-vertex search.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define I2D(i,j,Nj) j + Nj*i


template <typename scalar_t>
__device__ scalar_t calculateDistance(
    size_t i_v, 
    size_t j_v, 
    const scalar_t *d_coord, 
    size_t n_coords
    ){
    scalar_t distsq = 0;
    // if (i_v == j_v) return 0;
    for (size_t i = 0; i < n_coords; i++) {
        scalar_t dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist * dist;
    }
    return distsq;
    }


template <typename scalar_t>
__device__ int32_t searchLargestDistance(
    int32_t i_v, 
    scalar_t* d_dist, 
    int32_t n_neigh, 
    scalar_t& maxdist
    ){
    maxdist = 0;
    int32_t maxidx = 0;
    if (n_neigh < 2)
        return maxidx;
    for (int32_t n = 1; n < n_neigh; n++) { //0 is self
        scalar_t distsq = d_dist[I2D(i_v, n, n_neigh)];
        bool isgreater = distsq > maxdist;
        bool isless = !isgreater;
        maxdist = distsq*isgreater + maxdist*isless;
        maxidx = n*isgreater + maxidx*isless;
        }
    return maxidx;
}


template <typename scalar_t> 
__global__ void set_defaults(
    scalar_t *d_dist,
    int32_t* d_indices,
    int32_t n_vert,
    int32_t n_neigh)
{
    const int32_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_v >= n_vert) return;

    const int32_t n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= n_neigh) return;

    // Initialize first neighbor as the self loop, other neighbors as -1
    if (n == 0)
        d_indices[I2D(i_v, n, n_neigh)] = i_v;
    else
        d_indices[I2D(i_v, n, n_neigh)] = -1;

    d_dist[I2D(i_v, n, n_neigh)] = 0;
}

template <typename scalar_t, bool Packed>
__global__ void select_knn_kernel(
    const scalar_t *d_coord,
    const int32_t *d_row_splits,
    const int32_t *d_mask,
    int32_t *d_indices,
    scalar_t *d_dist,

    const int32_t n_vert,
    const int32_t n_neigh,
    const int32_t n_coords,

    const scalar_t max_radius,
    const int32_t *tasks,
    const int32_t n_tasks
    ){

    const int32_t j_rs = Packed ? tasks[blockIdx.x] : blockIdx.y;

    //really no buffering at all here

    const int32_t start_vert = d_row_splits[j_rs];
    const int32_t end_vert = d_row_splits[j_rs + 1];

    const int32_t i_v = Packed
        ? tasks[n_tasks + blockIdx.x] + threadIdx.x
        : blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    if (i_v >= end_vert || i_v >= n_vert)
        return;//this will be a problem with actual RS

    //protection against n_vert<n_neigh
    int32_t max_neighbours = n_neigh;
    int32_t nvert_in_row = end_vert - start_vert;
    if (nvert_in_row < n_neigh) { max_neighbours = nvert_in_row; }

    int32_t nfilled = 1;
    int32_t maxidx_local = 0;
    scalar_t maxdistsq = 0;

    int32_t j_v = 0;
    for (j_v = start_vert; j_v < end_vert && nfilled < max_neighbours; j_v++) {
        scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
        if (i_v == j_v || (max_radius > 0 && distsq > max_radius)) continue;
        // if (max_radius > 0 && distsq > max_radius) continue;
        //fill up
        d_indices[I2D(i_v, nfilled, n_neigh)] = j_v;
        d_dist[I2D(i_v, nfilled, n_neigh)] = distsq;
        if (distsq > maxdistsq) {
            maxdistsq = distsq;
            maxidx_local = nfilled;
        }
        nfilled++;
    }
    // for the rest start from where we left off and only do index shuffling
    for(; j_v < end_vert; j_v++ ) {
        scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
        if (i_v == j_v || distsq > maxdistsq) continue;

        //replace former max
        d_indices[I2D(i_v, maxidx_local, n_neigh)] = j_v;
        d_dist[I2D(i_v, maxidx_local, n_neigh)] = distsq;
        //search new max
        maxidx_local = searchLargestDistance(i_v, d_dist, n_neigh, maxdistsq);
    }
}


std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_fn(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor mask,
    int64_t n_neighbours,
    double max_radius,
    int64_t mask_mode)
{
    CHECK_CUDA(coords);
    CHECK_CUDA(row_splits);
    CHECK_CUDA(mask);

    const c10::cuda::CUDAGuard device_guard(coords.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(coords.get_device());
    const auto n_vert = coords.size(0);
    const auto n_coords = coords.size(1);
    const auto n_rs = row_splits.size(0);
    const auto n_neigh = n_neighbours;

    if (max_radius > 0) max_radius *= max_radius;

    auto output_dist_tensor = torch::zeros({ n_vert, n_neighbours },
        torch::TensorOptions().dtype(coords.dtype()).device(coords.device()));
    auto output_idx_tensor = torch::zeros({ n_vert, n_neighbours },
        torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));

    if (n_vert == 0) return std::make_tuple(output_idx_tensor, output_dist_tensor);

    // get the grid and block values for parallel CUDA programming
    // Blocksize 256 over n_vert, blocksize 4 over n_neighbours
    dim3 block(256, 4);
    // Ensure enough blocks in the grid over these dims by rounding up
    dim3 grid((n_vert+block.x-1)/block.x, (n_neigh+block.y-1)/block.y);

    AT_DISPATCH_FLOATING_TYPES(coords.type(), "set_defaults", ([&] {
        set_defaults <scalar_t> <<<grid, block, 0, stream>>> (
            output_dist_tensor.data_ptr<scalar_t>(),
            output_idx_tensor.data_ptr<int32_t>(),
            n_vert,
            n_neigh);
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<int32_t> cpu_rowsplits(n_rs);
    C10_CUDA_CHECK(cudaMemcpyAsync(cpu_rowsplits.data(), row_splits.data_ptr<int32_t>(),
        n_rs * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(cpu_rowsplits.front() == 0 && cpu_rowsplits.back() == n_vert,
                "row_splits must span coords");


    const size_t block_size = 1024; // Deliberately identical to the legacy kernel.
    size_t max_blocks = 0;
    for (int32_t event = 0; event < n_rs - 1; event++) {
        int32_t count = cpu_rowsplits[event + 1] - cpu_rowsplits[event];
        TORCH_CHECK(count >= 0, "row_splits must be nondecreasing");
        max_blocks = std::max(max_blocks, (count + block_size - 1) / block_size);
    }
    // Different events write disjoint rows. Candidate order within a row is unchanged.
    const dim3 event_grid(max_blocks, n_rs - 1);
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "select_knn_kernel", ([&] {
        select_knn_kernel <scalar_t, false> <<<event_grid, block_size, 0, stream>>> (
            coords.data_ptr<scalar_t>(), row_splits.data_ptr<int32_t>(),
            mask.data_ptr<int32_t>(), output_idx_tensor.data_ptr<int32_t>(),
            output_dist_tensor.data_ptr<scalar_t>(), n_vert, n_neigh, n_coords,
            max_radius, nullptr, 0);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output_idx_tensor, output_dist_tensor);

}


// Host launch information is prepared from the CPU Batch.ptr once per batch.
// No device scalar reads, device-to-host copies, or stream synchronization here.
// zero_init=true and packed=false are retained for isolated A/B measurements.
std::tuple<torch::Tensor, torch::Tensor> select_knn_planned_cuda_fn(
    torch::Tensor coords, torch::Tensor row_splits, torch::Tensor mask,
    int64_t n_neighbours, double max_radius, int64_t mask_mode,
    torch::Tensor tasks, int64_t block_size, int64_t max_blocks,
    bool packed, bool zero_init)
{
    const c10::cuda::CUDAGuard device_guard(coords.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(coords.get_device());
    const auto n_vert = coords.size(0);
    auto dist_options = coords.options();
    auto idx_options = coords.options().dtype(torch::kInt32);
    auto distances = zero_init ? torch::zeros({n_vert, n_neighbours}, dist_options)
                               : torch::empty({n_vert, n_neighbours}, dist_options);
    auto indices = zero_init ? torch::zeros({n_vert, n_neighbours}, idx_options)
                             : torch::empty({n_vert, n_neighbours}, idx_options);
    if (n_vert == 0) return std::make_tuple(indices, distances);
    if (max_radius > 0) max_radius *= max_radius;

    // This writes EVERY output element, including padding and the self slot.
    dim3 block(256, 4);
    dim3 grid((n_vert + block.x - 1) / block.x,
              (n_neighbours + block.y - 1) / block.y);
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "set_defaults", ([&] {
        set_defaults<scalar_t><<<grid, block, 0, stream>>>(
            distances.data_ptr<scalar_t>(), indices.data_ptr<int32_t>(),
            n_vert, n_neighbours);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto n_tasks = tasks.size(1);
    const dim3 event_grid(packed ? n_tasks : max_blocks,
                          packed ? 1 : row_splits.numel() - 1);
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "select_knn_kernel", ([&] {
        if (packed) {
            select_knn_kernel<scalar_t, true><<<event_grid, block_size, 0, stream>>>(
                coords.data_ptr<scalar_t>(), row_splits.data_ptr<int32_t>(),
                mask.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<scalar_t>(), n_vert, n_neighbours, coords.size(1),
                max_radius, tasks.data_ptr<int32_t>(), n_tasks);
        } else {
            select_knn_kernel<scalar_t, false><<<event_grid, block_size, 0, stream>>>(
                coords.data_ptr<scalar_t>(), row_splits.data_ptr<int32_t>(),
                mask.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<scalar_t>(), n_vert, n_neighbours, coords.size(1),
                max_radius, nullptr, 0);
        }
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(indices, distances);
}
