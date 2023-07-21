//
// Created by jimmy on 22-6-27.
//
// just calculate normalization part, no gamma or bias

#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <iostream>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

__inline__ __device__ float WarpAllReduceSum(float val) {
  for (int mask = 1; mask < 32; mask *= 2) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <typename T>
__global__ void layernorm_inplace_(T *input, long long rows, int cols) {
  int threadidx_x = threadIdx.x / 32;
  int threadidx_y = threadIdx.x % 32;
  long long row_offset = (long long)(blockIdx.x * 4 + threadidx_x);
  int cols_per_thread = (cols + 31) / 32;
  int cols_this_thread = cols_per_thread;

  int last_y = (cols / cols_per_thread);

  if (threadidx_y == last_y) {
    cols_this_thread = cols - cols_per_thread * last_y;
  } else if (threadidx_y > last_y) {
    cols_this_thread = 0;
  }

  float buf[32];
  float f_cols = 1.0f * cols;

  int lane_id = threadidx_y;

  if (row_offset < rows) {
    T *row_input = input + row_offset * cols;
    T *row_output = row_input;

    // read into buffer
#pragma unroll
    for (int i = 0; i < cols_this_thread; i++) {
      int idx = lane_id * cols_per_thread + i;
      buf[i] = static_cast<float>(row_input[idx]);
    }

    float thread_sum = 0.f;
#pragma unroll
    for (int i = 0; i < cols_this_thread; i++) {
      thread_sum += buf[i] / f_cols;
    }

    float mean = WarpAllReduceSum(thread_sum);
    float thread_square_sum = 0.f;

#pragma unroll
    for (int i = 0; i < cols_this_thread; i++) {
      thread_square_sum += (buf[i] - mean) * (buf[i] - mean) / f_cols;
    }
    float warp_var = WarpAllReduceSum(thread_square_sum);
    // warp_var = fdividef(warp_var, f_cols);  // warp_var / f_cols;
    warp_var = warp_var + 0.000001;  // add epsilon
    float warp_std = rsqrtf(warp_var);
    mean = -1.0 * mean * warp_std;

    // write back
#pragma unroll
    for (int i = 0; i < cols_this_thread; i++) {
      row_output[lane_id * cols_per_thread + i] =
          static_cast<T>(fmaf(buf[i], warp_std, mean));
    }
  }
}

void layernorm_inplace_forward_(at::Tensor input, long long rows, int cols) {
  CHECK_INPUT(input);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  int grid = (rows + 3) / 4;
  dim3 block(128);

  if (input.dtype() == torch::kFloat32) {
    layernorm_inplace_<float>
        <<<grid, block>>>((float *)input.data_ptr(), rows, cols);
  } else if (input.dtype() == torch::kFloat16) {
    layernorm_inplace_<c10::Half>
        <<<grid, block>>>((c10::Half *)input.data_ptr(), rows, cols);
  } else {
    layernorm_inplace_<c10::BFloat16>
        <<<grid, block>>>((c10::BFloat16 *)input.data_ptr(), rows, cols);
  }
}
