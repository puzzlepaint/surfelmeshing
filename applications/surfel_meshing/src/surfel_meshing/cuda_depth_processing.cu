// Copyright 2018 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


// Avoid warnings in Qt includes with CUDA compiler
#pragma GCC diagnostic ignored "-Wattributes"
// Avoid warnings in Eigen includes with CUDA compiler
#pragma diag_suppress code_is_unreachable

#include "surfel_meshing/cuda_depth_processing.cuh"

#include <libvis/cuda/cuda_util.h>
#include <math_constants.h>

#include "surfel_meshing/cuda_util.cuh"

// Uncomment this to run CUDA kernels sequentially for debugging.
// #define CUDA_SEQUENTIAL_CHECKS
#ifdef WIN32
#define M_PI       3.14159265358979323846
#endif

namespace vis {

__global__ void BilateralFilteringAndDepthCutoffCUDAKernel(
    float denom_xy,
    float sigma_value_factor,
    int radius,
    int radius_squared,
    u16 value_to_ignore,
    u16 max_depth,
    float depth_valid_region_radius_squared,
    CUDABuffer_<u16> input_depth,
    CUDABuffer_<u16> output_depth) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < output_depth.width() && y < output_depth.height()) {
    int image_half_width = output_depth.width() / 2;
    int image_half_height = output_depth.height() / 2;
    float center_distance_squared =
        (x - image_half_width) * (x - image_half_width) +
        (y - image_half_height) * (y - image_half_height);
    if (center_distance_squared > depth_valid_region_radius_squared) {
      output_depth(y, x) = value_to_ignore;
      return;
    }
    
    // Depth cutoff.
    u16 center_value = input_depth(y, x);
    if (center_value == value_to_ignore || center_value > max_depth) {
      output_depth(y, x) = value_to_ignore;
      return;
    }
    
    // Bilateral filtering.
    const float adapted_sigma_value = center_value * sigma_value_factor;
    const float adapted_denom_value = 2.0f * adapted_sigma_value * adapted_sigma_value;
    
    float sum = 0;
    float weight = 0;
    
    const int min_y = max(static_cast<int>(0), static_cast<int>(y - radius));
    const int max_y = min(static_cast<int>(output_depth.height() - 1), static_cast<int>(y + radius));
    for (int sample_y = min_y; sample_y <= max_y; ++ sample_y) {
      const int dy = sample_y - y;
      
      const int min_x = max(static_cast<int>(0), static_cast<int>(x - radius));
      const int max_x = min(static_cast<int>(output_depth.width() - 1), static_cast<int>(x + radius));
      for (int sample_x = min_x; sample_x <= max_x; ++ sample_x) {
        const int dx = sample_x - x;
        
        const int grid_distance_squared = dx * dx + dy * dy;
        if (grid_distance_squared > radius_squared) {
          continue;
        }
        
        u16 sample = input_depth(sample_y, sample_x);
        if (sample == value_to_ignore) {
          continue;
        }
        
        float value_distance_squared = center_value - sample;
        value_distance_squared *= value_distance_squared;
        float w = exp(-grid_distance_squared / denom_xy + -value_distance_squared / adapted_denom_value);
        sum += w * sample;
        weight += w;
      }
    }
    
    output_depth(y, x) = (weight == 0) ? value_to_ignore : (sum / weight + 0.5f);
  }
}

void BilateralFilteringAndDepthCutoffCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value_factor,
    u16 value_to_ignore,
    float radius_factor,
    u16 max_depth,
    float depth_valid_region_radius,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  int radius = radius_factor * sigma_xy + 0.5f;
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(output_depth->width(), kBlockWidth),
                GetBlockCount(output_depth->height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  BilateralFilteringAndDepthCutoffCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      2.0f * sigma_xy * sigma_xy,
      sigma_value_factor,
      radius,
      radius * radius,
      value_to_ignore,
      max_depth,
      depth_valid_region_radius * depth_valid_region_radius,
      input_depth,
      *output_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


// Helper to pass arrays to the kernel.
template <int count, typename DepthT>
struct OutlierDepthMapFusionCUDAKernelParam {
  CUDAMatrix3x4 other_TR_reference[count - 1];
  CUDABuffer_<DepthT> other_depths[count - 1];
};

template <int count, typename DepthT>
__global__ void OutlierDepthMapFusionCUDAKernel(
    float max_tolerance_factor,
    float min_tolerance_factor,
    CUDABuffer_<DepthT> input_depth,
    float fx, float fy, float cx, float cy,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    OutlierDepthMapFusionCUDAKernelParam<count, DepthT> p,
    CUDABuffer_<DepthT> output_depth) {
  constexpr int kOtherCount = count - 1;
  
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < output_depth.width() && y < output_depth.height()) {
    DepthT depth_value = input_depth(y, x);
    if (depth_value == 0) {
      output_depth(y, x) = 0;
      return;
    }
    
    float3 reference_point =
        make_float3(depth_value * (fx_inv * x + cx_inv),
                    depth_value * (fy_inv * y + cy_inv),
                    depth_value);
    
    // Project the pixel into the other depth maps to verify that there are enough close other depth estimates.
    bool ok = true;
    for (int other_index = 0; other_index < kOtherCount; ++ other_index) {
      float3 other_point = p.other_TR_reference[other_index] * reference_point;
      
      if (other_point.z <= 0) {
        ok = false;
        break;
      }
      
      float2 pixel_pos =
          make_float2(fx * (other_point.x / other_point.z) + cx,
                      fy * (other_point.y / other_point.z) + cy);
      // TODO: for pixel_pos.x or .y in ]-1, 0] this will also treat the pixel as in the image
      int px = static_cast<int>(pixel_pos.x);
      int py = static_cast<int>(pixel_pos.y);
      if (px < 0 || py < 0 ||
          px >= output_depth.width() || py >= output_depth.height()) {
        ok = false;
        break;
      }
      
      DepthT other_depth_value = p.other_depths[other_index](py, px);
      if (other_depth_value <= 0 ||
          other_depth_value > max_tolerance_factor * other_point.z ||
          other_depth_value < min_tolerance_factor * other_point.z) {
        ok = false;
        break;
      }
    }
    
    output_depth(y, x) = ok ? depth_value : 0;
  }
}

template <int count, typename DepthT>
void OutlierDepthMapFusionCUDA(
    cudaStream_t stream,
    float tolerance,
    const CUDABuffer_<DepthT>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<DepthT>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kOtherCount = count - 1;
  
  OutlierDepthMapFusionCUDAKernelParam<count, DepthT> p;
  for (int i = 0; i < kOtherCount; ++ i) {
    p.other_TR_reference[i] = others_TR_reference[i];
    
    p.other_depths[i] = *other_depths[i];
  }
  
  const float max_tolerance_factor = 1 + tolerance;
  const float min_tolerance_factor = 1 - tolerance;
  
  // Unprojection intrinsics for pixel center convention.
  const float fx_inv = 1.0f / depth_fx;
  const float fy_inv = 1.0f / depth_fy;
  const float cx_pixel_center = depth_cx - 0.5f;
  const float cy_pixel_center = depth_cy - 0.5f;
  const float cx_inv_pixel_center = -cx_pixel_center / depth_fx;
  const float cy_inv_pixel_center = -cy_pixel_center / depth_fy;
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(output_depth->width(), kBlockWidth),
                GetBlockCount(output_depth->height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  OutlierDepthMapFusionCUDAKernel<count, DepthT>
  <<<grid_dim, block_dim, 0, stream>>>(
      max_tolerance_factor,
      min_tolerance_factor,
      input_depth,
      depth_fx, depth_fy, depth_cx, depth_cy,
      fx_inv, fy_inv, cx_inv_pixel_center, cy_inv_pixel_center,
      p,
      *output_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

template
void OutlierDepthMapFusionCUDA<9, u16>(
    cudaStream_t stream,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);
template
void OutlierDepthMapFusionCUDA<7, u16>(
    cudaStream_t stream,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);
template
void OutlierDepthMapFusionCUDA<5, u16>(
    cudaStream_t stream,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);
template
void OutlierDepthMapFusionCUDA<3, u16>(
    cudaStream_t stream,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);


template <int count, typename DepthT>
__global__ void OutlierDepthMapFusionCUDAKernel(
    int required_count,
    float max_tolerance_factor,
    float min_tolerance_factor,
    CUDABuffer_<DepthT> input_depth,
    float fx, float fy, float cx, float cy,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    OutlierDepthMapFusionCUDAKernelParam<count, DepthT> p,
    CUDABuffer_<DepthT> output_depth) {
  constexpr int kOtherCount = count - 1;
  
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < output_depth.width() && y < output_depth.height()) {
    DepthT depth_value = input_depth(y, x);
    if (depth_value == 0) {
      output_depth(y, x) = 0;
      return;
    }
    
    float3 reference_point =
        make_float3(depth_value * (fx_inv * x + cx_inv),
                    depth_value * (fy_inv * y + cy_inv),
                    depth_value);
    
    // Project the pixel into the other depth maps to verify that there are enough close other depth estimates.
    int ok_count = 0;
    for (int other_index = 0; other_index < kOtherCount; ++ other_index) {
      float3 other_point = p.other_TR_reference[other_index] * reference_point;
      
      if (other_point.z <= 0) {
        continue;
      }
      
      float2 pixel_pos =
          make_float2(fx * (other_point.x / other_point.z) + cx,
                      fy * (other_point.y / other_point.z) + cy);
      // TODO: for pixel_pos.x or .y in ]-1, 0] this will also treat the pixel as in the image
      int px = static_cast<int>(pixel_pos.x);
      int py = static_cast<int>(pixel_pos.y);
      if (px < 0 || py < 0 ||
          px >= output_depth.width() || py >= output_depth.height()) {
        continue;
      }
      
      DepthT other_depth_value = p.other_depths[other_index](py, px);
      if (other_depth_value <= 0 ||
          other_depth_value > max_tolerance_factor * other_point.z ||
          other_depth_value < min_tolerance_factor * other_point.z) {
        continue;
      }
      
      // TODO: Break if required_count cannot be achieved anymore given the number of remaining other depth maps to check?
      ++ ok_count;
    }
    
    output_depth(y, x) = (ok_count >= required_count) ? depth_value : 0;
  }
}

template <int count, typename DepthT>
void OutlierDepthMapFusionCUDA(
    cudaStream_t stream,
    int required_count,
    float tolerance,
    const CUDABuffer_<DepthT>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<DepthT>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kOtherCount = count - 1;
  
  OutlierDepthMapFusionCUDAKernelParam<count, DepthT> p;
  for (int i = 0; i < kOtherCount; ++ i) {
    p.other_TR_reference[i] = others_TR_reference[i];
    
    p.other_depths[i] = *other_depths[i];
  }
  
  const float max_tolerance_factor = 1 + tolerance;
  const float min_tolerance_factor = 1 - tolerance;
  
  // Unprojection intrinsics for pixel center convention.
  const float fx_inv = 1.0f / depth_fx;
  const float fy_inv = 1.0f / depth_fy;
  const float cx_pixel_center = depth_cx - 0.5f;
  const float cy_pixel_center = depth_cy - 0.5f;
  const float cx_inv_pixel_center = -cx_pixel_center / depth_fx;
  const float cy_inv_pixel_center = -cy_pixel_center / depth_fy;
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(output_depth->width(), kBlockWidth),
                GetBlockCount(output_depth->height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  OutlierDepthMapFusionCUDAKernel<count, DepthT>
  <<<grid_dim, block_dim, 0, stream>>>(
      required_count,
      max_tolerance_factor,
      min_tolerance_factor,
      input_depth,
      depth_fx, depth_fy, depth_cx, depth_cy,
      fx_inv, fy_inv, cx_inv_pixel_center, cy_inv_pixel_center,
      p,
      *output_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

template
void OutlierDepthMapFusionCUDA<9, u16>(
    cudaStream_t stream,
    int required_count,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);
template
void OutlierDepthMapFusionCUDA<7, u16>(
    cudaStream_t stream,
    int required_count,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);
template
void OutlierDepthMapFusionCUDA<5, u16>(
    cudaStream_t stream,
    int required_count,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);
template
void OutlierDepthMapFusionCUDA<3, u16>(
    cudaStream_t stream,
    int required_count,
    float tolerance,
    const CUDABuffer_<u16>& input_depth,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer_<u16>* output_depth);


// TODO: This is potentially faster using a box filter.
template <int radius, typename DepthT>
__global__ void ErodeDepthMapCUDAKernel(
    CUDABuffer_<DepthT> input_depth,
    CUDABuffer_<DepthT> output_depth) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < output_depth.width() && y < output_depth.height()) {
    if (x < radius || y < radius ||
        x >= output_depth.width() - radius ||
        y >= output_depth.height() - radius) {
      output_depth(y, x) = 0;
    } else {
      bool all_valid = true;
      for (int dy = y - radius; dy <= y + radius; ++ dy) {
        for (int dx = x - radius; dx <= x + radius; ++ dx) {
          if (input_depth(dy, dx) == 0) {
            all_valid = false;
          }
        }
      }
      output_depth(y, x) = all_valid ? input_depth(y, x) : 0;
    }
  }
}

template <typename DepthT>
void ErodeDepthMapCUDA(
    cudaStream_t stream,
    int radius,
    const CUDABuffer_<DepthT>& input_depth,
    CUDABuffer_<DepthT>* output_depth) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(output_depth->width(), kBlockWidth),
                GetBlockCount(output_depth->height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  if (radius == 1) {
    ErodeDepthMapCUDAKernel<1, DepthT>
    <<<grid_dim, block_dim, 0, stream>>>(
        input_depth,
        *output_depth);
  } else if (radius == 2) {
    ErodeDepthMapCUDAKernel<2, DepthT>
    <<<grid_dim, block_dim, 0, stream>>>(
        input_depth,
        *output_depth);
  } else if (radius == 3) {
    ErodeDepthMapCUDAKernel<3, DepthT>
    <<<grid_dim, block_dim, 0, stream>>>(
        input_depth,
        *output_depth);
  } else {
    LOG(FATAL) << "radius value of " << radius << " is not supported.";
  }
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

template
void ErodeDepthMapCUDA<u16>(
    cudaStream_t stream,
    int radius,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth);


template <typename DepthT>
__global__ void CopyWithoutBorderCUDAKernel(
    CUDABuffer_<DepthT> input_depth,
    CUDABuffer_<DepthT> output_depth) {
  constexpr int kBorderSize = 1;
  
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < output_depth.width() && y < output_depth.height()) {
    if (x < kBorderSize || y < kBorderSize ||
        x >= output_depth.width() - kBorderSize ||
        y >= output_depth.height() - kBorderSize) {
      output_depth(y, x) = 0;
    } else {
      output_depth(y, x) = input_depth(y, x);
    }
  }
}

template <typename DepthT>
void CopyWithoutBorderCUDA(
    cudaStream_t stream,
    const CUDABuffer_<DepthT>& input_depth,
    CUDABuffer_<DepthT>* output_depth) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(output_depth->width(), kBlockWidth),
                GetBlockCount(output_depth->height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  CopyWithoutBorderCUDAKernel<DepthT>
  <<<grid_dim, block_dim, 0, stream>>>(
      input_depth,
      *output_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

template
void CopyWithoutBorderCUDA<u16>(
    cudaStream_t stream,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth);


__global__ void ComputeNormalsAndDropBadPixelsCUDAKernel(
    float normal_dot_threshold,
    float inv_depth_scaling,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    CUDABuffer_<u16> in_depth,
    CUDABuffer_<u16> out_depth,
    CUDABuffer_<float2> out_normals) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < in_depth.width() && y < in_depth.height()) {
    if (in_depth(y, x) == 0) {
      out_depth(y, x) = 0;
      out_normals(y, x) = make_float2(0, 0);
      return;
    }
    
    u16 right_depth = in_depth(y, x + 1);
    u16 left_depth = in_depth(y, x - 1);
    u16 bottom_depth = in_depth(y + 1, x);
    u16 top_depth = in_depth(y - 1, x);
    if (right_depth == 0 || left_depth == 0 || bottom_depth == 0 || top_depth == 0) {
      out_depth(y, x) = 0;
      out_normals(y, x) = make_float2(0, 0);
      return;
    }
    
    float3 left_point;
    UnprojectPoint(x - 1, y, inv_depth_scaling * left_depth,
                   fx_inv, fy_inv, cx_inv, cy_inv, &left_point);
    
    float3 top_point;
    UnprojectPoint(x, y - 1, inv_depth_scaling * top_depth,
                   fx_inv, fy_inv, cx_inv, cy_inv, &top_point);
    
    float3 right_point;
    UnprojectPoint(x + 1, y, inv_depth_scaling * right_depth,
                   fx_inv, fy_inv, cx_inv, cy_inv, &right_point);
    
    float3 bottom_point;
    UnprojectPoint(x, y + 1, inv_depth_scaling * bottom_depth,
                   fx_inv, fy_inv, cx_inv, cy_inv, &bottom_point);
    
    float3 left_to_right =
        make_float3(right_point.x - left_point.x,
                    right_point.y - left_point.y,
                    right_point.z - left_point.z);
    float3 bottom_to_top =
        make_float3(top_point.x - bottom_point.x,
                    top_point.y - bottom_point.y,
                    top_point.z - bottom_point.z);
    
    float3 normal;
    CrossProduct(left_to_right, bottom_to_top, &normal);
    
    float length = Norm(normal);
    if (!(length > 1e-6f)) {
      normal = make_float3(0, 0, -1);  // avoid NaNs
    } else {
      float inv_length = ((fy_inv < 0) ? -1.0f : 1.0f) / length;  // Account for negative fy in ICL-NUIM data
      normal = make_float3(normal.x * inv_length, normal.y * inv_length, normal.z * inv_length);
    }
    
    out_normals(y, x) = make_float2(normal.x, normal.y);
    
    // Discard depth if the normal points too far away from the viewing direction.
    float3 viewing_direction = make_float3(fx_inv * x + cx_inv, fy_inv * y + cy_inv, 1);
    float inv_dir_length = 1.0f / Norm(viewing_direction);
    viewing_direction = make_float3(inv_dir_length * viewing_direction.x,
                                    inv_dir_length * viewing_direction.y,
                                    inv_dir_length * viewing_direction.z);
    float dot = viewing_direction.x * normal.x +
                viewing_direction.y * normal.y +
                viewing_direction.z * normal.z;
    out_depth(y, x) = (dot >= normal_dot_threshold) ? 0 : in_depth(y, x);
  }
}

void ComputeNormalsAndDropBadPixelsCUDA(
    cudaStream_t stream,
    float observation_angle_threshold_deg,
    float depth_scaling,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>& in_depth,
    CUDABuffer_<u16>* out_depth,
    CUDABuffer_<float2>* out_normals) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  // Unprojection intrinsics for pixel center convention.
  const float fx_inv = 1.0f / depth_fx;
  const float fy_inv = 1.0f / depth_fy;
  const float cx_pixel_center = depth_cx - 0.5f;
  const float cy_pixel_center = depth_cy - 0.5f;
  const float cx_inv_pixel_center = -cx_pixel_center / depth_fx;
  const float cy_inv_pixel_center = -cy_pixel_center / depth_fy;
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(out_depth->width(), kBlockWidth),
                GetBlockCount(out_depth->height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  ComputeNormalsAndDropBadPixelsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      -1 * cosf(M_PI / 180.f * observation_angle_threshold_deg),
      1.0f / depth_scaling,
      fx_inv, fy_inv, cx_inv_pixel_center, cy_inv_pixel_center,
      in_depth,
      *out_depth,
      *out_normals);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__global__ void ComputePointRadiiAndRemoveIsolatedPixelsCUDAKernel(
    float point_radius_extension_factor_squared,
    float clamp_factor_term,
    float inv_depth_scaling,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<u16> out_depth) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    if (depth_buffer(y, x) == 0) {
      out_depth(y, x) = 0;
      return;
    }
    
    float depth = inv_depth_scaling * depth_buffer(y, x);
    float3 local_position =
        make_float3(depth * (fx_inv * x + cx_inv),
                    depth * (fy_inv * y + cy_inv),
                    depth);
    
    // Determine the radius of the surfel such that it can connect to its
    // 8-neighborhood, however clamp the radius such that it is not larger
    // than the distance to a 4-neighbor at a given maximum angle to the
    // camera's optical axis.
    int neighbor_count = 0;
    float radius_squared = 0;
    float min_neighbor_distance_squared = CUDART_INF_F;
    for (int dy = y - 1, end_dy = y + 2; dy < end_dy; ++ dy) {
      for (int dx = x - 1, end_dx = x + 2; dx < end_dx; ++ dx) {
        float ddepth = inv_depth_scaling * depth_buffer(dy, dx);
        if ((dx == x && dy == y) ||
            ddepth <= 0) {
          continue;
        }
        ++ neighbor_count;
        
        float3 other_point =
            make_float3(ddepth * (fx_inv * dx + cx_inv),
                        ddepth * (fy_inv * dy + cy_inv),
                        ddepth);
        float3 local_to_other = make_float3(other_point.x - local_position.x,
                                            other_point.y - local_position.y,
                                            other_point.z - local_position.z);
        float distance_squared =
            local_to_other.x * local_to_other.x + local_to_other.y * local_to_other.y + local_to_other.z * local_to_other.z;
        if (distance_squared > radius_squared) {
          radius_squared = distance_squared;
        }
        if (distance_squared < min_neighbor_distance_squared) {
          min_neighbor_distance_squared = distance_squared;
        }
      }
    }
    
    radius_squared *= point_radius_extension_factor_squared;
    float distance_squared_clamp = clamp_factor_term * min_neighbor_distance_squared;
    if (radius_squared > distance_squared_clamp) {
      radius_squared = distance_squared_clamp;
    }
    
    // If we only have neighbors on one side, the radius computation will be
    // affected since the angle of the surface cannot be determined properly.
    // Require at least a reasonable number of neighbors. Use 8 for the most
    // noise-free results.
    constexpr int kMinNeighborPixelsForRadiusComputation = 8;
    
    radius_buffer(y, x) = radius_squared;
    out_depth(y, x) = (neighbor_count < kMinNeighborPixelsForRadiusComputation) ? 0 : depth_buffer(y, x);
  }
}

void ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
    cudaStream_t stream,
    float point_radius_extension_factor,
    float point_radius_clamp_factor,
    float depth_scaling,
    float depth_fx,
    float depth_fy,
    float depth_cx,
    float depth_cy,
    const CUDABuffer_<u16>& depth_buffer,
    CUDABuffer_<float>* radius_buffer,
    CUDABuffer_<u16>* out_depth) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  // Unprojection intrinsics for pixel center convention.
  const float fx_inv = 1.0f / depth_fx;
  const float fy_inv = 1.0f / depth_fy;
  const float cx_pixel_center = depth_cx - 0.5f;
  const float cy_pixel_center = depth_cy - 0.5f;
  const float cx_inv_pixel_center = -cx_pixel_center / depth_fx;
  const float cy_inv_pixel_center = -cy_pixel_center / depth_fy;
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(depth_buffer.width(), kBlockWidth),
                GetBlockCount(depth_buffer.height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  ComputePointRadiiAndRemoveIsolatedPixelsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      point_radius_extension_factor * point_radius_extension_factor,
      point_radius_clamp_factor * point_radius_clamp_factor * sqrtf(2) * sqrtf(2),
      1.0f / depth_scaling,
      fx_inv, fy_inv, cx_inv_pixel_center, cy_inv_pixel_center,
      depth_buffer,
      *radius_buffer,
      *out_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

}
