// Copyright 2019 ETH Zürich, Thomas Schöps
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

#include "surfel_meshing/cuda_surfel_reconstruction_kernels.h"
#include "surfel_meshing/cuda_surfel_reconstruction_kernels.cuh"

#include <libvis/cuda/cuda_util.h>
#include <libvis/cuda/cuda_matrix.cuh>

namespace vis {

void CreateNewSurfelsCUDA(
    cudaStream_t stream,
    u32 frame_index,
    const SE3f& global_T_local,
    float depth_scaling,
    float radius_factor_for_regularization_neighbors,
    const PinholeCamera4f& depth_camera,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<float2>& normals_buffer,
    const CUDABuffer<float>& radius_buffer,
    const CUDABuffer<Vec3u8>& color_buffer,
    const CUDABuffer<u32>& supporting_surfels,
    const CUDABuffer<u32>& conflicting_surfels,
    void** new_surfels_temp_storage,
    usize* new_surfels_temp_storage_bytes,
    CUDABuffer<u8>* new_surfel_flag_vector,
    CUDABuffer<u32>* new_surfel_indices,
    u32 surfel_count,
    CUDABuffer<float>* surfels,
    u32* new_surfel_count,
    u8* new_surfel_count_2) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  const float fx = depth_camera.parameters()[0];
  const float fy = depth_camera.parameters()[1];
  const float cx = depth_camera.parameters()[2];
  const float cy = depth_camera.parameters()[3];
  
  // Unprojection intrinsics for pixel center convention.
  const float fx_inv = 1.0f / fx;
  const float fy_inv = 1.0f / fy;
  const float cx_pixel_center = cx - 0.5f;
  const float cy_pixel_center = cy - 0.5f;
  const float cx_inv_pixel_center = -cx_pixel_center / fx;
  const float cy_inv_pixel_center = -cy_pixel_center / fy;
  
  // The first kernel marks in a sequential (non-pitched) vector whether a new surfel is created for the corresponding pixel or not.
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(depth_buffer.width(), kBlockWidth),
                GetBlockCount(depth_buffer.height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  CallCreateNewSurfelsCUDASerializingKernel(
      stream,
      grid_dim,
      block_dim,
      depth_buffer.ToCUDA(),
      supporting_surfels.ToCUDA(),
      conflicting_surfels.ToCUDA(),
      new_surfel_flag_vector->ToCUDA());
  
  // Indices for the new surfels are computed with a parallel exclusive prefix sum from CUB.
  if (*new_surfels_temp_storage_bytes == 0) {
    CallCUBExclusiveSum(
        *new_surfels_temp_storage,
        *new_surfels_temp_storage_bytes,
        new_surfel_flag_vector->ToCUDA().address(),
        new_surfel_indices->ToCUDA().address(),
        depth_buffer.width() * depth_buffer.height(),
        stream);
    
    cudaMalloc(new_surfels_temp_storage, *new_surfels_temp_storage_bytes);
  }
  
  CallCUBExclusiveSum(
      *new_surfels_temp_storage,
      *new_surfels_temp_storage_bytes,
      new_surfel_flag_vector->ToCUDA().address(),
      new_surfel_indices->ToCUDA().address(),
      depth_buffer.width() * depth_buffer.height(),
      stream);
  
  // Read back the number of new surfels to the CPU by reading the last element
  // in new_surfel_indices and new_surfel_flag_vector.
  // TODO: Do this concurrently with the next kernel call?
  new_surfel_indices->DownloadPartAsync(
      (depth_buffer.width() * depth_buffer.height() - 1) * sizeof(u32),
      1 * sizeof(u32),
      stream,
      new_surfel_count);
  new_surfel_flag_vector->DownloadPartAsync(
      (depth_buffer.width() * depth_buffer.height() - 1) * sizeof(u8),
      1 * sizeof(u8),
      stream,
      new_surfel_count_2);
  
  // Now that the indices are known, the actual surfel creation is done.
  CallCreateNewSurfelsCUDACreationKernel(
      stream,
      grid_dim,
      block_dim,
      frame_index,
      1.0f / depth_scaling,
      fx_inv, fy_inv, cx_inv_pixel_center, cy_inv_pixel_center,
      CUDAMatrix3x4(global_T_local.matrix3x4()),
      depth_buffer.ToCUDA(),
      normals_buffer.ToCUDA(),
      radius_buffer.ToCUDA(),
      *reinterpret_cast<const CUDABuffer_<uchar3>*>(&color_buffer.ToCUDA()),
      supporting_surfels.ToCUDA(),
      new_surfel_flag_vector->ToCUDA(),
      new_surfel_indices->ToCUDA(),
      surfel_count,
      surfels->ToCUDA(),
      radius_factor_for_regularization_neighbors * radius_factor_for_regularization_neighbors);
}

void BlendMeasurementsCUDA(
    cudaStream_t stream,
    int measurement_blending_radius,
    float depth_correction_factor,
    CUDABuffer<u16>* depth_buffer,
    const CUDABuffer<u32>& supporting_surfels,
    const CUDABuffer<u32>& supporting_surfel_counts,
    const CUDABuffer<float>& supporting_surfel_depth_sums,
    CUDABuffer<u8>* distance_map,
    CUDABuffer<float>* surfel_depth_average_deltas,
    CUDABuffer<u8>* new_distance_map,
    CUDABuffer<float>* new_surfel_depth_average_deltas) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  distance_map->Clear(0, stream);
  new_distance_map->Clear(0, stream);
  
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(GetBlockCount(supporting_surfels.width(), kBlockWidth),
                GetBlockCount(supporting_surfels.height(), kBlockHeight));
  dim3 block_dim(kBlockWidth, kBlockHeight);
  
  // Find pixels with distance == 1, having a depth measurement next to the measurement border, and supporting surfels.
  CallBlendMeasurementsCUDAStartKernel(
      stream,
      grid_dim,
      block_dim,
      1.0f / depth_correction_factor,
      depth_buffer->ToCUDA(),
      supporting_surfels.ToCUDA(),
      supporting_surfel_counts.ToCUDA(),
      supporting_surfel_depth_sums.ToCUDA(),
      distance_map->ToCUDA(),
      surfel_depth_average_deltas->ToCUDA(),
      new_distance_map->ToCUDA(),
      new_surfel_depth_average_deltas->ToCUDA());
  
  // Find pixels with distances in [2, measurement_blending_radius] and average surfel depths.
  for (int iteration = 2; iteration < measurement_blending_radius; ++ iteration) {
    CallBlendMeasurementsCUDAIterationKernel(
        stream,
        grid_dim,
        block_dim,
        iteration,
        1.0f / (measurement_blending_radius - 1.0f),
        1.0f / depth_correction_factor,
        depth_buffer->ToCUDA(),
        supporting_surfels.ToCUDA(),
        distance_map->ToCUDA(),
        surfel_depth_average_deltas->ToCUDA(),
        new_distance_map->ToCUDA(),
        new_surfel_depth_average_deltas->ToCUDA());
  }
}

void IntegrateMeasurementsCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float max_surfel_confidence,
    float sensor_noise_factor,
    float normal_compatibility_threshold_deg,
    const SE3f& global_T_local,
    float depth_scaling,
    const PinholeCamera4f& depth_camera,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<float2>& normals_buffer,
    const CUDABuffer<float>& radius_buffer,
    const CUDABuffer<Vec3u8>& color_buffer,
    const CUDABuffer<u32>& supporting_surfels,
    const CUDABuffer<u32>& supporting_surfel_counts,
    const CUDABuffer<u32>& conflicting_surfels,
    const CUDABuffer<float>& first_surfel_depth,
    u32 surfel_count,
    CUDABuffer<float>* surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  const float fx = depth_camera.parameters()[0];
  const float fy = depth_camera.parameters()[1];
  const float cx = depth_camera.parameters()[2];
  const float cy = depth_camera.parameters()[3];
  
  // Unprojection intrinsics for pixel center convention.
  const float fx_inv = 1.0f / fx;
  const float fy_inv = 1.0f / fy;
  const float cx_pixel_center = cx - 0.5f;
  const float cy_pixel_center = cy - 0.5f;
  const float cx_inv_pixel_center = -cx_pixel_center / fx;
  const float cy_inv_pixel_center = -cy_pixel_center / fy;
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  CallIntegrateMeasurementsCUDAKernel(
      stream,
      grid_dim,
      block_dim,
      frame_index,
      surfel_integration_active_window_size,
      max_surfel_confidence,
      sensor_noise_factor,
      cosf(M_PI / 180.0f * normal_compatibility_threshold_deg),
      1.0f / depth_scaling,
      fx, fy, cx, cy,
      fx_inv, fy_inv, cx_inv_pixel_center, cy_inv_pixel_center,
      CUDAMatrix3x4(global_T_local.inverse().matrix3x4()),
      CUDAMatrix3x4(global_T_local.matrix3x4()),
      depth_buffer.ToCUDA(),
      normals_buffer.ToCUDA(),
      radius_buffer.ToCUDA(),
      *reinterpret_cast<const CUDABuffer_<uchar3>*>(&color_buffer.ToCUDA()),
      supporting_surfels.ToCUDA(),
      supporting_surfel_counts.ToCUDA(),
      conflicting_surfels.ToCUDA(),
      first_surfel_depth.ToCUDA(),
      surfel_count,
      surfels->ToCUDA());
}

void UpdateNeighborsCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float radius_factor_for_regularization_neighbors,
    const CUDABuffer<u32>& supporting_surfels,
    const CUDABuffer<u32>& /*conflicting_surfels*/,
    const PinholeCamera4f& depth_camera,
    const SE3f& local_T_global,
    float sensor_noise_factor,
    float depth_correction_factor,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<float2>& /*normals_buffer*/,
    const CUDABuffer<float>& radius_buffer,
    const CUDABuffer<float>& first_surfel_depth,
    usize surfel_count,
    CUDABuffer<float>* surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  const float fx = depth_camera.parameters()[0];
  const float fy = depth_camera.parameters()[1];
  const float cx = depth_camera.parameters()[2];
  const float cy = depth_camera.parameters()[3];
  
  constexpr int kSurfelsBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kSurfelsBlockWidth));
  dim3 block_dim(kSurfelsBlockWidth);
  
  CallUpdateNeighborsCUDAKernel(
      stream,
      grid_dim,
      block_dim,
      frame_index,
      surfel_integration_active_window_size,
      radius_factor_for_regularization_neighbors * radius_factor_for_regularization_neighbors,
      supporting_surfels.ToCUDA(),
      fx, fy, cx, cy,
      CUDAMatrix3x4(local_T_global.matrix3x4()),
      sensor_noise_factor,
      depth_correction_factor,
      depth_buffer.ToCUDA(),
      radius_buffer.ToCUDA(),
      first_surfel_depth.ToCUDA(),
      surfel_count,
      surfels->ToCUDA());
  
  
  CallUpdateNeighborsCUDARemoveReplacedNeighborsKernel(
      stream,
      grid_dim,
      block_dim,
      frame_index,
      surfel_count,
      surfels->ToCUDA());
}

void RenderMinDepthCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    const SE3f& local_T_global,
    const PinholeCamera4f& depth_camera,
    CUDABuffer<float>* first_surfel_depth,
    u32 surfel_count,
    const CUDABuffer<float>& surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  const float fx = depth_camera.parameters()[0];
  const float fy = depth_camera.parameters()[1];
  const float cx = depth_camera.parameters()[2];
  const float cy = depth_camera.parameters()[3];
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  CallRenderMinDepthCUDAKernel(
      stream,
      grid_dim,
      block_dim,
      frame_index,
      surfel_integration_active_window_size,
      fx, fy, cx, cy,
      CUDAMatrix3x4(local_T_global.matrix3x4()),
      surfel_count,
      surfels.ToCUDA(),
      first_surfel_depth->ToCUDA());
}

void AssociateSurfelsCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float sensor_noise_factor,
    float normal_compatibility_threshold_deg,
    const SE3f& local_T_global,
    const PinholeCamera4f& depth_camera,
    float depth_correction_factor,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<float2>& normals_buffer,
    const CUDABuffer<float>& radius_buffer,
    CUDABuffer<u32>* supporting_surfels,
    CUDABuffer<u32>* supporting_surfel_counts,
    CUDABuffer<float>* supporting_surfel_depth_sums,
    CUDABuffer<u32>* conflicting_surfels,
    CUDABuffer<float>* first_surfel_depth,
    u32 surfel_count,
    const CUDABuffer<float>& surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  const float fx = depth_camera.parameters()[0];
  const float fy = depth_camera.parameters()[1];
  const float cx = depth_camera.parameters()[2];
  const float cy = depth_camera.parameters()[3];
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  CallAssociateSurfelsCUDAKernel(
      stream,
      grid_dim,
      block_dim,
      frame_index,
      surfel_integration_active_window_size,
      fx, fy, cx, cy,
      CUDAMatrix3x4(local_T_global.matrix3x4()),
      sensor_noise_factor,
      cosf(M_PI / 180.0f * normal_compatibility_threshold_deg),
      surfel_count,
      surfels.ToCUDA(),
      depth_correction_factor,
      depth_buffer.ToCUDA(),
      normals_buffer.ToCUDA(),
      radius_buffer.ToCUDA(),
      supporting_surfels->ToCUDA(),
      supporting_surfel_counts->ToCUDA(),
      supporting_surfel_depth_sums->ToCUDA(),
      conflicting_surfels->ToCUDA(),
      first_surfel_depth->ToCUDA());
}

void MergeSurfelsCUDA(
    cudaStream_t stream,
    u32 /*frame_index*/,
    int /*surfel_integration_active_window_size*/,
    float sensor_noise_factor,
    float normal_compatibility_threshold_deg,
    const SE3f& local_T_global,
    const PinholeCamera4f& depth_camera,
    float depth_correction_factor,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<float2>& normals_buffer,
    const CUDABuffer<float>& radius_buffer,
    CUDABuffer<u32>* supporting_surfels,
    CUDABuffer<u32>* supporting_surfel_counts,
    CUDABuffer<float>* supporting_surfel_depth_sums,
    CUDABuffer<u32>* conflicting_surfels,
    CUDABuffer<float>* first_surfel_depth,
    u32 surfel_count,
    u32* merge_count,
    CUDABuffer<float>* surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  const float fx = depth_camera.parameters()[0];
  const float fy = depth_camera.parameters()[1];
  const float cx = depth_camera.parameters()[2];
  const float cy = depth_camera.parameters()[3];
  
  dim3 grid_dim(GetBlockCount(surfel_count, kMergeBlockWidth));
  dim3 block_dim(kMergeBlockWidth);
  
  static CUDABuffer<u32> num_merges_buffer(1, 1);  // TODO: do not use static
  num_merges_buffer.Clear(0, stream);
  
  CallMergeSurfelsCUDAKernel(
      stream,
      grid_dim,
      block_dim,
      fx, fy, cx, cy,
      CUDAMatrix3x4(local_T_global.matrix3x4()),
      sensor_noise_factor,
      cosf(M_PI / 180.0f * normal_compatibility_threshold_deg),
      surfel_count,
      surfels->ToCUDA(),
      depth_correction_factor,
      depth_buffer.ToCUDA(),
      normals_buffer.ToCUDA(),
      radius_buffer.ToCUDA(),
      supporting_surfels->ToCUDA(),
      supporting_surfel_counts->ToCUDA(),
      supporting_surfel_depth_sums->ToCUDA(),
      conflicting_surfels->ToCUDA(),
      first_surfel_depth->ToCUDA(),
      num_merges_buffer.ToCUDA());
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  u32 num_merges = 0;
  num_merges_buffer.DownloadAsync(stream, &num_merges);
  cudaStreamSynchronize(stream);
  *merge_count += num_merges;
}

}
