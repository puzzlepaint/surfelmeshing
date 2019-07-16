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

#pragma once

#include <cuda_runtime.h>
#include <libvis/camera.h>
#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>
#include <libvis/cuda/cuda_buffer.h>

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
    u8* new_surfel_count_2);

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
    CUDABuffer<float>* new_surfel_depth_average_deltas);

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
    CUDABuffer<float>* surfels);

void UpdateNeighborsCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float radius_factor_for_regularization_neighbors,
    const CUDABuffer<u32>& supporting_surfels,
    const CUDABuffer<u32>& conflicting_surfels,
    const PinholeCamera4f& depth_camera,
    const SE3f& local_T_global,
    float sensor_noise_factor,
    float depth_correction_factor,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<float2>& normals_buffer,
    const CUDABuffer<float>& radius_buffer,
    const CUDABuffer<float>& first_surfel_depth,
    usize surfel_count,
    CUDABuffer<float>* surfels);

void RenderMinDepthCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    const SE3f& local_T_global,
    const PinholeCamera4f& depth_camera,
    CUDABuffer<float>* first_surfel_depth,
    u32 surfel_count,
    const CUDABuffer<float>& surfels);

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
    const CUDABuffer<float>& surfels);

void MergeSurfelsCUDA(
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
    u32* merge_count,
    CUDABuffer<float>* surfels);

}
