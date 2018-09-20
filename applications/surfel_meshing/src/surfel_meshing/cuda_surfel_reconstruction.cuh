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


#pragma once

#include <cuda_runtime.h>

#include <libvis/camera.h>
#include <libvis/libvis.h>
#include <libvis/opengl.h>
#include <libvis/sophus.h>

#include <libvis/cuda/cuda_buffer.h>

namespace vis {

// The surfel structure is stored in a large buffer. It is organized
// such that each row stores one attribute and each column stores the
// attribute values for one surfel.

// TODO: The attributes are quite wasteful memory-wise. I would guess that some
//       attributes could be packed without any apparent negative consequences.
//       For example, the normals could be stored with (much) less precision,
//       perhaps 10 bit per component to fit into a 32 bit value?

// float attributes:
constexpr int kSurfelX = 0;
constexpr int kSurfelY = 1;
constexpr int kSurfelZ = 2;
constexpr int kSurfelSmoothX = 3;
constexpr int kSurfelSmoothY = 4;
constexpr int kSurfelSmoothZ = 5;
constexpr int kSurfelConfidence = 6;
constexpr int kSurfelRadiusSquared = 7;
constexpr int kSurfelNormalX = 8;
constexpr int kSurfelNormalY = 9;
constexpr int kSurfelNormalZ = 10;
constexpr int kSurfelGradientX = 11;
constexpr int kSurfelGradientY = 12;
constexpr int kSurfelGradientZ = 13;
constexpr int kSurfelAccumX = 14;
constexpr int kSurfelAccumY = 15;
constexpr int kSurfelAccumZ = 16;

// u32 attributes:
constexpr int kSurfelCreationStamp = 17;
constexpr int kSurfelLastUpdateStamp = 18;
constexpr int kSurfelNeighbor0 = 19;  // and 20, 21, 22 for the other neighbors.
// (not an attribute itself):
constexpr int kSurfelNeighborCount = 4;
constexpr int kSurfelGradientCount = 23;

// Vec4u8 attributes:
constexpr int kSurfelColor = 24;  // (r, g, b, neighbor detach request flag)

constexpr int kSurfelAttributeCount = 25;


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

void UpdateSurfelVertexBufferCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    u32 surfel_count,
    const CUDABuffer<float>& surfels,
    u32 latest_triangulated_frame_index,
    u32 latest_mesh_surfel_count,
    cudaGraphicsResource_t vertex_buffer_resource,
    bool visualize_last_update_timestamp,
    bool visualize_creation_timestamp,
    bool visualize_radii,
    bool visualize_normals);

void UpdateNeighborIndexBufferCUDA(
    cudaStream_t stream,
    u32 surfel_count,
    const CUDABuffer<float>& surfels,
    cudaGraphicsResource_t neighbor_index_buffer_resource);

void UpdateNormalVertexBufferCUDA(
    cudaStream_t stream,
    u32 surfel_count,
    const CUDABuffer<float>& surfels,
    cudaGraphicsResource_t normal_vertex_buffer_resource);

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

void RegularizeSurfelsCUDA(
    cudaStream_t stream,
    bool disable_denoising,
    u32 frame_index,
    float radius_factor_for_regularization_neighbors,
    float regularizer_weight,
    int regularization_frame_window_size,
    u32 surfel_count,
    CUDABuffer<float>* surfels);

void ExportVerticesCUDA(
    cudaStream_t stream,
    u32 surfel_count,
    const CUDABuffer<float>& surfels,
    CUDABuffer<float>* position_buffer,
    CUDABuffer<u8>* color_buffer);

void DebugPrintSurfelCUDA(
    cudaStream_t stream,
    usize surfel_index,
    const CUDABuffer<float>& surfels);

}
