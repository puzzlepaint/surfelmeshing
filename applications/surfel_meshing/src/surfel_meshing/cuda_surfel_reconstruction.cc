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


#define LIBVIS_ENABLE_TIMING

#include "surfel_meshing/cuda_surfel_reconstruction.h"

#include <libvis/image_display.h>
#include <libvis/timing.h>

#include "surfel_meshing/cuda_depth_processing.cuh"
#include "surfel_meshing/cuda_surfel_reconstruction_kernels.cuh"
#include "surfel_meshing/cuda_surfel_reconstruction_kernels.h"
#include "surfel_meshing/surfel.h"

namespace vis {

CUDASurfelReconstruction::CUDASurfelReconstruction(
    usize max_surfel_count,
    const PinholeCamera4f& depth_camera,
    cudaGraphicsResource_t vertex_buffer_resource,
    cudaGraphicsResource_t neighbor_index_buffer_resource,
    cudaGraphicsResource_t normal_vertex_buffer_resource,
    const shared_ptr<SurfelMeshingRenderWindow>& render_window)
    : surfel_count_(0),
      merge_count_(0),
      max_surfel_count_(max_surfel_count),
      depth_camera_(depth_camera),
      vertex_buffer_resource_(vertex_buffer_resource),
      neighbor_index_buffer_resource_(neighbor_index_buffer_resource),
      normal_vertex_buffer_resource_(normal_vertex_buffer_resource),
      render_window_(render_window) {
  surfels_.reset(new CUDABuffer<float>(kSurfelAttributeCount, max_surfel_count));
  
  distance_map_.reset(new CUDABuffer<u8>(depth_camera_.height(), depth_camera_.width()));
  surfel_depth_average_deltas_.reset(new CUDABuffer<float>(depth_camera_.height(), depth_camera_.width()));;
  new_distance_map_.reset(new CUDABuffer<u8>(depth_camera_.height(), depth_camera_.width()));;
  new_surfel_depth_average_deltas_.reset(new CUDABuffer<float>(depth_camera_.height(), depth_camera_.width()));;
  
  supporting_surfels_.reset(new CUDABuffer<u32>(depth_camera_.height(), depth_camera_.width()));
  supporting_surfel_counts_.reset(new CUDABuffer<u32>(depth_camera_.height(), depth_camera_.width()));
  supporting_surfel_depth_sums_.reset(new CUDABuffer<float>(depth_camera_.height(), depth_camera_.width()));
  conflicting_surfels_.reset(new CUDABuffer<u32>(depth_camera_.height(), depth_camera_.width()));
  first_surfel_depth_.reset(new CUDABuffer<float>(depth_camera_.height(), depth_camera_.width()));
  new_surfel_flag_vector_.reset(new CUDABuffer<u8>(1, depth_camera_.height() * depth_camera_.width()));
  new_surfel_indices_.reset(new CUDABuffer<u32>(1, depth_camera_.height() * depth_camera_.width()));
  
  new_surfels_temp_storage_ = nullptr;
  new_surfels_temp_storage_bytes_ = 0;
  
  cudaEventCreate(&data_association_start_event_);
  cudaEventCreate(&data_association_end_event_);
  cudaEventCreate(&surfel_merging_start_event_);
  cudaEventCreate(&surfel_merging_end_event_);
  cudaEventCreate(&measurement_blending_start_event_);
  cudaEventCreate(&measurement_blending_end_event_);
  cudaEventCreate(&integration_start_event_);
  cudaEventCreate(&integration_end_event_);
  cudaEventCreate(&neighbor_update_start_event_);
  cudaEventCreate(&neighbor_update_end_event_);
  cudaEventCreate(&new_surfel_creation_start_event_);
  cudaEventCreate(&new_surfel_creation_end_event_);
  cudaEventCreate(&regularization_start_event_);
  cudaEventCreate(&regularization_end_event_);
}

CUDASurfelReconstruction::~CUDASurfelReconstruction() {
  cudaFree(new_surfels_temp_storage_);
  
  cudaEventDestroy(data_association_start_event_);
  cudaEventDestroy(data_association_end_event_);
  cudaEventDestroy(surfel_merging_start_event_);
  cudaEventDestroy(surfel_merging_end_event_);
  cudaEventDestroy(measurement_blending_start_event_);
  cudaEventDestroy(measurement_blending_end_event_);
  cudaEventDestroy(integration_start_event_);
  cudaEventDestroy(integration_end_event_);
  cudaEventDestroy(neighbor_update_start_event_);
  cudaEventDestroy(neighbor_update_end_event_);
  cudaEventDestroy(new_surfel_creation_start_event_);
  cudaEventDestroy(new_surfel_creation_end_event_);
  cudaEventDestroy(regularization_start_event_);
  cudaEventDestroy(regularization_end_event_);
}

void CUDASurfelReconstruction::Integrate(
    cudaStream_t stream,
    u32 frame_index,
    float depth_scaling,
    CUDABuffer<u16>* depth_buffer,
    const CUDABuffer<float2>& normals_buffer,
    const CUDABuffer<float>& radius_buffer,
    const CUDABuffer<Vec3u8>& color_buffer,
    const SE3f& global_T_local,
    float sensor_noise_factor,
    float max_surfel_confidence,
    float regularizer_weight,
    int regularization_frame_window_size,
    bool do_blending,
    int measurement_blending_radius,
    int regularization_iterations_per_integration_iteration,
    float radius_factor_for_regularization_neighbors,
    float normal_compatibility_threshold_deg,
    int surfel_integration_active_window_size) {
  cudaEventRecord(data_association_start_event_, stream);
  
  // TODO: These "clear"s might be faster as a single kernel call.
  supporting_surfels_->Clear(Surfel::kInvalidIndex, stream);
  supporting_surfel_counts_->Clear(0, stream);
  supporting_surfel_depth_sums_->Clear(0, stream);
  conflicting_surfels_->Clear(Surfel::kInvalidIndex, stream);
  first_surfel_depth_->Clear(numeric_limits<float>::infinity(), stream);
  
  RenderMinDepthCUDA(
      stream,
      frame_index,
      surfel_integration_active_window_size,
      global_T_local.inverse(),
      depth_camera_,
      first_surfel_depth_.get(),
      surfel_count_,
      *surfels_);
  
  AssociateSurfelsCUDA(
      stream,
      frame_index,
      surfel_integration_active_window_size,
      sensor_noise_factor,
      normal_compatibility_threshold_deg,
      global_T_local.inverse(),
      depth_camera_,
      1.0f / depth_scaling,
      *depth_buffer,
      normals_buffer,
      radius_buffer,
      supporting_surfels_.get(),
      supporting_surfel_counts_.get(),
      supporting_surfel_depth_sums_.get(),
      conflicting_surfels_.get(),
      first_surfel_depth_.get(),
      surfel_count_,
      *surfels_);
  
  cudaEventRecord(data_association_end_event_, stream);
  cudaEventRecord(surfel_merging_start_event_, stream);
  
  // TEST: Merge surfels: delete surfels which project to the same pixel as
  //       a supported surfels and have very similar attributes.
  MergeSurfelsCUDA(
      stream,
      frame_index,
      surfel_integration_active_window_size,
      sensor_noise_factor,
      normal_compatibility_threshold_deg,
      global_T_local.inverse(),
      depth_camera_,
      1.0f / depth_scaling,
      *depth_buffer,
      normals_buffer,
      radius_buffer,
      supporting_surfels_.get(),
      supporting_surfel_counts_.get(),
      supporting_surfel_depth_sums_.get(),
      conflicting_surfels_.get(),
      first_surfel_depth_.get(),
      surfel_count_,
      &merge_count_,
      surfels_.get());
  
  cudaEventRecord(surfel_merging_end_event_, stream);
  cudaEventRecord(measurement_blending_start_event_, stream);
  
  // NOTE: This can change the supporting surfels for the affected pixels, in
  //       principle they should be adapted afterwards but this is not done.
  if (do_blending) {
    BlendMeasurementsCUDA(
        stream,
        measurement_blending_radius,
        1.0f / depth_scaling,
        depth_buffer,
        *supporting_surfels_,
        *supporting_surfel_counts_,
        *supporting_surfel_depth_sums_,
        distance_map_.get(),
        surfel_depth_average_deltas_.get(),
        new_distance_map_.get(),
        new_surfel_depth_average_deltas_.get());
  }
  
  cudaEventRecord(measurement_blending_end_event_, stream);
  cudaEventRecord(integration_start_event_, stream);
  
  IntegrateMeasurementsCUDA(
      stream,
      frame_index,
      surfel_integration_active_window_size,
      max_surfel_confidence,
      sensor_noise_factor,
      normal_compatibility_threshold_deg,
      global_T_local,
      depth_scaling,
      depth_camera_,
      *depth_buffer,
      normals_buffer,
      radius_buffer,
      color_buffer,
      *supporting_surfels_,
      *supporting_surfel_counts_,
      *conflicting_surfels_,
      *first_surfel_depth_,
      surfel_count_,
      surfels_.get());
  
  cudaEventRecord(integration_end_event_, stream);
  cudaEventRecord(neighbor_update_start_event_, stream);
  
  UpdateNeighborsCUDA(
      stream,
      frame_index,
      surfel_integration_active_window_size,
      radius_factor_for_regularization_neighbors,
      *supporting_surfels_,
      *conflicting_surfels_,
      depth_camera_,
      global_T_local.inverse(),
      sensor_noise_factor,
      1.0f / depth_scaling,
      *depth_buffer,
      normals_buffer,
      radius_buffer,
      *first_surfel_depth_,
      surfel_count_,
      surfels_.get());
  
  cudaEventRecord(neighbor_update_end_event_, stream);
  cudaEventRecord(new_surfel_creation_start_event_, stream);
  
  u32 new_surfel_count;
  u8 new_surfel_count_2;
  CreateNewSurfelsCUDA(
      stream,
      frame_index,
      global_T_local,
      depth_scaling,
      radius_factor_for_regularization_neighbors,
      depth_camera_,
      *depth_buffer,
      normals_buffer,
      radius_buffer,
      color_buffer,
      *supporting_surfels_,
      *conflicting_surfels_,
      &new_surfels_temp_storage_,
      &new_surfels_temp_storage_bytes_,
      new_surfel_flag_vector_.get(),
      new_surfel_indices_.get(),
      surfel_count_,
      surfels_.get(),
      &new_surfel_count,
      &new_surfel_count_2);
  
  cudaEventRecord(new_surfel_creation_end_event_, stream);
  
  cudaStreamSynchronize(stream);
  surfel_count_ += new_surfel_count + new_surfel_count_2;
  
  cudaEventRecord(regularization_start_event_, stream);
  
  if (regularization_iterations_per_integration_iteration == 0) {
    RegularizeSurfelsCUDA(
        stream,
        /*disable_denoising*/ true,
        frame_index,
        radius_factor_for_regularization_neighbors,
        regularizer_weight,
        regularization_frame_window_size,
        surfel_count_,
        &surfels_->ToCUDA());
  } else {
    for (int i = 0; i < regularization_iterations_per_integration_iteration; ++ i) {
      RegularizeSurfelsCUDA(
          stream,
          /*disable_denoising*/ false,
          frame_index,
          radius_factor_for_regularization_neighbors,
          regularizer_weight,
          regularization_frame_window_size,
          surfel_count_,
          &surfels_->ToCUDA());
    }
  }
  
  cudaEventRecord(regularization_end_event_, stream);
}

void CUDASurfelReconstruction::Regularize(
    cudaStream_t stream,
    u32 frame_index,
    float regularizer_weight,
    float radius_factor_for_regularization_neighbors,
    int regularization_frame_window_size) {
  RegularizeSurfelsCUDA(
      stream,
      /*disable_denoising*/ false,
      frame_index,
      radius_factor_for_regularization_neighbors,
      regularizer_weight,
      regularization_frame_window_size,
      surfel_count_,
      &surfels_->ToCUDA());
}

void CUDASurfelReconstruction::TransferAllToCPU(
    cudaStream_t stream,
    u32 frame_index,
    CUDASurfelsCPU* buffers) {
  CUDASurfelBuffersCPU* buffer = buffers->write_buffers();
  
  buffer->frame_index = frame_index;
  buffer->surfel_count = surfel_count_;
  
  surfels_->DownloadPartAsync(kSurfelSmoothX * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_x_buffer);
  surfels_->DownloadPartAsync(kSurfelSmoothY * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_y_buffer);
  surfels_->DownloadPartAsync(kSurfelSmoothZ * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_z_buffer);
  
  surfels_->DownloadPartAsync(kSurfelRadiusSquared * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_radius_squared_buffer);
  
  surfels_->DownloadPartAsync(kSurfelNormalX * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_normal_x_buffer);
  surfels_->DownloadPartAsync(kSurfelNormalY * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_normal_y_buffer);
  surfels_->DownloadPartAsync(kSurfelNormalZ * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(float), stream, buffer->surfel_normal_z_buffer);
  
  surfels_->DownloadPartAsync(kSurfelLastUpdateStamp * surfels_->ToCUDA().pitch(), surfel_count_ * sizeof(u32), stream, reinterpret_cast<float*>(buffer->surfel_last_update_stamp_buffer));
}

void CUDASurfelReconstruction::UpdateVisualizationBuffers(
    cudaStream_t stream,
    u32 frame_index,
    u32 latest_triangulated_frame_index,
    u32 latest_mesh_surfel_count,
    int surfel_integration_active_window_size,
    bool visualize_last_update_timestamp,
    bool visualize_creation_timestamp,
    bool visualize_radii,
    bool visualize_normals) {
  CHECK(sizeof(Point3fC3u8) % sizeof(float) == 0);
  u32 point_size_in_floats = sizeof(Point3fC3u8) / sizeof(float);
  UpdateSurfelVertexBufferCUDA(
      stream,
      frame_index,
      surfel_integration_active_window_size,
      surfel_count_,
      surfels_->ToCUDA(),
      latest_triangulated_frame_index,
      latest_mesh_surfel_count,
      vertex_buffer_resource_,
      point_size_in_floats,
      visualize_last_update_timestamp,
      visualize_creation_timestamp,
      visualize_radii,
      visualize_normals);
  
  if (neighbor_index_buffer_resource_) {
    UpdateNeighborIndexBufferCUDA(
        stream,
        surfel_count_,
        surfels_->ToCUDA(),
        neighbor_index_buffer_resource_);
  }
  
  if (normal_vertex_buffer_resource_) {
    UpdateNormalVertexBufferCUDA(
        stream,
        surfel_count_,
        surfels_->ToCUDA(),
        normal_vertex_buffer_resource_);
  }
}

void CUDASurfelReconstruction::ExportVertices(
    cudaStream_t stream,
    CUDABuffer<float>* position_buffer,
    CUDABuffer<u8>* color_buffer) {
  ExportVerticesCUDA(stream, surfel_count_, surfels_->ToCUDA(), &position_buffer->ToCUDA(), &color_buffer->ToCUDA());
}

void CUDASurfelReconstruction::GetTimings(
    float* data_association,
    float* surfel_merging,
    float* measurement_blending,
    float* integration,
    float* neighbor_update,
    float* new_surfel_creation,
    float* regularization) {
  cudaEventSynchronize(regularization_end_event_);
  
  cudaEventElapsedTime(data_association, data_association_start_event_, data_association_end_event_);
  cudaEventElapsedTime(surfel_merging, surfel_merging_start_event_, surfel_merging_end_event_);
  cudaEventElapsedTime(measurement_blending, measurement_blending_start_event_, measurement_blending_end_event_);
  cudaEventElapsedTime(integration, integration_start_event_, integration_end_event_);
  cudaEventElapsedTime(neighbor_update, neighbor_update_start_event_, neighbor_update_end_event_);
  cudaEventElapsedTime(new_surfel_creation, new_surfel_creation_start_event_, new_surfel_creation_end_event_);
  cudaEventElapsedTime(regularization, regularization_start_event_, regularization_end_event_);
}

}
