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

#include <memory>

#include <cuda_runtime.h>
#include <libvis/libvis.h>

#include <libvis/cuda/cuda_buffer.h>
#include "surfel_meshing/cuda_surfels_cpu.h"
#include "surfel_meshing/surfel_meshing_render_window.h"

namespace vis {

// Main class for CUDA-based surfel reconstruction.
class CUDASurfelReconstruction {
 public:
  // Constructor, allocates CUDA buffers and events.
  CUDASurfelReconstruction(
      usize max_surfel_count,
      const PinholeCamera4f& depth_camera,
      cudaGraphicsResource_t vertex_buffer_resource,
      cudaGraphicsResource_t neighbor_index_buffer_resource,
      cudaGraphicsResource_t normal_vertex_buffer_resource,
      const shared_ptr<SurfelMeshingRenderWindow>& render_window);
  
  // Destructor.
  ~CUDASurfelReconstruction();
  
  // Performs integration of a depth map into the surfel cloud.
  void Integrate(
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
      int surfel_integration_active_window_size);
  
  // Performs an extra (gradient descent) regularization iteration. Normally
  // this does not need to be called explicitly since Integrate() already does
  // it.
  void Regularize(
      cudaStream_t stream,
      u32 frame_index,
      float regularizer_weight,
      float radius_factor_for_regularization_neighbors,
      int regularization_frame_window_size);
  
  // Transfers all surfels to the CPU. The "buffers" object must be locked with
  // LockWriteBuffers() when this is called.
  void TransferAllToCPU(
      cudaStream_t stream,
      u32 frame_index,
      CUDASurfelsCPU* buffers);
  
  // Updates the visualization (vertex, index) buffers based on the surfels.
  void UpdateVisualizationBuffers(
      cudaStream_t stream,
      u32 frame_index,
      u32 latest_triangulated_frame_index,
      u32 latest_mesh_surfel_count,
      int surfel_integration_active_window_size,
      bool visualize_last_update_timestamp,
      bool visualize_creation_timestamp,
      bool visualize_radii,
      bool visualize_normals);
  
  // Exports surfel positions and colors to separate buffers.
  void ExportVertices(
      cudaStream_t stream,
      CUDABuffer<float>* position_buffer,
      CUDABuffer<u8>* color_buffer);
  
  // Determines and returns the CUDA timings for the last iteration.
  void GetTimings(
      float* data_association,
      float* surfel_merging,
      float* measurement_blending,
      float* integration,
      float* neighbor_update,
      float* new_surfel_creation,
      float* regularization);
  
  // Returns the current surfel count.
  inline u32 surfel_count() const { return surfel_count_ - merge_count_; }
  
  // Returns the number of surfel entries in use.
  inline u32 surfels_size() const { return surfel_count_; }
  
private:
  CUDABufferPtr<float> surfels_;
  
  CUDABufferPtr<u8> distance_map_;
  CUDABufferPtr<float> surfel_depth_average_deltas_;
  CUDABufferPtr<u8> new_distance_map_;
  CUDABufferPtr<float> new_surfel_depth_average_deltas_;
  
  CUDABufferPtr<u32> supporting_surfels_;
  CUDABufferPtr<u32> supporting_surfel_counts_;
  CUDABufferPtr<float> supporting_surfel_depth_sums_;
  CUDABufferPtr<u32> conflicting_surfels_;
  CUDABufferPtr<float> first_surfel_depth_;
  CUDABufferPtr<u8> new_surfel_flag_vector_;
  CUDABufferPtr<u32> new_surfel_indices_;
  
  void* new_surfels_temp_storage_;
  usize new_surfels_temp_storage_bytes_;
  
  u32 surfel_count_;
  u32 merge_count_;
  usize max_surfel_count_;
  
  const PinholeCamera4f& depth_camera_;
  
  cudaEvent_t preprocessing_start_event_;
  cudaEvent_t preprocessing_end_event_;
  cudaEvent_t data_association_start_event_;
  cudaEvent_t data_association_end_event_;
  cudaEvent_t surfel_merging_start_event_;
  cudaEvent_t surfel_merging_end_event_;
  cudaEvent_t measurement_blending_start_event_;
  cudaEvent_t measurement_blending_end_event_;
  cudaEvent_t integration_start_event_;
  cudaEvent_t integration_end_event_;
  cudaEvent_t neighbor_update_start_event_;
  cudaEvent_t neighbor_update_end_event_;
  cudaEvent_t new_surfel_creation_start_event_;
  cudaEvent_t new_surfel_creation_end_event_;
  cudaEvent_t regularization_start_event_;
  cudaEvent_t regularization_end_event_;
  
  cudaGraphicsResource_t vertex_buffer_resource_;
  cudaGraphicsResource_t neighbor_index_buffer_resource_;
  cudaGraphicsResource_t normal_vertex_buffer_resource_;
  shared_ptr<SurfelMeshingRenderWindow> render_window_;  // for debugging only
};

}
