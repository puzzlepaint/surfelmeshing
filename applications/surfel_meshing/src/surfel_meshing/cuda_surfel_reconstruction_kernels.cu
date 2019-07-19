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

#include "surfel_meshing/cuda_surfel_reconstruction_kernels.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <libvis/cuda/cuda_matrix.cuh>
#include <libvis/cuda/cuda_util.h>
#include <math_constants.h>

#include "surfel_meshing/cuda_util.cuh"

// Uncomment this to run CUDA kernels sequentially for debugging.
// #define CUDA_SEQUENTIAL_CHECKS

namespace vis {

// This threshold is not exposed as a program argument since I am not sure
// whether any other value than 0 would be useful.
constexpr float kSurfelNormalToViewingDirThreshold = 0;

// For a surfel with a given radius, the observation radius can be up to this
// factor worse (larger) while the observation is still integrated into the
// surfel. Observations with larger radii than that are discarded.
// TODO: Expose as a program argument?
constexpr float kMaxObservationRadiusFactorForIntegration = 1.5f;

// Not exposed as a program argument since it did not seem to work well.
constexpr bool kCheckScaleCompatibilityForIntegration = false;

// Not exposed as a program argument since disabling it might not make sense.
constexpr bool kCheckScaleCompatibilityForNeighborAssignment = true;

// If this is set to true, slightly occluded surfels will be protected better,
// but the surfel integration will be unable to merge duplicate surfaces after
// loop closures.
constexpr bool kProtectSlightlyOccludedSurfels = false;
constexpr float kOcclusionDepthFactor = 0.01f;

// This must be equal to kInvalidSurfelIndex, but we cannot include surfel.h
// here since it includes Eigen headers.
constexpr u32 kInvalidSurfelIndex = 4294967295;  // std::numeric_limits<u32>::max();


__forceinline__ __device__ bool IsSurfelActiveForIntegration(
    u32 surfel_index,
    const CUDABuffer_<float>& surfels,
    u32 frame_index,
    int surfel_integration_active_window_size) {
  // Alternatives:
  // kSurfelCreationStamp --> surfels are always deactivated after a certain time and never reactivated. Creates the least artifacts during deformations, but leads to many surfels.
  // kSurfelLastUpdateStamp --> surfels stay active. Leads to problems during deformation at observation boundaries (where the surfels are next to each other, but kSurfelLastUpdateStamp differs strongly).
  return static_cast<int>(*reinterpret_cast<const u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index))) >
             static_cast<int>(frame_index) - surfel_integration_active_window_size;
}


__global__ void CreateNewSurfelsCUDASerializingKernel(
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<u8> new_surfel_flag_vector) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    // TODO: Is this border necessary here, or should it rather be integrated into the depth map erosion?
    constexpr int kBorder = 1;
    bool new_surfel = x >= kBorder &&
                      y >= kBorder &&
                      x < depth_buffer.width() - kBorder &&
                      y < depth_buffer.height() - kBorder &&
                      depth_buffer(y, x) > 0 &&
                      supporting_surfels(y, x) == kInvalidSurfelIndex &&
                      conflicting_surfels(y, x) == kInvalidSurfelIndex;
    u32 seq_index = x + y * depth_buffer.width();
    new_surfel_flag_vector(0, seq_index) = new_surfel ? 1 : 0;
  }
}

void CallCreateNewSurfelsCUDASerializingKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<u32>& supporting_surfels,
    const CUDABuffer_<u32>& conflicting_surfels,
    const CUDABuffer_<u8>& new_surfel_flag_vector) {
  CreateNewSurfelsCUDASerializingKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      depth_buffer,
      supporting_surfels,
      conflicting_surfels,
      new_surfel_flag_vector);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

__global__ void CreateNewSurfelsCUDACreationKernel(
    u32 frame_index,
    float inv_depth_scaling,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    CUDAMatrix3x4 global_T_local,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<uchar3> color_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u8> new_surfel_flag_vector,
    CUDABuffer_<u32> new_surfel_indices,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float radius_factor_for_regularization_neighbors_squared) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    u32 seq_index = x + y * depth_buffer.width();
    if (new_surfel_flag_vector(0, seq_index) != 1) {
      return;
    }
    
    u32 surfel_index = surfel_count + new_surfel_indices(0, seq_index);
    
    float depth = inv_depth_scaling * depth_buffer(y, x);
    float3 local_position;
    UnprojectPoint(x, y, depth, fx_inv, fy_inv, cx_inv, cy_inv, &local_position);
    float3 global_position = global_T_local * local_position;
    
    surfels(kSurfelX, surfel_index) = global_position.x;
    surfels(kSurfelY, surfel_index) = global_position.y;
    surfels(kSurfelZ, surfel_index) = global_position.z;
    surfels(kSurfelSmoothX, surfel_index) = global_position.x;
    surfels(kSurfelSmoothY, surfel_index) = global_position.y;
    surfels(kSurfelSmoothZ, surfel_index) = global_position.z;
    
    float2 normal_xy = normals_buffer(y, x);
    const float normal_z = -sqrtf(::max(0.f, 1 - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y));
    float3 global_normal = global_T_local.Rotate(make_float3(normal_xy.x, normal_xy.y, normal_z));
    
    surfels(kSurfelNormalX, surfel_index) = global_normal.x;
    surfels(kSurfelNormalY, surfel_index) = global_normal.y;
    surfels(kSurfelNormalZ, surfel_index) = global_normal.z;
    
    uchar3 color = color_buffer(y, x);
    *(reinterpret_cast<uchar4*>(&surfels(kSurfelColor, surfel_index))) = make_uchar4(color.x, color.y, color.z, 0);
    
    surfels(kSurfelConfidence, surfel_index) = 1;
    *reinterpret_cast<u32*>(&surfels(kSurfelCreationStamp, surfel_index)) = frame_index;
    *reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index)) = frame_index;
    
    const float radius_squared = radius_buffer(y, x);
    surfels(kSurfelRadiusSquared, surfel_index) = radius_squared;
    
    // Determine initial neighbors.
    float3 neighbor_position_sum = make_float3(0, 0, 0);
    int existing_neighbor_count_plus_1 = 1;
    constexpr int kDirectionsX[4] = {-1, 1, 0, 0};
    constexpr int kDirectionsY[4] = {0, 0, -1, 1};
    for (int direction = 0; direction < 4; ++ direction) {
      u32 neighbor_index = supporting_surfels(y + kDirectionsY[direction], x + kDirectionsX[direction]);

      if (neighbor_index != kInvalidSurfelIndex) {
        float3 this_to_neighbor = make_float3(surfels(kSurfelX, neighbor_index) - global_position.x,
                                              surfels(kSurfelY, neighbor_index) - global_position.y,
                                              surfels(kSurfelZ, neighbor_index) - global_position.z);
        float distance_squared =
            this_to_neighbor.x * this_to_neighbor.x + this_to_neighbor.y * this_to_neighbor.y + this_to_neighbor.z * this_to_neighbor.z;
        if (distance_squared > radius_factor_for_regularization_neighbors_squared * radius_squared) {
          neighbor_index = kInvalidSurfelIndex;
        } else {
          neighbor_position_sum = make_float3(
              neighbor_position_sum.x + surfels(kSurfelSmoothX, neighbor_index),
              neighbor_position_sum.y + surfels(kSurfelSmoothY, neighbor_index),
              neighbor_position_sum.z + surfels(kSurfelSmoothZ, neighbor_index));
          ++ existing_neighbor_count_plus_1;
        }
      } else {
        u32 seq_neighbor_index = (x + kDirectionsX[direction]) + (y + kDirectionsY[direction]) * depth_buffer.width();
        if (new_surfel_flag_vector(0, seq_neighbor_index) == 1) {
          float other_depth = inv_depth_scaling * depth_buffer(y + kDirectionsY[direction], x + kDirectionsX[direction]);
          float approximate_distance_squared = (depth - other_depth) * (depth - other_depth);
          if (approximate_distance_squared <= radius_factor_for_regularization_neighbors_squared * radius_squared) {
            neighbor_index = surfel_count + new_surfel_indices(0, seq_neighbor_index);
          }
        }
      }
      
      *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + direction, surfel_index)) = neighbor_index;
    }
    
    // Try to get a better initialization for the regularized surfel position.
    surfels(kSurfelSmoothX, surfel_index) = (surfels(kSurfelSmoothX, surfel_index) + neighbor_position_sum.x) / existing_neighbor_count_plus_1;
    surfels(kSurfelSmoothY, surfel_index) = (surfels(kSurfelSmoothY, surfel_index) + neighbor_position_sum.y) / existing_neighbor_count_plus_1;
    surfels(kSurfelSmoothZ, surfel_index) = (surfels(kSurfelSmoothZ, surfel_index) + neighbor_position_sum.z) / existing_neighbor_count_plus_1;
  }
}

void CallCreateNewSurfelsCUDACreationKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    u32 frame_index,
    float inv_depth_scaling,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    CUDAMatrix3x4 global_T_local,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<uchar3> color_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u8> new_surfel_flag_vector,
    CUDABuffer_<u32> new_surfel_indices,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float radius_factor_for_regularization_neighbors_squared) {
  CreateNewSurfelsCUDACreationKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      inv_depth_scaling,
      fx_inv, fy_inv, cx_inv, cy_inv,
      global_T_local,
      depth_buffer,
      normals_buffer,
      radius_buffer,
      color_buffer,
      supporting_surfels,
      new_surfel_flag_vector,
      new_surfel_indices,
      surfel_count,
      surfels,
      radius_factor_for_regularization_neighbors_squared);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


template <bool visualize_last_update_timestamp,
          bool visualize_creation_timestamp,
          bool visualize_radii,
          bool visualize_normals>
__global__ void UpdateSurfelVertexBufferCUDAKernel(
    u32 frame_index,
    int surfel_integration_active_window_size,
    u32 point_size_in_floats,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    u32 latest_triangulated_frame_index,
    u32 latest_mesh_surfel_count,
    float* vertex_buffer_ptr) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    const u32 surfel_creation_stamp = *reinterpret_cast<u32*>(&surfels(kSurfelCreationStamp, surfel_index));
    // Only output if it is an old surfel that has not been replaced since the last mesh was created,
    // or if it is a new surfel which does not appear in the mesh yet.
    const bool output_vertex = surfel_creation_stamp <= latest_triangulated_frame_index ||
                               surfel_index >= latest_mesh_surfel_count;
    
    // Vertex layout (Point3fC3u8):
    // float x, float y, float z, u8 r, u8 g, u8 b, u8 unused;
    // Using NaN for one of the vertex coordinates to prevent it from being
    // drawn if the surfel was replaced recently and the triangulation not
    // adjusted yet. This makes the adjacent triangles disappear. Not sure
    // whether that is portable, but it works as intended on my system ...
    vertex_buffer_ptr[surfel_index * point_size_in_floats + 0] = output_vertex ? surfels(kSurfelSmoothX, surfel_index) : CUDART_NAN_F;
    vertex_buffer_ptr[surfel_index * point_size_in_floats + 1] = surfels(kSurfelSmoothY, surfel_index);
    vertex_buffer_ptr[surfel_index * point_size_in_floats + 2] = surfels(kSurfelSmoothZ, surfel_index);
    
    if (visualize_last_update_timestamp || visualize_creation_timestamp) {
      const u32 last_update_timestamp = *reinterpret_cast<u32*>(&surfels(visualize_creation_timestamp ? kSurfelCreationStamp : kSurfelLastUpdateStamp, surfel_index));
      const int age = frame_index - last_update_timestamp;
      constexpr int kVisualizationMinAge = 1;
      const int kVisualizationMaxAge = visualize_creation_timestamp ? 3000 : surfel_integration_active_window_size;
      if (age < kVisualizationMinAge) {
        // Special color for surfels updated in the last frame: red.
        uchar4 color = make_uchar4(255, 80, 80, 0);
        vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
      } else if (age > kVisualizationMaxAge) {
        // Old surfels: blue
        uchar4 color = make_uchar4(40, 40, 255, 0);
        vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
      } else {
        float blend_factor = (age - kVisualizationMinAge) * 1.0f / (kVisualizationMaxAge - kVisualizationMinAge);
        blend_factor = ::min(1.0f, ::max(0.0f, blend_factor));
        u8 intensity = 255 - static_cast<u8>(255.99f * blend_factor);
        uchar4 color = make_uchar4(intensity, intensity, intensity, 0);
        vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
      }
    } else if (visualize_radii) {
      const float radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
      const float radius = sqrtf(radius_squared);
      constexpr float kVisualizationMinRadius = 0.0005f;  // 0.5 mm
      constexpr float kVisualizationMaxRadius = 0.01f;   // 1 cm
      float blend_factor = (radius - kVisualizationMinRadius) / (kVisualizationMaxRadius - kVisualizationMinRadius);
      blend_factor = ::min(1.0f, ::max(0.0f, blend_factor));
      u8 red = 255.99f * blend_factor;
      u8 green = 255 - red;
      u8 blue = 80;
      uchar4 color = make_uchar4(red, green, blue, 0);
      vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
    } else if (visualize_normals) {
      float3 normal = make_float3(surfels(kSurfelNormalX, surfel_index),
                                  surfels(kSurfelNormalY, surfel_index),
                                  surfels(kSurfelNormalZ, surfel_index));
      uchar4 color = make_uchar4(255.99f / 2.0f * (normal.x + 1.0f),
                                 255.99f / 2.0f * (normal.y + 1.0f),
                                 255.99f / 2.0f * (normal.z + 1.0f),
                                 0);
      vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
    } else {
      vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = surfels(kSurfelColor, surfel_index);
    }
  }
}

void UpdateSurfelVertexBufferCUDA(
    cudaStream_t stream,
    u32 frame_index,
    int surfel_integration_active_window_size,
    u32 surfel_count,
    const CUDABuffer_<float>& surfels,
    u32 latest_triangulated_frame_index,
    u32 latest_mesh_surfel_count,
    cudaGraphicsResource_t vertex_buffer_resource,
    u32 point_size_in_floats,
    bool visualize_last_update_timestamp,
    bool visualize_creation_timestamp,
    bool visualize_radii,
    bool visualize_normals) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  // Map OpenGL buffer object for writing from CUDA.
  cudaGraphicsMapResources(1, &vertex_buffer_resource, stream);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  usize num_bytes;
  float* vertex_buffer_ptr;
  cudaGraphicsResourceGetMappedPointer((void**)&vertex_buffer_ptr, &num_bytes, vertex_buffer_resource);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  #define CALL_KERNEL(visualize_last_update_timestamp, \
                      visualize_creation_timestamp, \
                      visualize_radii, \
                      visualize_normals) \
      UpdateSurfelVertexBufferCUDAKernel \
      <visualize_last_update_timestamp, \
       visualize_creation_timestamp, \
       visualize_radii, \
       visualize_normals> \
      <<<grid_dim, block_dim, 0, stream>>>( \
          frame_index, \
          surfel_integration_active_window_size, \
          point_size_in_floats, \
          surfel_count, \
          surfels, \
          latest_triangulated_frame_index, \
          latest_mesh_surfel_count, \
          vertex_buffer_ptr)
  if (visualize_last_update_timestamp) {
    CALL_KERNEL(true, false, false, false);
  } else if (visualize_creation_timestamp) {
    CALL_KERNEL(false, true, false, false);
  } else if (visualize_radii) {
    CALL_KERNEL(false, false, true, false);
  } else if (visualize_normals) {
    CALL_KERNEL(false, false, false, true);
  }else {
    CALL_KERNEL(false, false, false, false);
  }
  #undef CALL_KERNEL
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  cudaGraphicsUnmapResources(1, &vertex_buffer_resource, stream);
}


__global__ void UpdateNeighborIndexBufferCUDAKernel(
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    unsigned int* neighbor_index_buffer_ptr) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    #pragma unroll
    for (int i = 0; i < kSurfelNeighborCount; ++ i) {
      neighbor_index_buffer_ptr[surfel_index * 2 * kSurfelNeighborCount + 2 * i + 0] = surfel_index;
      u32 neighbor_index = *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + i, surfel_index));
      neighbor_index_buffer_ptr[surfel_index * 2 * kSurfelNeighborCount + 2 * i + 1] =
          (neighbor_index == kInvalidSurfelIndex) ? surfel_index : neighbor_index;
    }
  }
}

void UpdateNeighborIndexBufferCUDA(
    cudaStream_t stream,
    u32 surfel_count,
    const CUDABuffer_<float>& surfels,
    cudaGraphicsResource_t neighbor_index_buffer_resource) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  // Map OpenGL buffer object for writing from CUDA.
  cudaGraphicsMapResources(1, &neighbor_index_buffer_resource, stream);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  usize num_bytes;
  unsigned int* index_buffer_ptr;
  cudaGraphicsResourceGetMappedPointer((void**)&index_buffer_ptr, &num_bytes, neighbor_index_buffer_resource);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  UpdateNeighborIndexBufferCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      surfel_count,
      surfels,
      index_buffer_ptr);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  cudaGraphicsUnmapResources(1, &neighbor_index_buffer_resource, stream);
}


__global__ void UpdateNormalVertexBufferCUDAKernel(
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float* normal_vertex_buffer_ptr) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    normal_vertex_buffer_ptr[6 * surfel_index + 0] = surfels(kSurfelSmoothX, surfel_index);
    normal_vertex_buffer_ptr[6 * surfel_index + 1] = surfels(kSurfelSmoothY, surfel_index);
    normal_vertex_buffer_ptr[6 * surfel_index + 2] = surfels(kSurfelSmoothZ, surfel_index);
    
    float radius = sqrtf(surfels(kSurfelRadiusSquared, surfel_index));
    normal_vertex_buffer_ptr[6 * surfel_index + 3] = surfels(kSurfelSmoothX, surfel_index) + radius * surfels(kSurfelNormalX, surfel_index);
    normal_vertex_buffer_ptr[6 * surfel_index + 4] = surfels(kSurfelSmoothY, surfel_index) + radius * surfels(kSurfelNormalY, surfel_index);
    normal_vertex_buffer_ptr[6 * surfel_index + 5] = surfels(kSurfelSmoothZ, surfel_index) + radius * surfels(kSurfelNormalZ, surfel_index);
  }
}

void UpdateNormalVertexBufferCUDA(
    cudaStream_t stream,
    u32 surfel_count,
    const CUDABuffer_<float>& surfels,
    cudaGraphicsResource_t normal_vertex_buffer_resource) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  // Map OpenGL buffer object for writing from CUDA.
  cudaGraphicsMapResources(1, &normal_vertex_buffer_resource, stream);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  usize num_bytes;
  float* vertex_buffer_ptr;
  cudaGraphicsResourceGetMappedPointer((void**)&vertex_buffer_ptr, &num_bytes, normal_vertex_buffer_resource);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  UpdateNormalVertexBufferCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      surfel_count,
      surfels,
      vertex_buffer_ptr);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  cudaGraphicsUnmapResources(1, &normal_vertex_buffer_resource, stream);
}


__global__ void BlendMeasurementsCUDAStartKernel(
    float depth_scaling,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<float> supporting_surfel_depth_sums,
    CUDABuffer_<u8> distance_map,
    CUDABuffer_<float> surfel_depth_average_deltas,
    CUDABuffer_<u8> new_distance_map,
    CUDABuffer_<float> new_surfel_depth_average_deltas) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  constexpr int kBorder = 1;
  if (x >= kBorder && y >= kBorder && x < supporting_surfels.width() - kBorder && y < supporting_surfels.height() - kBorder) {
    // Only consider pixels with valid measurement depth and supporting surfels.
    if (depth_buffer(y, x) == 0 || supporting_surfels(y, x) == kInvalidSurfelIndex) {
      return;
    }
    
    bool measurement_border_pixel = false;
    bool surfel_border_pixel = false;
    for (int wy = y - 1, wy_end = y + 1; wy <= wy_end; ++ wy) {
      for (int wx = x - 1, wx_end = x + 1; wx <= wx_end; ++ wx) {
        if (depth_buffer(wy, wx) == 0) {
          measurement_border_pixel = true;
        } else if (supporting_surfels(wy, wx) == kInvalidSurfelIndex) {
          surfel_border_pixel = true;
        }
      }
    }
    
    if (surfel_border_pixel) {
      // TODO: Interpolation should start at the depth after this iteration's integration in this case
      
      new_distance_map(y, x) = 1;
      
      float surfel_depth_average = supporting_surfel_depth_sums(y, x) / supporting_surfel_counts(y, x);
      new_surfel_depth_average_deltas(y, x) = surfel_depth_average - depth_buffer(y, x) / depth_scaling;
    }
    
    if (measurement_border_pixel) {
      distance_map(y, x) = 1;
      
      float surfel_depth_average = supporting_surfel_depth_sums(y, x) / supporting_surfel_counts(y, x);
      surfel_depth_average_deltas(y, x) = surfel_depth_average - depth_buffer(y, x) / depth_scaling;
      
      depth_buffer(y, x) = depth_scaling * surfel_depth_average + 0.5f;  // TODO: This assignment can happen while other threads read, does it matter?
    } else {
      distance_map(y, x) = 255;  // unknown distance
    }
  }
}

void CallBlendMeasurementsCUDAStartKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    float depth_scaling,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<float> supporting_surfel_depth_sums,
    CUDABuffer_<u8> distance_map,
    CUDABuffer_<float> surfel_depth_average_deltas,
    CUDABuffer_<u8> new_distance_map,
    CUDABuffer_<float> new_surfel_depth_average_deltas) {
  BlendMeasurementsCUDAStartKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      depth_scaling,
      depth_buffer,
      supporting_surfels,
      supporting_surfel_counts,
      supporting_surfel_depth_sums,
      distance_map,
      surfel_depth_average_deltas,
      new_distance_map,
      new_surfel_depth_average_deltas);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

__global__ void BlendMeasurementsCUDAIterationKernel(
    int iteration,
    float interpolation_factor_term,
    float depth_scaling,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u8> distance_map,
    CUDABuffer_<float> surfel_depth_average_deltas,
    CUDABuffer_<u8> new_distance_map,
    CUDABuffer_<float> new_surfel_depth_average_deltas) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  constexpr int kBorder = 1;
  if (x >= kBorder && y >= kBorder && x < supporting_surfels.width() - kBorder && y < supporting_surfels.height() - kBorder) {
    if (distance_map(y, x) == 255) {  // unknown distance
      float delta_sum = 0;
      int count = 0;
      
      for (int wy = y - 1, wy_end = y + 1; wy <= wy_end; ++ wy) {
        for (int wx = x - 1, wx_end = x + 1; wx <= wx_end; ++ wx) {
          if (distance_map(wy, wx) == iteration - 1) {
            delta_sum += surfel_depth_average_deltas(wy, wx);
            ++ count;
          }
        }
      }
      
      if (count > 0) {
        distance_map(y, x) = iteration;  // TODO: This assignment can happen while other threads read, does it matter?
        float surfel_delta_average = delta_sum / count;
        surfel_depth_average_deltas(y, x) = surfel_delta_average;
        
        float interpolation_factor = (iteration - 1) * interpolation_factor_term;
        depth_buffer(y, x) += depth_scaling * (1 - interpolation_factor) * surfel_delta_average + 0.5f;
      }
    }
    
    if (depth_buffer(y, x) != 0 && supporting_surfels(y, x) == kInvalidSurfelIndex && new_distance_map(y, x) == 0) {
      float delta_sum = 0;
      int count = 0;
      
      for (int wy = y - 1, wy_end = y + 1; wy <= wy_end; ++ wy) {
        for (int wx = x - 1, wx_end = x + 1; wx <= wx_end; ++ wx) {
          if (new_distance_map(wy, wx) == iteration - 1) {
            delta_sum += new_surfel_depth_average_deltas(wy, wx);
            ++ count;
          }
        }
      }
      
      if (count > 0) {
        new_distance_map(y, x) = iteration;  // TODO: This assignment can happen while other threads read, does it matter?
        float surfel_delta_average = delta_sum / count;
        new_surfel_depth_average_deltas(y, x) = surfel_delta_average;
        
        float interpolation_factor = (iteration - 1) * interpolation_factor_term;
        depth_buffer(y, x) += depth_scaling * (1 - interpolation_factor) * surfel_delta_average + 0.5f;
      }
    }
  }
}

void CallBlendMeasurementsCUDAIterationKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    int iteration,
    float interpolation_factor_term,
    float depth_scaling,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u8> distance_map,
    CUDABuffer_<float> surfel_depth_average_deltas,
    CUDABuffer_<u8> new_distance_map,
    CUDABuffer_<float> new_surfel_depth_average_deltas) {
  BlendMeasurementsCUDAIterationKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      iteration,
      interpolation_factor_term,
      depth_scaling,
      depth_buffer,
      supporting_surfels,
      distance_map,
      surfel_depth_average_deltas,
      new_distance_map,
      new_surfel_depth_average_deltas);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__device__ void IntegrateOrConflictSurfel(
    bool integrate, u32 frame_index, int x, int y,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    const float3& cam_space_surfel_pos,
    unsigned int surfel_index,
    CUDABuffer_<float>& surfels,
    const CUDAMatrix3x4& local_T_global,
    const CUDAMatrix3x4& global_T_local,
    float max_surfel_confidence,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    float depth_correction_factor,
    const CUDABuffer_<u16>& depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<uchar3> color_buffer,
    CUDABuffer_<u32>& /*supporting_surfels*/,
    CUDABuffer_<u32>& supporting_surfel_counts,
    CUDABuffer_<u32>& conflicting_surfels,
    CUDABuffer_<float>& first_surfel_depth) {
  // Check whether the surfel falls on a depth pixel.
  float measurement_depth = depth_correction_factor * depth_buffer(y, x);
  if (measurement_depth <= 0) {
    integrate = false;
  }
  if (!__syncthreads_or(integrate)) {
    return;
  }
  
  // Check if this or another surfel is conflicting.
  bool conflicting = false;
  const float first_surfel_depth_value = first_surfel_depth(y, x);
  if (first_surfel_depth_value < (1 - sensor_noise_factor) * measurement_depth) {
    // This or another surfel is conflicting.
    if (first_surfel_depth_value == cam_space_surfel_pos.z) {
      // This surfel is conflicting with the measurement.
      if (conflicting_surfels(y, x) == surfel_index) {
        conflicting = integrate;
      }
    }
    integrate = false;
  }
  if (!__syncthreads_or(integrate || conflicting)) {
    return;
  }
  
  // Determine the depth from which on surfels are considered to be occluded.
  float occlusion_depth = (1 + sensor_noise_factor) * measurement_depth;
  if (kProtectSlightlyOccludedSurfels && first_surfel_depth_value < occlusion_depth) {
    // TODO: Would it be better to use the front surfel's radius for that?
    occlusion_depth = (1 + kOcclusionDepthFactor) * first_surfel_depth_value;
  }
  
  // Check whether this surfel is occluded.
  if (cam_space_surfel_pos.z > occlusion_depth) {
    // Surfel is occluded.
    integrate = false;
  }
  if (!__syncthreads_or(integrate || conflicting)) {
    return;
  }
  
  
  // Read data.
  float depth = depth_correction_factor * depth_buffer(y, x);
  float3 local_position;
  UnprojectPoint(x, y, depth, fx_inv, fy_inv, cx_inv, cy_inv, &local_position);
  float3 global_position = global_T_local * local_position;
  
  float2 normal_xy = normals_buffer(y, x);
  const float normal_z = -sqrtf(::max(0.f, 1 - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y));
  float3 global_normal = global_T_local.Rotate(make_float3(normal_xy.x, normal_xy.y, normal_z));
  
  uchar3 color = color_buffer(y, x);
  
  // Handle conflicts.
  // Critical section. HACK: replace surfel x coordinate with NaN to signal locked state.
  while (__syncthreads_or(conflicting)) {
    float assumed_x = surfels(kSurfelX, surfel_index);
    if (conflicting &&
        !::isnan(assumed_x) &&
        atomicCAS(reinterpret_cast<int*>(&surfels(kSurfelX, surfel_index)),
                  __float_as_int(assumed_x),
                  __float_as_int(CUDART_NAN_F)) == __float_as_int(assumed_x)) {
      // Handle the conflict with surfel_index.
      float confidence = surfels(kSurfelConfidence, surfel_index);
      confidence -= 1;
      if (confidence <= 0) {
        // Delete the old surfel by replacing it with a new one.
        assumed_x = global_position.x;
        surfels(kSurfelY, surfel_index) = global_position.y;
        surfels(kSurfelZ, surfel_index) = global_position.z;
        
        surfels(kSurfelSmoothX, surfel_index) = global_position.x;
        surfels(kSurfelSmoothY, surfel_index) = global_position.y;
        surfels(kSurfelSmoothZ, surfel_index) = global_position.z;
        
        surfels(kSurfelNormalX, surfel_index) = global_normal.x;
        surfels(kSurfelNormalY, surfel_index) = global_normal.y;
        surfels(kSurfelNormalZ, surfel_index) = global_normal.z;
        
        *(reinterpret_cast<uchar4*>(&surfels(kSurfelColor, surfel_index))) = make_uchar4(color.x, color.y, color.z, 1);  // Sets the neighbor detach request flag.
        
        surfels(kSurfelRadiusSquared, surfel_index) = radius_buffer(y, x);
        
        #pragma unroll
        for (int i = 0; i < kSurfelNeighborCount; ++ i) {
          // TODO: (Sh/c)ould the neighbors be initialized to something here instead of being removed completely?
          *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + i, surfel_index)) = kInvalidSurfelIndex;
        }
        
        surfels(kSurfelConfidence, surfel_index) = 1;
        *reinterpret_cast<u32*>(&surfels(kSurfelCreationStamp, surfel_index)) = frame_index;
        *reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index)) = frame_index;
      } else {
        surfels(kSurfelConfidence, surfel_index) = confidence;
      }
      
      // Release lock by setting x coordinate.
      // Not sure whether the atomicExch is necessary here, an atomic assignment would suffice.
      atomicExch(reinterpret_cast<int*>(&surfels(kSurfelX, surfel_index)), __float_as_int(assumed_x));
      
      conflicting = false;
    }
    // Force execution of the if case to avoid hang coming from the fact that
    // only the threads which don't go into the if case are executed otherwise.
    __syncthreads();
  }
  
  // Early exit if none of the threads in the warp needs to integrate data.
  if (!__syncthreads_or(integrate)) {
    return;
  }
  
  // The measurement supports the surfel. Determine whether they belong to the
  // same surface (then the measurement should be integrated into the surfel),
  // or to different surfaces (then the measurement must not be integrated).
  
  // Check whether the surfel normal looks towards the camera (instead of away from it).
  float surfel_distance = Norm(cam_space_surfel_pos);
  float3 global_surfel_normal = make_float3(surfels(kSurfelNormalX, surfel_index),
                                            surfels(kSurfelNormalY, surfel_index),
                                            surfels(kSurfelNormalZ, surfel_index));
  float3 local_surfel_normal = local_T_global.Rotate(global_surfel_normal);
  
  float dot_angle = (1.0f / surfel_distance) * (cam_space_surfel_pos.x * local_surfel_normal.x +
                                                cam_space_surfel_pos.y * local_surfel_normal.y +
                                                cam_space_surfel_pos.z * local_surfel_normal.z);
  if (dot_angle > kSurfelNormalToViewingDirThreshold) {
    integrate = false;
  }
  if (!__syncthreads_or(integrate)) {
    return;
  }
  
  // Check whether the surfel normal is compatible with the measurement normal.
  if (measurement_depth < cam_space_surfel_pos.z) {
    float dot_angle = global_surfel_normal.x * global_normal.x +
                      global_surfel_normal.y * global_normal.y +
                      global_surfel_normal.z * global_normal.z;
    if (dot_angle < cos_normal_compatibility_threshold) {
      integrate = false;
    }
  }
  
  // Check whether the observation scale is compatible with the surfel scale.
  const float surfel_radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
  if (surfel_radius_squared < 0) {
    integrate = false;
  }
  if (kCheckScaleCompatibilityForIntegration) {
    const float observation_radius_squared = radius_buffer(y, x);
    if (observation_radius_squared / surfel_radius_squared > kMaxObservationRadiusFactorForIntegration * kMaxObservationRadiusFactorForIntegration) {
      integrate = false;
    }
    if (!__syncthreads_or(integrate)) {
      return;
    }
  }
  
  
  // Integrate.
  // Critical section. HACK: replace surfel x coordinate with NaN to signal locked state.
  __syncthreads();  // Not sure if necessary
  while (__syncthreads_or(integrate)) {
    const float assumed_x = surfels(kSurfelX, surfel_index);
    if (integrate &&
        !::isnan(assumed_x) &&
        atomicCAS(reinterpret_cast<int*>(&surfels(kSurfelX, surfel_index)),
                  __float_as_int(assumed_x),
                  __float_as_int(CUDART_NAN_F)) == __float_as_int(assumed_x)) {
      // TODO: Check why this max(1, ...) is necessary
      const float weight = 1.0f / ::max(1, supporting_surfel_counts(y, x));
      
      float new_surfel_x = assumed_x;
      
      // If the surfel has been created (i.e., replaced) in this iteration, do not
      // integrate the data, since the association is probably not valid anymore.
      // Also, the neighbor detach request flag should be kept in that case.
      if (*reinterpret_cast<u32*>(&surfels(kSurfelCreationStamp, surfel_index)) < frame_index) {
        const float confidence = surfels(kSurfelConfidence, surfel_index);
        surfels(kSurfelConfidence, surfel_index) =
            (confidence + weight < max_surfel_confidence) ?
            (confidence + weight) :
            max_surfel_confidence;
        float normalization_factor = 1.0f / (confidence + weight);
        
        new_surfel_x = (confidence * assumed_x + weight * global_position.x) * normalization_factor;  // assumed_x is the old surfel x value.
        surfels(kSurfelY, surfel_index) = (confidence * surfels(kSurfelY, surfel_index) + weight * global_position.y) * normalization_factor;
        surfels(kSurfelZ, surfel_index) = (confidence * surfels(kSurfelZ, surfel_index) + weight * global_position.z) * normalization_factor;
        
        float3 new_normal = make_float3(confidence * surfels(kSurfelNormalX, surfel_index) + weight * global_normal.x,
                                        confidence * surfels(kSurfelNormalY, surfel_index) + weight * global_normal.y,
                                        confidence * surfels(kSurfelNormalZ, surfel_index) + weight * global_normal.z);
        float normal_normalization = 1.0f / sqrtf(new_normal.x * new_normal.x + new_normal.y * new_normal.y + new_normal.z * new_normal.z);
        surfels(kSurfelNormalX, surfel_index) = normal_normalization * new_normal.x;
        surfels(kSurfelNormalY, surfel_index) = normal_normalization * new_normal.y;
        surfels(kSurfelNormalZ, surfel_index) = normal_normalization * new_normal.z;
        
        surfels(kSurfelRadiusSquared, surfel_index) = ::min(surfels(kSurfelRadiusSquared, surfel_index), radius_buffer(y, x));
        
        const uchar4 old_color = *(reinterpret_cast<uchar4*>(&surfels(kSurfelColor, surfel_index)));
        const uchar3 new_color = make_uchar3(
            (confidence * old_color.x + weight * color.x) * normalization_factor + 0.5f,
            (confidence * old_color.y + weight * color.y) * normalization_factor + 0.5f,
            (confidence * old_color.z + weight * color.z) * normalization_factor + 0.5f);
        *(reinterpret_cast<uchar4*>(&surfels(kSurfelColor, surfel_index))) = make_uchar4(new_color.x, new_color.y, new_color.z, 0);  // NOTE: Unsets the neighbor detach request flag
        
        *reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index)) = frame_index;
      }
      
      // Release lock by setting x coordinate.
      // Not sure whether the atomicExch is necessary here, an atomic assignment would suffice.
      atomicExch(reinterpret_cast<int*>(&surfels(kSurfelX, surfel_index)), __float_as_int(new_surfel_x));
      
      integrate = false;
    }
    // Force execution of the if case to avoid hang coming from the fact that
    // only the threads which don't go into the if case are executed otherwise.
    __syncthreads();
  }
}

__launch_bounds__(/*maxThreadsPerBlock*/ 1024, /*minBlocksPerMultiprocessor*/ 1)
__global__ void IntegrateMeasurementsCUDAKernel(
    u32 frame_index,
    int surfel_integration_active_window_size,
    float max_surfel_confidence,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    float inv_depth_scaling,
    float fx, float fy, float cx, float cy,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    CUDAMatrix3x4 local_T_global,
    CUDAMatrix3x4 global_T_local,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<uchar3> color_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<float> first_surfel_depth,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  bool integrate = true;
  
  // Check whether the surfel projects onto the image. Keep all threads active
  // such that the __syncthreads() later will work.
  if (surfel_index >= surfel_count) {
    surfel_index = 0;
    integrate = false;
  }
  
  if (!IsSurfelActiveForIntegration(surfel_index, surfels, frame_index, surfel_integration_active_window_size)) {
    integrate = false;
  }
  if (!__syncthreads_or(integrate)) {
    return;
  }
  
  float3 global_position =
      make_float3(surfels(kSurfelX, surfel_index),
                  surfels(kSurfelY, surfel_index),
                  surfels(kSurfelZ, surfel_index));
  float3 local_position = local_T_global * global_position;
  if (local_position.z <= 0) {
    // TODO: Compute z before x and y such that this early exit decision can be done earlier?
    integrate = false;
  }
  // Early exit?
  if (!__syncthreads_or(integrate)) {
    return;
  }
  
  float2 pixel_pos =
      make_float2(fx * (local_position.x / local_position.z) + cx,
                  fy * (local_position.y / local_position.z) + cy);
  int px = static_cast<int>(pixel_pos.x);
  int py = static_cast<int>(pixel_pos.y);
  if (pixel_pos.x < 0 || pixel_pos.y < 0 ||
      px < 0 || py < 0 ||
      px >= depth_buffer.width() || py >= depth_buffer.height()) {
    px = 0;
    py = 0;
    integrate = false;
  }
  
  if (surfels(kSurfelRadiusSquared, surfel_index) < 0) {
    integrate = false;
  }
  
  // Early exit?
  if (!__syncthreads_or(integrate)) {
    return;
  }
  
  IntegrateOrConflictSurfel(
      integrate, frame_index, px, py,
      fx_inv, fy_inv, cx_inv, cy_inv,
      local_position,
      surfel_index, surfels,
      local_T_global,
      global_T_local,
      max_surfel_confidence,
      sensor_noise_factor,
      cos_normal_compatibility_threshold,
      inv_depth_scaling, depth_buffer,
      normals_buffer,
      radius_buffer,
      color_buffer,
      supporting_surfels,
      supporting_surfel_counts,
      conflicting_surfels,
      first_surfel_depth);
  
  float x_frac = pixel_pos.x - px;
  float y_frac = pixel_pos.y - py;
  int offset_x = 0;
  int offset_y = 0;
  if (x_frac < y_frac) {
    // Surfel is within the bottom-left triangle half of the pixel.
    if (x_frac < 1 - y_frac) {
      // Surfel is on the left side of the pixel.
      if (px > 1) {
        offset_x = px - 1;
        offset_y = py;
      } else {
        integrate = false;
      }
    } else {
      // Surfel is on the bottom side of the pixel.
      if (py < depth_buffer.height() - 1) {
        offset_x = px;
        offset_y = py + 1;
      } else {
        integrate = false;
      }
    }
  } else {
    // Surfel is within the top-right triangle half of the pixel.
    if (x_frac < 1 - y_frac) {
      // Surfel is on the top side of the pixel.
      if (py > 0) {
        offset_x = px;
        offset_y = py - 1;
      } else {
        integrate = false;
      }
    } else {
      // Surfel is on the right side of the pixel.
      if (px < depth_buffer.width() - 1) {
        offset_x = px + 1;
        offset_y = py;
      } else {
        integrate = false;
      }
    }
  }
  
  IntegrateOrConflictSurfel(
      integrate, frame_index, offset_x, offset_y,
      fx_inv, fy_inv, cx_inv, cy_inv,
      local_position,
      surfel_index, surfels,
      local_T_global,
      global_T_local,
      max_surfel_confidence,
      sensor_noise_factor,
      cos_normal_compatibility_threshold,
      inv_depth_scaling, depth_buffer,
      normals_buffer,
      radius_buffer,
      color_buffer,
      supporting_surfels,
      supporting_surfel_counts,
      conflicting_surfels,
      first_surfel_depth);
  
  // TODO: use half integration weight if the surfel is associated to two pixels?
}

void CallIntegrateMeasurementsCUDAKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float max_surfel_confidence,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    float inv_depth_scaling,
    float fx, float fy, float cx, float cy,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    CUDAMatrix3x4 local_T_global,
    CUDAMatrix3x4 global_T_local,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<uchar3> color_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<float> first_surfel_depth,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  IntegrateMeasurementsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      surfel_integration_active_window_size,
      max_surfel_confidence,
      sensor_noise_factor,
      cos_normal_compatibility_threshold,
      inv_depth_scaling,
      fx, fy, cx, cy,
      fx_inv, fy_inv, cx_inv, cy_inv,
      local_T_global,
      global_T_local,
      depth_buffer,
      normals_buffer,
      radius_buffer,
      color_buffer,
      supporting_surfels,
      supporting_surfel_counts,
      conflicting_surfels,
      first_surfel_depth,
      surfel_count,
      surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__global__ void UpdateNeighborsCUDAKernel(
    u32 frame_index,
    int surfel_integration_active_window_size,
    float radius_factor_for_regularization_neighbors_squared,
    CUDABuffer_<u32> supporting_surfels,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    float sensor_noise_factor,
    float depth_correction_factor,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<float> first_surfel_depth,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    if (!IsSurfelActiveForIntegration(surfel_index, surfels, frame_index, surfel_integration_active_window_size)) {
      return;
    }
    
    // Project the surfel into the image.
    float3 global_position =
        make_float3(surfels(kSurfelX, surfel_index),
                    surfels(kSurfelY, surfel_index),
                    surfels(kSurfelZ, surfel_index));
    float3 cam_space_surfel_pos = local_T_global * global_position;
    if (cam_space_surfel_pos.z <= 0) {
      // TODO: Compute z before x and y such that this early exit decision can be done earlier?
      return;
    }
    
    float2 pixel_pos =
        make_float2(fx * (cam_space_surfel_pos.x / cam_space_surfel_pos.z) + cx,
                    fy * (cam_space_surfel_pos.y / cam_space_surfel_pos.z) + cy);
    int x = static_cast<int>(pixel_pos.x);
    int y = static_cast<int>(pixel_pos.y);
    
    // Use 1 pixel border.
    constexpr int kBorder = 1;
    if (x < kBorder || y < kBorder ||
        x >= supporting_surfels.width() - kBorder || y >= supporting_surfels.height() - kBorder) {
      return;
    }
    
    // Is the surfel occluded?
    float measurement_depth = depth_correction_factor * depth_buffer(y, x);
    float occlusion_depth = (1 + sensor_noise_factor) * measurement_depth;
    if (kProtectSlightlyOccludedSurfels) {
      const float first_surfel_depth_value = first_surfel_depth(y, x);
      if (first_surfel_depth_value < occlusion_depth) {
        // TODO: Would it be better to use the front surfel's radius for that?
        occlusion_depth = (1 + kOcclusionDepthFactor) * first_surfel_depth_value;
      }
    }
    if (cam_space_surfel_pos.z > occlusion_depth) {
      return;
    }
    
    // Check whether the surfel normal looks towards the camera (instead of away from it).
    float surfel_distance = Norm(cam_space_surfel_pos);
    float3 global_surfel_normal = make_float3(surfels(kSurfelNormalX, surfel_index),
                                              surfels(kSurfelNormalY, surfel_index),
                                              surfels(kSurfelNormalZ, surfel_index));
    float3 local_surfel_normal = local_T_global.Rotate(global_surfel_normal);
    
    float dot_angle = (1.0f / surfel_distance) * (cam_space_surfel_pos.x * local_surfel_normal.x +
                                                  cam_space_surfel_pos.y * local_surfel_normal.y +
                                                  cam_space_surfel_pos.z * local_surfel_normal.z);
    if (dot_angle > kSurfelNormalToViewingDirThreshold) {
      return;
    }
    
    // Check whether the surfel normal is compatible with the measurement normal (if enabled).
    /*if (measurement_depth < cam_space_surfel_pos.z) {
      float2 normal = normals_buffer(y, x);
      float3 local_normal = make_float3(normal.x, normal.y, -sqrtf(::max(0.f, 1 - normal.x * normal.x - normal.y * normal.y)));
      
      float dot_angle = local_surfel_normal.x * local_normal.x +
                        local_surfel_normal.y * local_normal.y +
                        local_surfel_normal.z * local_normal.z;
      if (dot_angle < kNormalCompatibilityThreshold) {
        return;
      }
    }*/
    
    const float surfel_radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
    if (surfel_radius_squared < 0) {
      return;
    }
    if (kCheckScaleCompatibilityForNeighborAssignment) {
      const float observation_radius_squared = radius_buffer(y, x);
      if (observation_radius_squared / surfel_radius_squared > kMaxObservationRadiusFactorForIntegration * kMaxObservationRadiusFactorForIntegration) {
        return;
      }
    }
    
    // We think that the surfel is visible, update its neighbors.
    
    float radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
    
    float3 global_normal =
        make_float3(surfels(kSurfelNormalX, surfel_index),
                    surfels(kSurfelNormalY, surfel_index),
                    surfels(kSurfelNormalZ, surfel_index));
    
    // Compute distances to existing neighbors.
    float neighbor_distances_squared[kSurfelNeighborCount];
    u32 neighbor_surfel_indices[kSurfelNeighborCount];
    for (int n = 0; n < kSurfelNeighborCount; ++ n) {
      neighbor_surfel_indices[n] = *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + n, surfel_index));
      if (neighbor_surfel_indices[n] == kInvalidSurfelIndex) {
        neighbor_distances_squared[n] = CUDART_INF_F;
      } else {
        float3 neighbor_position =
            make_float3(surfels(kSurfelX, neighbor_surfel_indices[n]),
                        surfels(kSurfelY, neighbor_surfel_indices[n]),
                        surfels(kSurfelZ, neighbor_surfel_indices[n]));
        float3 surfel_to_neighbor = make_float3(
            global_position.x - neighbor_position.x,
            global_position.y - neighbor_position.y,
            global_position.z - neighbor_position.z);
        neighbor_distances_squared[n] = surfel_to_neighbor.x * surfel_to_neighbor.x + surfel_to_neighbor.y * surfel_to_neighbor.y + surfel_to_neighbor.z * surfel_to_neighbor.z;
      }
    }
    
    constexpr int kDirectionsX[4] = {-1, 1, 0, 0};
    constexpr int kDirectionsY[4] = {0, 0, -1, 1};
    for (int direction = 0; direction < 4; ++ direction) {
      u32 neighbor_index = supporting_surfels(y + kDirectionsY[direction], x + kDirectionsX[direction]);
      if (neighbor_index != kInvalidSurfelIndex &&
          neighbor_index != surfel_index) {
        // Check for closeness.
        float3 this_to_neighbor = make_float3(surfels(kSurfelX, neighbor_index) - global_position.x,
                                              surfels(kSurfelY, neighbor_index) - global_position.y,
                                              surfels(kSurfelZ, neighbor_index) - global_position.z);
        float distance_squared =
            this_to_neighbor.x * this_to_neighbor.x + this_to_neighbor.y * this_to_neighbor.y + this_to_neighbor.z * this_to_neighbor.z;
        if (distance_squared > radius_factor_for_regularization_neighbors_squared * radius_squared) {
          neighbor_index = kInvalidSurfelIndex;
        }
        
        if (neighbor_index != kInvalidSurfelIndex) {
          // Check for compatible normal.
          float3 neighbor_normal =
              make_float3(surfels(kSurfelNormalX, neighbor_index),
                          surfels(kSurfelNormalY, neighbor_index),
                          surfels(kSurfelNormalZ, neighbor_index));
          float normal_dot = global_normal.x * neighbor_normal.x +
                             global_normal.y * neighbor_normal.y +
                             global_normal.z * neighbor_normal.z;
          if (normal_dot <= 0) {
            neighbor_index = kInvalidSurfelIndex;
          }
          
          if (neighbor_index != kInvalidSurfelIndex) {
            // Check whether it is already a neighbor, or find the best insertion slot.
            int best_n = -1;
            float best_distance_squared = -1;
            for (int n = 0; n < kSurfelNeighborCount; ++ n) {
              if (neighbor_index == neighbor_surfel_indices[n]) {
                best_n = -1;
                break;
              } else if (neighbor_distances_squared[n] > best_distance_squared) {
                best_n = n;
                best_distance_squared = neighbor_distances_squared[n];
              }
            }
            
            if (best_n >= 0 && distance_squared < best_distance_squared) {
              neighbor_surfel_indices[best_n] = neighbor_index;
              neighbor_distances_squared[best_n] = distance_squared;
            }
          }
        }
      }
    }
    
    // Write the neighbor indices back to global memory.
    for (int n = 0; n < kSurfelNeighborCount; ++ n) {
      *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + n, surfel_index)) = neighbor_surfel_indices[n];
    }
  }
}

void CallUpdateNeighborsCUDAKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float radius_factor_for_regularization_neighbors_squared,
    CUDABuffer_<u32> supporting_surfels,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    float sensor_noise_factor,
    float depth_correction_factor,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<float> first_surfel_depth,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  UpdateNeighborsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      surfel_integration_active_window_size,
      radius_factor_for_regularization_neighbors_squared,
      supporting_surfels,
      fx, fy, cx, cy,
      local_T_global,
      sensor_noise_factor,
      depth_correction_factor,
      depth_buffer,
      radius_buffer,
      first_surfel_depth,
      surfel_count,
      surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

__global__ void UpdateNeighborsCUDARemoveReplacedNeighborsKernel(
    u32 frame_index,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    for (int neighbor_index = 0; neighbor_index < kSurfelNeighborCount; ++ neighbor_index) {
      u32 neighbor_surfel_index = *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + neighbor_index, surfel_index));
      if (neighbor_surfel_index != kInvalidSurfelIndex) {
        if (*reinterpret_cast<u8*>(&reinterpret_cast<uchar4*>(&surfels(kSurfelColor, neighbor_surfel_index))->w) == 1) {
          // This neighbor has the neighbor detach request flag set. Remove it.
          *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + neighbor_index, surfel_index)) = kInvalidSurfelIndex;
        }
      }
    }
  }
}

void CallUpdateNeighborsCUDARemoveReplacedNeighborsKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    u32 frame_index,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  UpdateNeighborsCUDARemoveReplacedNeighborsKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      surfel_count,
      surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__forceinline__ __device__ void RenderMinDepthAtPixel(
    int x, int y,
    const float3& cam_space_surfel_pos,
    CUDABuffer_<float>& first_surfel_depth) {
  // Should behave properly as long as all the floats are positive.
  atomicMin(reinterpret_cast<int*>(&first_surfel_depth(y, x)), __float_as_int(cam_space_surfel_pos.z));
}

__global__ void RenderMinDepthCUDAKernel(
    u32 frame_index,
    int surfel_integration_active_window_size,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    CUDABuffer_<float> first_surfel_depth) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    if (!IsSurfelActiveForIntegration(surfel_index, surfels, frame_index, surfel_integration_active_window_size)) {
      return;
    }
    
    float3 global_position =
        make_float3(surfels(kSurfelX, surfel_index),
                    surfels(kSurfelY, surfel_index),
                    surfels(kSurfelZ, surfel_index));
    float3 local_position = local_T_global * global_position;
    if (local_position.z <= 0) {
      // TODO: Compute z before x and y such that this early exit can be done earlier?
      return;
    }
    
    float2 pixel_pos =
        make_float2(fx * (local_position.x / local_position.z) + cx,
                    fy * (local_position.y / local_position.z) + cy);
    int px = static_cast<int>(pixel_pos.x);
    int py = static_cast<int>(pixel_pos.y);
    if (pixel_pos.x < 0 || pixel_pos.y < 0 ||
        px < 0 || py < 0 ||
        px >= first_surfel_depth.width() || py >= first_surfel_depth.height()) {
      return;
    }
    
    RenderMinDepthAtPixel(
        px, py, local_position,
        first_surfel_depth);
    
    float x_frac = pixel_pos.x - px;
    float y_frac = pixel_pos.y - py;
    bool integrate = true;
    int offset_x;
    int offset_y;
    if (x_frac < y_frac) {
      // Surfel is within the bottom-left triangle half of the pixel.
      if (x_frac < 1 - y_frac) {
        // Surfel is on the left side of the pixel.
        if (px > 1) {
          offset_x = px - 1;
          offset_y = py;
        } else {
          integrate = false;
        }
      } else {
        // Surfel is on the bottom side of the pixel.
        if (py < first_surfel_depth.height() - 1) {
          offset_x = px;
          offset_y = py + 1;
        } else {
          integrate = false;
        }
      }
    } else {
      // Surfel is within the top-right triangle half of the pixel.
      if (x_frac < 1 - y_frac) {
        // Surfel is on the top side of the pixel.
        if (py > 0) {
          offset_x = px;
          offset_y = py - 1;
        } else {
          integrate = false;
        }
      } else {
        // Surfel is on the right side of the pixel.
        if (px < first_surfel_depth.width() - 1) {
          offset_x = px + 1;
          offset_y = py;
        } else {
          integrate = false;
        }
      }
    }
    
    if (integrate) {
      RenderMinDepthAtPixel(
          offset_x, offset_y, local_position,
          first_surfel_depth);
    }
  }
}

void CallRenderMinDepthCUDAKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    CUDABuffer_<float> first_surfel_depth) {
  RenderMinDepthCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      surfel_integration_active_window_size,
      fx, fy, cx, cy,
      local_T_global,
      surfel_count,
      surfels,
      first_surfel_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__device__ void ConsiderSurfelAssociationToPixel(
    int x, int y,
    const float3& cam_space_surfel_pos,
    unsigned int surfel_index,
    const CUDABuffer_<float>& surfels,
    const CUDAMatrix3x4& local_T_global,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    float depth_correction_factor,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<float2>& normals_buffer,
    const CUDABuffer_<float>& radius_buffer,
    CUDABuffer_<u32>& supporting_surfels,
    CUDABuffer_<u32>& supporting_surfel_counts,
    CUDABuffer_<float>& supporting_surfel_depth_sums,
    CUDABuffer_<u32>& conflicting_surfels,
    CUDABuffer_<float>& first_surfel_depth) {
  // Check whether the surfel falls on a depth pixel.
  float measurement_depth = depth_correction_factor * depth_buffer(y, x);
  if (measurement_depth <= 0) {
    return;
  }
  
  // Check if this or another surfel is conflicting.
  const float first_surfel_depth_value = first_surfel_depth(y, x);
  if (first_surfel_depth_value < (1 - sensor_noise_factor) * measurement_depth) {
    // This or another surfel is conflicting.
    if (first_surfel_depth_value == cam_space_surfel_pos.z) {
      // This surfel is conflicting.
      conflicting_surfels(y, x) = surfel_index;
    }
    return;
  }
  
  // Determine the depth from which on surfels are considered to be occluded.
  float occlusion_depth = (1 + sensor_noise_factor) * measurement_depth;
  if (kProtectSlightlyOccludedSurfels) {
    if (first_surfel_depth_value < occlusion_depth) {
      // TODO: Would it be better to use the front surfel's radius for that?
      occlusion_depth = (1 + kOcclusionDepthFactor) * first_surfel_depth_value;
    }
  }
  
  // Check if this surfel is occluded.
  if (cam_space_surfel_pos.z > occlusion_depth) {
    // Surfel is occluded.
    return;
  }
  
  // The measurement supports the surfel. Determine whether they belong to the
  // same surface (then the measurement should be integrated into the surfel),
  // or to different surfaces (then the measurement must not be integrated).
  
  // Check whether the surfel normal looks towards the camera (instead of away from it).
  float surfel_distance = Norm(cam_space_surfel_pos);
  float3 global_surfel_normal = make_float3(surfels(kSurfelNormalX, surfel_index),
                                            surfels(kSurfelNormalY, surfel_index),
                                            surfels(kSurfelNormalZ, surfel_index));
  float3 local_surfel_normal = local_T_global.Rotate(global_surfel_normal);
  
  float dot_angle = (1.0f / surfel_distance) * (cam_space_surfel_pos.x * local_surfel_normal.x +
                                                cam_space_surfel_pos.y * local_surfel_normal.y +
                                                cam_space_surfel_pos.z * local_surfel_normal.z);
  if (dot_angle > kSurfelNormalToViewingDirThreshold) {
    return;
  }
  
  // Check whether the surfel normal is compatible with the measurement normal.
  if (measurement_depth < cam_space_surfel_pos.z) {
    float2 normal = normals_buffer(y, x);
    float3 local_normal = make_float3(normal.x, normal.y, -sqrtf(::max(0.f, 1 - normal.x * normal.x - normal.y * normal.y)));
    
    float dot_angle = local_surfel_normal.x * local_normal.x +
                      local_surfel_normal.y * local_normal.y +
                      local_surfel_normal.z * local_normal.z;
    if (dot_angle < cos_normal_compatibility_threshold) {
      // HACK: Avoid creation of a new surfel here in case there is no other conflicting or supporting surfel
      //       by setting conflicting_surfels(y, x) to an invalid index unequal to kInvalidSurfelIndex.
      // TODO: This can be harmful since it can prevent the creation of valid surfaces. Delete it?
//       atomicCAS(&conflicting_surfels(y, x), kInvalidSurfelIndex, kInvalidSurfelIndex - 1);
      return;
    }
  }
  
  // The measurement seems to belong to the same surface as the surfel.
  
  // Check whether the observation scale is compatible with the surfel scale.
  const float surfel_radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
  if (surfel_radius_squared <= 0) {
    return;
  }
  if (kCheckScaleCompatibilityForIntegration) {
    const float observation_radius_squared = radius_buffer(y, x);
    if (observation_radius_squared / surfel_radius_squared > kMaxObservationRadiusFactorForIntegration * kMaxObservationRadiusFactorForIntegration) {
      // HACK: Avoid creation of a new surfel here in case there is no other conflicting or supporting surfel
      //       by setting conflicting_surfels(y, x) to an invalid index unequal to kInvalidSurfelIndex.
      atomicCAS(&conflicting_surfels(y, x), kInvalidSurfelIndex, kInvalidSurfelIndex - 1);
      return;
    }
  }
  
  // Replace the supporting surfel entry only if it was previously empty
  atomicCAS(&supporting_surfels(y, x), kInvalidSurfelIndex, surfel_index);
  
  // Add to supporting surfel count for the pixel
  atomicAdd(&supporting_surfel_counts(y, x), 1);
  
  // Add to the supporting surfel depth sum for the pixel
  atomicAdd(&supporting_surfel_depth_sums(y, x), cam_space_surfel_pos.z);
}

__global__ void AssociateSurfelsCUDAKernel(
    u32 frame_index,
    int surfel_integration_active_window_size,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float depth_correction_factor,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<float> supporting_surfel_depth_sums,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<float> first_surfel_depth) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    if (!IsSurfelActiveForIntegration(surfel_index, surfels, frame_index, surfel_integration_active_window_size)) {
      return;
    }
    
    float3 global_position =
        make_float3(surfels(kSurfelX, surfel_index),
                    surfels(kSurfelY, surfel_index),
                    surfels(kSurfelZ, surfel_index));
    float3 local_position = local_T_global * global_position;
    if (local_position.z <= 0) {
      // TODO: Compute z before x and y such that this early exit can be done earlier?
      return;
    }
    
    float2 pixel_pos =
        make_float2(fx * (local_position.x / local_position.z) + cx,
                    fy * (local_position.y / local_position.z) + cy);
    int px = static_cast<int>(pixel_pos.x);
    int py = static_cast<int>(pixel_pos.y);
    if (pixel_pos.x < 0 || pixel_pos.y < 0 ||
        px < 0 || py < 0 ||
        px >= depth_buffer.width() || py >= depth_buffer.height()) {
      return;
    }
    
    ConsiderSurfelAssociationToPixel(
        px, py, local_position,
        surfel_index, surfels,
        local_T_global,
        sensor_noise_factor,
        cos_normal_compatibility_threshold,
        depth_correction_factor, depth_buffer, normals_buffer, radius_buffer, supporting_surfels,
        supporting_surfel_counts, supporting_surfel_depth_sums, conflicting_surfels, first_surfel_depth);
    
    float x_frac = pixel_pos.x - px;
    float y_frac = pixel_pos.y - py;
    bool integrate = true;
    int offset_x;
    int offset_y;
    if (x_frac < y_frac) {
      // Surfel is within the bottom-left triangle half of the pixel.
      if (x_frac < 1 - y_frac) {
        // Surfel is on the left side of the pixel.
        if (px > 1) {
          offset_x = px - 1;
          offset_y = py;
        } else {
          integrate = false;
        }
      } else {
        // Surfel is on the bottom side of the pixel.
        if (py < depth_buffer.height() - 1) {
          offset_x = px;
          offset_y = py + 1;
        } else {
          integrate = false;
        }
      }
    } else {
      // Surfel is within the top-right triangle half of the pixel.
      if (x_frac < 1 - y_frac) {
        // Surfel is on the top side of the pixel.
        if (py > 0) {
          offset_x = px;
          offset_y = py - 1;
        } else {
          integrate = false;
        }
      } else {
        // Surfel is on the right side of the pixel.
        if (px < depth_buffer.width() - 1) {
          offset_x = px + 1;
          offset_y = py;
        } else {
          integrate = false;
        }
      }
    }
    
    if (integrate) {
      ConsiderSurfelAssociationToPixel(
          offset_x, offset_y, local_position,
          surfel_index, surfels,
          local_T_global,
          sensor_noise_factor,
          cos_normal_compatibility_threshold,
          depth_correction_factor, depth_buffer, normals_buffer, radius_buffer, supporting_surfels,
          supporting_surfel_counts, supporting_surfel_depth_sums, conflicting_surfels, first_surfel_depth);
    }
  }
}

void CallAssociateSurfelsCUDAKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    u32 frame_index,
    int surfel_integration_active_window_size,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float depth_correction_factor,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<float> supporting_surfel_depth_sums,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<float> first_surfel_depth) {
  AssociateSurfelsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      surfel_integration_active_window_size,
      fx, fy, cx, cy,
      local_T_global,
      sensor_noise_factor,
      cos_normal_compatibility_threshold,
      surfel_count,
      surfels,
      depth_correction_factor,
      depth_buffer,
      normals_buffer,
      radius_buffer,
      supporting_surfels,
      supporting_surfel_counts,
      supporting_surfel_depth_sums,
      conflicting_surfels,
      first_surfel_depth);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__device__ bool ConsiderSurfelMergeAtPixel(
    int x, int y,
    const float3& cam_space_surfel_pos,
    const float3& global_surfel_pos,
    unsigned int surfel_index,
    CUDABuffer_<float>& surfels,
    const CUDAMatrix3x4& local_T_global,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    float depth_correction_factor,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<float2>& normals_buffer,
    const CUDABuffer_<float>& radius_buffer,
    CUDABuffer_<u32>& supporting_surfels,
    CUDABuffer_<u32>& supporting_surfel_counts,
    CUDABuffer_<float>& supporting_surfel_depth_sums,
    CUDABuffer_<u32>& conflicting_surfels,
    CUDABuffer_<float>& first_surfel_depth) {
  // Check whether the surfel falls on a depth pixel.
  float measurement_depth = depth_correction_factor * depth_buffer(y, x);
  if (measurement_depth <= 0) {
    return false;
  }
  
  // Check if this or another surfel is conflicting.
  const float first_surfel_depth_value = first_surfel_depth(y, x);
  if (first_surfel_depth_value < (1 - sensor_noise_factor) * measurement_depth) {
    // This or another surfel is conflicting.
    if (first_surfel_depth_value == cam_space_surfel_pos.z) {
      // This surfel is conflicting.
      conflicting_surfels(y, x) = surfel_index;
    }
    return false;
  }
  
  // Determine the depth from which on surfels are considered to be occluded.
  float occlusion_depth = (1 + sensor_noise_factor) * measurement_depth;
  if (kProtectSlightlyOccludedSurfels) {
    if (first_surfel_depth_value < occlusion_depth) {
      // TODO: Would it be better to use the front surfel's radius for that?
      occlusion_depth = (1 + kOcclusionDepthFactor) * first_surfel_depth_value;
    }
  }
  
  // Check if this surfel is occluded.
  if (cam_space_surfel_pos.z > occlusion_depth) {
    // Surfel is occluded.
    return false;
  }
  
  // The measurement supports the surfel. Determine whether they belong to the
  // same surface (then the measurement should be integrated into the surfel),
  // or to different surfaces (then the measurement must not be integrated).
  
  // Check whether the surfel normal looks towards the camera (instead of away from it).
  float surfel_distance = Norm(cam_space_surfel_pos);
  float3 global_surfel_normal = make_float3(surfels(kSurfelNormalX, surfel_index),
                                            surfels(kSurfelNormalY, surfel_index),
                                            surfels(kSurfelNormalZ, surfel_index));
  float3 local_surfel_normal = local_T_global.Rotate(global_surfel_normal);
  
  float dot_angle = (1.0f / surfel_distance) * (cam_space_surfel_pos.x * local_surfel_normal.x +
                                                cam_space_surfel_pos.y * local_surfel_normal.y +
                                                cam_space_surfel_pos.z * local_surfel_normal.z);
  if (dot_angle > kSurfelNormalToViewingDirThreshold) {
    return false;
  }
  
  // Check whether the surfel normal is compatible with the measurement normal.
  if (measurement_depth < cam_space_surfel_pos.z) {
    float2 normal = normals_buffer(y, x);
    float3 local_normal = make_float3(normal.x, normal.y, -sqrtf(::max(0.f, 1 - normal.x * normal.x - normal.y * normal.y)));
    
    float dot_angle = local_surfel_normal.x * local_normal.x +
                      local_surfel_normal.y * local_normal.y +
                      local_surfel_normal.z * local_normal.z;
    if (dot_angle < cos_normal_compatibility_threshold) {
      return false;
    }
  }
  
  // The measurement seems to belong to the same surface as the surfel.
  
  // Check whether the observation scale is compatible with the surfel scale.
  const float surfel_radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
  if (kCheckScaleCompatibilityForIntegration) {
    const float observation_radius_squared = radius_buffer(y, x);
    if (observation_radius_squared / surfel_radius_squared > kMaxObservationRadiusFactorForIntegration * kMaxObservationRadiusFactorForIntegration) {
      return false;
    }
  }
  
  // Never merge the supported surfel.
  u32 supported_surfel = supporting_surfels(y, x);
  if (supported_surfel == surfel_index || supported_surfel == kInvalidSurfelIndex) {
    return false;
  }
  
  // Compare the surfel to the supported surfel. Merge only if very similar.
  // Radius:
  const float other_radius_squared = surfels(kSurfelRadiusSquared, supported_surfel);
  float radius_diff = surfel_radius_squared / other_radius_squared;
  constexpr float kRadiusDiffThreshold = 1.2f;
  constexpr float kRadiusDiffThresholdSq = kRadiusDiffThreshold * kRadiusDiffThreshold;
  if (radius_diff > kRadiusDiffThresholdSq || radius_diff < 1 / kRadiusDiffThresholdSq) {
    return false;
  }
  
  // Distance:
  float3 other_global_position =
      make_float3(surfels(kSurfelX, supported_surfel),
                  surfels(kSurfelY, supported_surfel),
                  surfels(kSurfelZ, supported_surfel));
  float distance_squared = SquaredDistance(global_surfel_pos, other_global_position);
  constexpr float kDistanceThresholdFactor = 0.5f * (0.25f * 0.25f);
  if (distance_squared > kDistanceThresholdFactor * (surfel_radius_squared + other_radius_squared)) {
    return false;
  }
  
  // Normal:
  float3 other_surfel_normal = make_float3(surfels(kSurfelNormalX, supported_surfel),
                                           surfels(kSurfelNormalY, supported_surfel),
                                           surfels(kSurfelNormalZ, supported_surfel));
  dot_angle = Dot(global_surfel_normal, other_surfel_normal);
  constexpr float kCosNormalMergeThreshold = 0.93969f;  // 20 degrees
  if (dot_angle < kCosNormalMergeThreshold) {
    return false;
  }
  
  // Merge the surfel.
  *reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index)) = 0;
  surfels(kSurfelRadiusSquared, surfel_index) = -1;
  *reinterpret_cast<u8*>(&reinterpret_cast<uchar4*>(&surfels(kSurfelColor, surfel_index))->w) = 1;  // Set neighbor detach request flag
  
  return true;
}

__global__ void MergeSurfelsCUDAKernel(
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float depth_correction_factor,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<float> supporting_surfel_depth_sums,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<float> first_surfel_depth,
    CUDABuffer_<u32> num_merges_buffer) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  bool merged = false;
  
  if (surfel_index < surfel_count) {
//     if (IsSurfelActiveForIntegration(surfel_index, surfels, frame_index, surfel_integration_active_window_size)) {
    if (surfels(kSurfelRadiusSquared, surfel_index) >= 0) {
      float3 global_position =
          make_float3(surfels(kSurfelX, surfel_index),
                      surfels(kSurfelY, surfel_index),
                      surfels(kSurfelZ, surfel_index));
      float3 local_position = local_T_global * global_position;
      if (local_position.z > 0) {  // TODO: Compute z before x and y such that this early exit can be done earlier?
        float2 pixel_pos =
            make_float2(fx * (local_position.x / local_position.z) + cx,
                        fy * (local_position.y / local_position.z) + cy);
        int px = static_cast<int>(pixel_pos.x);
        int py = static_cast<int>(pixel_pos.y);
        if (!(pixel_pos.x < 0 || pixel_pos.y < 0 ||
            px < 0 || py < 0 ||
            px >= depth_buffer.width() || py >= depth_buffer.height())) {
          merged = ConsiderSurfelMergeAtPixel(
              px, py, local_position, global_position,
              surfel_index, surfels,
              local_T_global,
              sensor_noise_factor,
              cos_normal_compatibility_threshold,
              depth_correction_factor, depth_buffer, normals_buffer, radius_buffer, supporting_surfels,
              supporting_surfel_counts, supporting_surfel_depth_sums, conflicting_surfels, first_surfel_depth);
        }
      }
    }
  }
  
  typedef typename cub::BlockReduce<int, kMergeBlockWidth, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceInt;
  __shared__ typename BlockReduceInt::TempStorage temp_storage;
  
  int num_merges = BlockReduceInt(temp_storage).Sum(merged ? 1 : 0);
  if (threadIdx.x == 0 && num_merges > 0) {
    atomicAdd(&num_merges_buffer(0, 0), static_cast<u32>(num_merges));
  }
}

void CallMergeSurfelsCUDAKernel(
    cudaStream_t stream,
    const dim3& grid_dim,
    const dim3& block_dim,
    float fx, float fy, float cx, float cy,
    CUDAMatrix3x4 local_T_global,
    float sensor_noise_factor,
    float cos_normal_compatibility_threshold,
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    float depth_correction_factor,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float2> normals_buffer,
    CUDABuffer_<float> radius_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u32> supporting_surfel_counts,
    CUDABuffer_<float> supporting_surfel_depth_sums,
    CUDABuffer_<u32> conflicting_surfels,
    CUDABuffer_<float> first_surfel_depth,
    CUDABuffer_<u32> num_merges_buffer) {
  MergeSurfelsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      fx, fy, cx, cy,
      local_T_global,
      sensor_noise_factor,
      cos_normal_compatibility_threshold,
      surfel_count,
      surfels,
      depth_correction_factor,
      depth_buffer,
      normals_buffer,
      radius_buffer,
      supporting_surfels,
      supporting_surfel_counts,
      supporting_surfel_depth_sums,
      conflicting_surfels,
      first_surfel_depth,
      num_merges_buffer);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}


__global__ void RegularizeSurfelsCUDAClearGradientsKernel(
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    // TODO: Put this in the last kernel of the denoising (and in the
    //       initialization) and expect that it remains zero in-between the
    //       calls in order to save one kernel call? Is this used anywhere else?
    surfels(kSurfelGradientX, surfel_index) = 0;
    surfels(kSurfelGradientY, surfel_index) = 0;
    surfels(kSurfelGradientZ, surfel_index) = 0;
    surfels(kSurfelGradientCount, surfel_index) = 0;
  }
}

__global__ void RegularizeSurfelsCUDAAccumulateNeighborGradientsKernel(
    u32 frame_index,
    int regularization_frame_window_size,
    float radius_factor_for_regularization_neighbors_squared,
    float regularizer_weight,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    // Count neighbors.
    int neighbor_count = 0;
    for (int neighbor_index = 0; neighbor_index < kSurfelNeighborCount; ++ neighbor_index) {
      u32 neighbor_surfel_index = *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + neighbor_index, surfel_index));
      if (neighbor_surfel_index == kInvalidSurfelIndex) {
        continue;
      }
      if (static_cast<int>(*reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, neighbor_surfel_index))) < static_cast<int>(frame_index - regularization_frame_window_size)) {
        continue;
      }
      ++ neighbor_count;
    }
    if (neighbor_count == 0) {
      return;
    }
    
    float3 smooth_position =
        make_float3(surfels(kSurfelSmoothX, surfel_index),
                    surfels(kSurfelSmoothY, surfel_index),
                    surfels(kSurfelSmoothZ, surfel_index));
    float3 normal =
        make_float3(surfels(kSurfelNormalX, surfel_index),
                    surfels(kSurfelNormalY, surfel_index),
                    surfels(kSurfelNormalZ, surfel_index));
    
    const float surfel_radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
    
    // Accumulate gradient terms for neighbors.
    float factor = 2 * regularizer_weight / neighbor_count;
    for (int neighbor_index = 0; neighbor_index < kSurfelNeighborCount; ++ neighbor_index) {
      u32 neighbor_surfel_index = *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + neighbor_index, surfel_index));
      if (neighbor_surfel_index == kInvalidSurfelIndex) {
        continue;
      }
      if (static_cast<int>(*reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, neighbor_surfel_index))) < static_cast<int>(frame_index - regularization_frame_window_size)) {
        continue;
      }
      
      float3 neighbor_position =
          make_float3(surfels(kSurfelSmoothX, neighbor_surfel_index),
                      surfels(kSurfelSmoothY, neighbor_surfel_index),
                      surfels(kSurfelSmoothZ, neighbor_surfel_index));
      
      float3 this_to_neighbor =
          make_float3(neighbor_position.x - smooth_position.x,
                      neighbor_position.y - smooth_position.y,
                      neighbor_position.z - smooth_position.z);
      float factor_times_normal_dot_difference = factor * (normal.x * this_to_neighbor.x + normal.y * this_to_neighbor.y + normal.z * this_to_neighbor.z);
      
      float3 gradient_term_for_neighbor =
          make_float3(factor_times_normal_dot_difference * normal.x,
                      factor_times_normal_dot_difference * normal.y,
                      factor_times_normal_dot_difference * normal.z);
      
      atomicAdd(&surfels(kSurfelGradientX, neighbor_surfel_index), gradient_term_for_neighbor.x);
      atomicAdd(&surfels(kSurfelGradientY, neighbor_surfel_index), gradient_term_for_neighbor.y);
      atomicAdd(&surfels(kSurfelGradientZ, neighbor_surfel_index), gradient_term_for_neighbor.z);
      atomicAdd(&surfels(kSurfelGradientCount, neighbor_surfel_index), regularizer_weight / neighbor_count);
      
      // If the neighbor is too far away, remove it.
      // NOTE / TODO: it can still happen that there are far away but inactive
      //              neighbors, which will influence an active surfel, since
      //              this check only removes active neighbors.
      //              However, I think this should be relatively rare.
      float neighbor_distance_squared = this_to_neighbor.x * this_to_neighbor.x + this_to_neighbor.y * this_to_neighbor.y + this_to_neighbor.z * this_to_neighbor.z;
      if (neighbor_distance_squared > radius_factor_for_regularization_neighbors_squared * surfel_radius_squared) {
        *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + neighbor_index, surfel_index)) = kInvalidSurfelIndex;
      }
    }
  }
}

__global__ void RegularizeSurfelsCUDAKernel(
    u32 frame_index,
    int regularization_frame_window_size,
    float regularizer_weight,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    if (static_cast<int>(*reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index))) < static_cast<int>(frame_index - regularization_frame_window_size)) {
      return;
    }
    
    float3 measured_position =
        make_float3(surfels(kSurfelX, surfel_index),
                    surfels(kSurfelY, surfel_index),
                    surfels(kSurfelZ, surfel_index));
    float3 smooth_position =
        make_float3(surfels(kSurfelSmoothX, surfel_index),
                    surfels(kSurfelSmoothY, surfel_index),
                    surfels(kSurfelSmoothZ, surfel_index));
    float3 normal =
        make_float3(surfels(kSurfelNormalX, surfel_index),
                    surfels(kSurfelNormalY, surfel_index),
                    surfels(kSurfelNormalZ, surfel_index));
    
    // Data term and neighbor-induced gradient terms
    constexpr float data_term_factor = 2;
    float3 gradient =
        make_float3(data_term_factor * (smooth_position.x - measured_position.x) + surfels(kSurfelGradientX, surfel_index),
                    data_term_factor * (smooth_position.y - measured_position.y) + surfels(kSurfelGradientY, surfel_index),
                    data_term_factor * (smooth_position.z - measured_position.z) + surfels(kSurfelGradientZ, surfel_index));
    
    // Regularization gradient terms
    int neighbor_count = 0;
    float3 regularization_gradient = make_float3(0, 0, 0);
    for (int neighbor_index = 0; neighbor_index < kSurfelNeighborCount; ++ neighbor_index) {
      u32 neighbor_surfel_index = *reinterpret_cast<u32*>(&surfels(kSurfelNeighbor0 + neighbor_index, surfel_index));
      if (neighbor_surfel_index == kInvalidSurfelIndex) {
        continue;
      }
      
      ++ neighbor_count;
      
      float3 neighbor_position =
          make_float3(surfels(kSurfelSmoothX, neighbor_surfel_index),
                      surfels(kSurfelSmoothY, neighbor_surfel_index),
                      surfels(kSurfelSmoothZ, neighbor_surfel_index));
      
      float3 this_to_neighbor =
          make_float3(neighbor_position.x - smooth_position.x,
                      neighbor_position.y - smooth_position.y,
                      neighbor_position.z - smooth_position.z);
      float normal_dot_difference = normal.x * this_to_neighbor.x + normal.y * this_to_neighbor.y + normal.z * this_to_neighbor.z;
      
      regularization_gradient =
          make_float3(regularization_gradient.x - normal_dot_difference * normal.x,
                      regularization_gradient.y - normal_dot_difference * normal.y,
                      regularization_gradient.z - normal_dot_difference * normal.z);
    }
    
    if (neighbor_count > 0) {
      // Apply constant factor to regularization gradient term
      float factor = 2 * regularizer_weight / neighbor_count;
      gradient =
          make_float3(gradient.x + factor * regularization_gradient.x,
                      gradient.y + factor * regularization_gradient.y,
                      gradient.z + factor * regularization_gradient.z);
    }
    
    const float residual_terms_weight_sum = 1 + regularizer_weight + surfels(kSurfelGradientCount, surfel_index);
    const float kStepSizeFactor = 0.5f / residual_terms_weight_sum;
    
    // Avoid divergence by limiting the step length to a multiple of the surfel
    // radius (multiple with this factor here).
    // TODO: It seems that this is not necessary anymore now that the step size
    //       is more intelligently chosen. Remove it (after some more extensive
    //       testing).
    constexpr float kMaxStepLengthFactor = 1.0f;
    float max_step_length = kMaxStepLengthFactor * sqrtf(surfels(kSurfelRadiusSquared, surfel_index));
    float step_length = kStepSizeFactor * sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z);
    float step_factor = kStepSizeFactor;
    if (step_length > max_step_length) {
      step_factor = max_step_length / step_length * kStepSizeFactor;
    }
    
    // NOTE: Writing the update into the gradient first to avoid race conditions
    //       (the smooth position may still be used by neighboring surfel updates).
    //       The next kernel call will move the result to the smooth position field.
    surfels(kSurfelGradientX, surfel_index) = smooth_position.x - step_factor * gradient.x;
    surfels(kSurfelGradientY, surfel_index) = smooth_position.y - step_factor * gradient.y;
    surfels(kSurfelGradientZ, surfel_index) = smooth_position.z - step_factor * gradient.z;
  }
}

__global__ void RegularizeSurfelsCUDAUpdateKernel(
    u32 frame_index,
    int regularization_frame_window_size,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    if (static_cast<int>(*reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index))) < static_cast<int>(frame_index - regularization_frame_window_size)) {
      return;
    }
    
    surfels(kSurfelSmoothX, surfel_index) = surfels(kSurfelGradientX, surfel_index);
    surfels(kSurfelSmoothY, surfel_index) = surfels(kSurfelGradientY, surfel_index);
    surfels(kSurfelSmoothZ, surfel_index) = surfels(kSurfelGradientZ, surfel_index);
  }
}

__global__ void RegularizeSurfelsCUDACopyOnlyKernel(
    u32 frame_index,
    int regularization_frame_window_size,
    u32 surfel_count,
    CUDABuffer_<float> surfels) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    // TODO: Only changed surfels need to be touched here.
    if (static_cast<int>(*reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index))) < static_cast<int>(frame_index - regularization_frame_window_size)) {
      return;
    }
    
    surfels(kSurfelSmoothX, surfel_index) = surfels(kSurfelX, surfel_index);
    surfels(kSurfelSmoothY, surfel_index) = surfels(kSurfelY, surfel_index);
    surfels(kSurfelSmoothZ, surfel_index) = surfels(kSurfelZ, surfel_index);
  }
}

void RegularizeSurfelsCUDA(
    cudaStream_t stream,
    bool disable_denoising,
    u32 frame_index,
    float radius_factor_for_regularization_neighbors,
    float regularizer_weight,
    int regularization_frame_window_size,
    u32 surfel_count,
    CUDABuffer_<float>* surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  if (disable_denoising) {
    // Only copy the raw surfel positions to the smoothed position fields.
    RegularizeSurfelsCUDACopyOnlyKernel
    <<<grid_dim, block_dim, 0, stream>>>(
        frame_index,
        regularization_frame_window_size,
        surfel_count,
        *surfels);
    #ifdef CUDA_SEQUENTIAL_CHECKS
      cudaDeviceSynchronize();
    #endif
    CHECK_CUDA_NO_ERROR();
    return;
  }
  
  RegularizeSurfelsCUDAClearGradientsKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      surfel_count,
      *surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  RegularizeSurfelsCUDAAccumulateNeighborGradientsKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      regularization_frame_window_size,
      radius_factor_for_regularization_neighbors * radius_factor_for_regularization_neighbors,
      regularizer_weight,
      surfel_count,
      *surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  RegularizeSurfelsCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      regularization_frame_window_size,
      regularizer_weight,
      surfel_count,
      *surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  RegularizeSurfelsCUDAUpdateKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      frame_index,
      regularization_frame_window_size,
      surfel_count,
      *surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

__global__ void ExportVerticesCUDAKernel(
    u32 surfel_count,
    CUDABuffer_<float> surfels,
    CUDABuffer_<float> position_buffer,
    CUDABuffer_<u8> color_buffer) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfel_count) {
    bool merged = surfels(kSurfelRadiusSquared, surfel_index) < 0;
    
    float* position_ptr = position_buffer.address();
    position_ptr[3 * surfel_index + 0] = merged ? CUDART_NAN_F : surfels(kSurfelSmoothX, surfel_index);
    position_ptr[3 * surfel_index + 1] = merged ? CUDART_NAN_F : surfels(kSurfelSmoothY, surfel_index);
    position_ptr[3 * surfel_index + 2] = merged ? CUDART_NAN_F : surfels(kSurfelSmoothZ, surfel_index);
    
    const uchar4 color = *(reinterpret_cast<uchar4*>(&surfels(kSurfelColor, surfel_index)));
    u8* color_ptr = color_buffer.address();
    color_ptr[3 * surfel_index + 0] = color.x;
    color_ptr[3 * surfel_index + 1] = color.y;
    color_ptr[3 * surfel_index + 2] = color.z;
  }
}

void ExportVerticesCUDA(
    cudaStream_t stream,
    u32 surfel_count,
    const CUDABuffer_<float>& surfels,
    CUDABuffer_<float>* position_buffer,
    CUDABuffer_<u8>* color_buffer) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  if (surfel_count == 0) {
    return;
  }
  
  constexpr int kBlockWidth = 1024;
  dim3 grid_dim(GetBlockCount(surfel_count, kBlockWidth));
  dim3 block_dim(kBlockWidth);
  
  ExportVerticesCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      surfel_count,
      surfels,
      *position_buffer,
      *color_buffer);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

__global__ void DebugPrintSurfelCUDAKernel(
    usize surfel_index,
    CUDABuffer_<float> surfels) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("DEBUGGING surfel %i on GPU ...\n", static_cast<int>(surfel_index));
    
    printf("DEBUG surfel raw position x on GPU: %f\n", surfels(kSurfelX, surfel_index));
    printf("DEBUG surfel raw position y on GPU: %f\n", surfels(kSurfelY, surfel_index));
    printf("DEBUG surfel raw position z on GPU: %f\n", surfels(kSurfelZ, surfel_index));
    
    printf("DEBUG surfel smooth position x on GPU: %f\n", surfels(kSurfelSmoothX, surfel_index));
    printf("DEBUG surfel smooth position y on GPU: %f\n", surfels(kSurfelSmoothY, surfel_index));
    printf("DEBUG surfel smooth position z on GPU: %f\n", surfels(kSurfelSmoothZ, surfel_index));
    
    printf("DEBUG surfel creation stamp on GPU: %i\n", static_cast<int>(*reinterpret_cast<u32*>(&surfels(kSurfelCreationStamp, surfel_index))));
  }
}

void DebugPrintSurfelCUDA(
    cudaStream_t stream,
    usize surfel_index,
    const CUDABuffer_<float>& surfels) {
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
  
  constexpr int kBlockWidth = 32;
  dim3 grid_dim(1);
  dim3 block_dim(kBlockWidth);
  DebugPrintSurfelCUDAKernel
  <<<grid_dim, block_dim, 0, stream>>>(
      surfel_index,
      surfels);
  #ifdef CUDA_SEQUENTIAL_CHECKS
    cudaDeviceSynchronize();
  #endif
  CHECK_CUDA_NO_ERROR();
}

void CallCUBExclusiveSum(
    void* temp_storage,
    usize& temp_storage_bytes,
    u8* d_in,
    u32* d_out,
    int num_items,
    cudaStream_t stream) {
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      stream);
}

}
