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
#include <libvis/sophus.h>

#include <libvis/cuda/cuda_buffer.h>

#include "surfel_meshing/cuda_matrix.cuh"

namespace vis {

// Performs:
// - Removal of depth values close to the corners of the image (where the Kinect seems to have too much bias).
// - Depth cutoff at max_depth.
// - Bilateral filtering with depth-dependent sigma_value.
void BilateralFilteringAndDepthCutoffCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value_factor,
    u16 value_to_ignore,
    float radius_factor,
    u16 max_depth,
    float depth_valid_region_radius,
    const CUDABuffer<u16>& input_depth,
    CUDABuffer<u16>* output_depth);

// Requires to observe a depth value in all other frames to accept it as inlier.
template <int count, typename DepthT>
void OutlierDepthMapFusionCUDA(
    cudaStream_t stream,
    float tolerance,
    const CUDABuffer<DepthT>& input_depth,
    const PinholeCamera4f& depth_camera,
    const CUDABuffer<DepthT>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer<u16>* output_depth);

// Variant of OutlierDepthMapFusionCUDA() which does not require to observe a depth value in all other frames.
template <int count, typename DepthT>
void OutlierDepthMapFusionCUDA(
    cudaStream_t stream,
    int required_count,
    float tolerance,
    const CUDABuffer<DepthT>& input_depth,
    const PinholeCamera4f& depth_camera,
    const CUDABuffer<DepthT>** other_depths,
    const CUDAMatrix3x4* others_TR_reference,
    CUDABuffer<u16>* output_depth);

template <typename DepthT>
void ErodeDepthMapCUDA(
    cudaStream_t stream,
    int radius,
    const CUDABuffer<DepthT>& input_depth,
    CUDABuffer<DepthT>* output_depth);

template <typename DepthT>
void CopyWithoutBorderCUDA(
    cudaStream_t stream,
    const CUDABuffer<DepthT>& input_depth,
    CUDABuffer<DepthT>* output_depth);

// This essentially erodes the depth map by another 1px.
// Assumes that no valid depth values exist at the border of the image (or will
// cause invalid memory accesses).
void ComputeNormalsAndDropBadPixelsCUDA(
    cudaStream_t stream,
    float observation_angle_threshold_deg,
    float depth_scaling,
    const PinholeCamera4f& depth_camera,
    const CUDABuffer<u16>& in_depth,
    CUDABuffer<u16>* out_depth,
    CUDABuffer<float2>* out_normals);  // TODO: Compress to char2?

void ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
    cudaStream_t stream,
    float point_radius_extension_factor,
    float point_radius_clamp_factor,
    float depth_scaling,
    const PinholeCamera4f& depth_camera,
    const CUDABuffer<u16>& depth_buffer,
    CUDABuffer<float>* radius_buffer,
    CUDABuffer<u16>* out_depth);

}
