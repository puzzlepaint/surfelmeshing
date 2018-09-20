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

#include <libvis/libvis.h>

namespace vis {

__forceinline__ __device__ float SquaredLength(const float3& vec) {
  return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

__forceinline__ __device__ float Dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float SquaredDistance(const float3& a, const float3& b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

// Replacement for std::swap on device.
template<typename T>
__forceinline__ __device__ void Swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

__forceinline__ __device__ void UnprojectPoint(
    int x, int y, float depth,
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    float3* result) {
  *result =
      make_float3(depth * (fx_inv * x + cx_inv),
                  depth * (fy_inv * y + cy_inv),
                  depth);
}

__forceinline__ __device__ void CrossProduct(const float3& a, const float3& b, float3* result) {
  *result = make_float3(a.y * b.z - b.y * a.z,
                        b.x * a.z - a.x * b.z,
                        a.x * b.y - b.x * a.y);
}

__forceinline__ __device__ float Norm(const float3& vec) {
  return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

}
