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

#include <mutex>

#include <libvis/eigen.h>
#include <libvis/libvis.h>

namespace vis {

// Contains CPU buffers for surfel attributes.
struct CUDASurfelBuffersCPU {
  CUDASurfelBuffersCPU(usize max_surfel_count) {
    surfel_x_buffer = new float[max_surfel_count];
    surfel_y_buffer = new float[max_surfel_count];
    surfel_z_buffer = new float[max_surfel_count];
    surfel_radius_squared_buffer = new float[max_surfel_count];
    surfel_normal_x_buffer = new float[max_surfel_count];
    surfel_normal_y_buffer = new float[max_surfel_count];
    surfel_normal_z_buffer = new float[max_surfel_count];
    surfel_last_update_stamp_buffer = new u32[max_surfel_count];
  }
  
  ~CUDASurfelBuffersCPU() {
    delete[] surfel_x_buffer;
    delete[] surfel_y_buffer;
    delete[] surfel_z_buffer;
    delete[] surfel_radius_squared_buffer;
    delete[] surfel_normal_x_buffer;
    delete[] surfel_normal_y_buffer;
    delete[] surfel_normal_z_buffer;
    delete[] surfel_last_update_stamp_buffer;
  }
  
  u32 frame_index;
  usize surfel_count;
  float* surfel_x_buffer;
  float* surfel_y_buffer;
  float* surfel_z_buffer;
  float* surfel_radius_squared_buffer;
  float* surfel_normal_x_buffer;
  float* surfel_normal_y_buffer;
  float* surfel_normal_z_buffer;
  u32* surfel_last_update_stamp_buffer;
};


// Holds two sets of buffers:
// - The write buffers are locked by the CUDA integration thread while
//   transferring data into them.
// - The triangulation thread locks the write buffers and exchanges the pointers
//   to then get lock-free access to the read buffers afterwards.
// 
// NOTE: For synchronous meshing mode, memory could be saved by allocating one set of buffers only.
class CUDASurfelsCPU {
 friend class CUDASurfelReconstruction;
 friend class SurfelMeshing;
 public:
  CUDASurfelsCPU(usize max_surfel_count)
      : write_buffers_(new CUDASurfelBuffersCPU(max_surfel_count)),
        read_buffers_(new CUDASurfelBuffersCPU(max_surfel_count)) {
    debug_wrote_data_ = false;
  }
  
  ~CUDASurfelsCPU() {
    delete write_buffers_;
    delete read_buffers_;
  }
  
  void LockWriteBuffers() {
    write_buffers_lock_.lock();
  }
  
  void UnlockWriteBuffers() {
    debug_wrote_data_ = true;
    write_buffers_lock_.unlock();
  }
  
  void WaitForLockAndSwapBuffers() {
    unique_lock<mutex> lock(write_buffers_lock_);
    if (!debug_wrote_data_) {
      LOG(FATAL) << "Trying to swap the CUDASurfelsCPU buffers, but no data was written. Possible multi-threading bug!";
    }
    std::swap(write_buffers_, read_buffers_);
    debug_wrote_data_ = false;
  }
  
  CUDASurfelBuffersCPU* write_buffers() { return write_buffers_; }
  const CUDASurfelBuffersCPU& read_buffers() const { return *read_buffers_; }
  
 private:
  bool debug_wrote_data_;
  std::mutex write_buffers_lock_;
  CUDASurfelBuffersCPU* write_buffers_;
  CUDASurfelBuffersCPU* read_buffers_;
};

}
