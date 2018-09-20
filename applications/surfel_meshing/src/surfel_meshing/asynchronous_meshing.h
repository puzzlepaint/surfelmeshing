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

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include <libvis/libvis.h>
#include <libvis/mesh.h>

namespace vis {

class CUDASurfelsCPU;
class SurfelMeshing;
class SurfelMeshingRenderWindow;

// Manages the surfel meshing thread.
class AsynchronousMeshing {
 public:
  // Starts the meshing thread.
  AsynchronousMeshing(
      SurfelMeshing* surfel_meshing,
      CUDASurfelsCPU* cuda_surfels_cpu_buffers,
      bool log_timings,
      const shared_ptr<SurfelMeshingRenderWindow>& render_window);
  
  // Notifies the thread about new input.
  void NotifyAboutNewInputSurfels();
  
  // Notifies the thread about new input (while the mutex is locked using
  // LockInputData()).
  void NotifyAboutNewInputSurfelsAlreadyLocked();
  
  // Requests the thread to exit and waits until it actually exits. It will
  // still finish the last iteration it started when this is called.
  void RequestExitAndWaitForIt();
  
  // Locks the input_data_mutex_.
  void LockInputData();
  
  // Unlocks the input_data_mutex_.
  void UnlockInputData();
  
  // Gets the output mesh (and frame index, surfel count). If no new output is
  // available, the retured pointer is null.
  void GetOutput(
      u32* output_frame_index,
      u32* output_surfel_count,
      shared_ptr<Mesh3fCu8>* output_mesh);
  
  // Returns the duration of the latest meshing iteration.
  inline float latest_triangulation_duration() const {
    return latest_triangulation_duration_;
  }
  
  // Returns the time_point at which the latest meshing iteration started.
  inline const chrono::steady_clock::time_point& latest_triangulation_start_time() const {
    unique_lock<mutex> lock(start_time_mutex_);
    return start_time_;
  }
  
  // Returns whether all work is done, i.e., the thread does not currently run
  // a meshing iteration and there is also no new input.
  inline bool all_work_done() const {
    return all_work_done_;
  }
  
 private:
  // Main function for the meshing thread.
  void ThreadMain();
  
  
  SurfelMeshing* surfel_meshing_;
  CUDASurfelsCPU* cuda_surfels_cpu_buffers_;
  bool log_timings_;
  ostringstream timings_log_;
  shared_ptr<SurfelMeshingRenderWindow> render_window_;
  
  mutex output_mutex_;
  // The mesh was made while considering the integrated surfels at this frame index:
  u32 output_made_for_surfel_frame_index_;
  // The mesh was made while this number of surfels was there:
  u32 output_made_with_surfel_count_;
  // The triangulated mesh (only the indices are valid, the vertices are given
  // by the surfels).
  shared_ptr<Mesh3fCu8> output_mesh_;
  
  mutex input_data_mutex_;
  condition_variable new_input_surfels_available_condition_;
  atomic<bool> new_input_surfels_available_;
  
  mutable mutex start_time_mutex_;
  chrono::steady_clock::time_point start_time_;
  atomic<float> latest_triangulation_duration_;
  
  atomic<bool> all_work_done_;
  
  atomic<bool> triangulation_thread_exit_requested_;
  unique_ptr<thread> triangulation_thread_;
};

}
