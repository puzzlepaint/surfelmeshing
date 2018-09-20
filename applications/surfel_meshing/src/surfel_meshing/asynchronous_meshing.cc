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

#include "surfel_meshing/asynchronous_meshing.h"

#include <libvis/timing.h>

#include "surfel_meshing/cuda_surfels_cpu.h"
#include "surfel_meshing/surfel_meshing_render_window.h"
#include "surfel_meshing/surfel_meshing.h"

namespace vis {

AsynchronousMeshing::AsynchronousMeshing(
    SurfelMeshing* surfel_meshing,
    CUDASurfelsCPU* cuda_surfels_cpu_buffers,
    bool log_timings,
    const shared_ptr<SurfelMeshingRenderWindow>& render_window)
    : surfel_meshing_(surfel_meshing),
      cuda_surfels_cpu_buffers_(cuda_surfels_cpu_buffers),
      log_timings_(log_timings),
      render_window_(render_window) {
  output_mesh_ = nullptr;
  all_work_done_ = false;
  
  triangulation_thread_exit_requested_ = false;
  new_input_surfels_available_ = false;
  triangulation_thread_.reset(new thread(bind(&AsynchronousMeshing::ThreadMain, this)));
}

void AsynchronousMeshing::ThreadMain() {
  while (true) {
    // Wait until there is a change to the surfels.
    // ### Input data lock start ###
    unique_lock<mutex> input_data_lock(input_data_mutex_);
    while (!new_input_surfels_available_ && !triangulation_thread_exit_requested_) {
      all_work_done_ = true;
      // all_work_done_condition_.notify_all();
      new_input_surfels_available_condition_.wait(input_data_lock);
    }
    
    // Exit if requested.
    if (triangulation_thread_exit_requested_) {
      return;
    }
    
    // Swap pointers in the shared buffer to get exclusive access to the latest
    // input.
    cuda_surfels_cpu_buffers_->WaitForLockAndSwapBuffers();
    new_input_surfels_available_ = false;
    input_data_lock.unlock();
    // ### Input data lock end ###
    
    
    ConditionalTimer sync_timer_1("Sync1");
    
    start_time_mutex_.lock();
    start_time_ = chrono::steady_clock::now();
    start_time_mutex_.unlock();
    
    // Convert buffers to CPU surfels.
    surfel_meshing_->IntegrateCUDABuffers(
        cuda_surfels_cpu_buffers_->read_buffers().frame_index,
        *cuda_surfels_cpu_buffers_);
    
    float sync_seconds_1 = sync_timer_1.Stop(false);
    
    
    // Check remeshing.
    ConditionalTimer check_remeshing_timer("CheckRemeshing()");
    surfel_meshing_->CheckRemeshing();
    float remeshing_seconds = check_remeshing_timer.Stop();
    
    // Triangulate.
    ConditionalTimer triangulate_timer("Triangulate()");
    surfel_meshing_->Triangulate();
    float meshing_seconds = triangulate_timer.Stop();
    
    
    // Output
    ConditionalTimer sync_timer_2("Sync2");
    
    shared_ptr<Mesh3fCu8> new_output_mesh(new Mesh3fCu8());
    surfel_meshing_->ConvertToMesh3fCu8(new_output_mesh.get(), true);
    
    unique_lock<mutex> output_lock(output_mutex_);
    output_made_for_surfel_frame_index_ = cuda_surfels_cpu_buffers_->read_buffers().frame_index;
    output_made_with_surfel_count_ = cuda_surfels_cpu_buffers_->read_buffers().surfel_count;
    output_mesh_ = new_output_mesh;
    output_lock.unlock();
    
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
    start_time_mutex_.lock();
    latest_triangulation_duration_ = 1e-9 * chrono::duration<double, nano>(end_time - start_time_).count();
    start_time_mutex_.unlock();
    
    float sync_seconds_2 = sync_timer_2.Stop(false);
    
    if (log_timings_) {
      timings_log_ << "frame " << cuda_surfels_cpu_buffers_->read_buffers().frame_index << endl;
      timings_log_ << "-remeshing " << (1000 * remeshing_seconds) << endl;
      timings_log_ << "-meshing " << (1000 * meshing_seconds) << endl;
      timings_log_ << "-synchronization " << (1000 * (sync_seconds_1 + sync_seconds_2)) << endl;
      timings_log_ << "-triangle_count " << surfel_meshing_->triangle_count() << endl;
      timings_log_ << "-deleted_triangle_count " << surfel_meshing_->deleted_triangle_count() << endl;
    }
  }
}

void AsynchronousMeshing::NotifyAboutNewInputSurfels() {
  unique_lock<mutex> input_data_lock(input_data_mutex_);
  new_input_surfels_available_ = true;
  all_work_done_ = false;
  input_data_lock.unlock();
  new_input_surfels_available_condition_.notify_all();
}

void AsynchronousMeshing::NotifyAboutNewInputSurfelsAlreadyLocked() {
  new_input_surfels_available_ = true;
  all_work_done_ = false;
  new_input_surfels_available_condition_.notify_all();
}

void AsynchronousMeshing::RequestExitAndWaitForIt() {
  unique_lock<mutex> input_data_lock(input_data_mutex_);
  triangulation_thread_exit_requested_ = true;
  input_data_lock.unlock();
  new_input_surfels_available_condition_.notify_all();
  
  triangulation_thread_->join();
  
  if (log_timings_) {
    FILE* file = fopen("timings_cpu.txt", "wb");  // TODO: get path from program arguments
    string str = timings_log_.str();
    fwrite(str.c_str(), 1, str.size(), file);
    fclose(file);
  }
}

void AsynchronousMeshing::LockInputData() {
  input_data_mutex_.lock();
}

void AsynchronousMeshing::UnlockInputData() {
  input_data_mutex_.unlock();
}

void AsynchronousMeshing::GetOutput(
    u32* output_frame_index,
    u32* output_surfel_count,
    shared_ptr<Mesh3fCu8>* output_mesh) {
  unique_lock<mutex> output_lock(output_mutex_);
  
  *output_frame_index = output_made_for_surfel_frame_index_;
  *output_surfel_count = output_made_with_surfel_count_;
  *output_mesh = output_mesh_;
  
  output_mesh_ = nullptr;
}

}
