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


#include <gtest/gtest.h>
#include <libvis/logging.h>

#include "surfel_meshing/surfel_meshing_render_window.h"
#include "surfel_meshing/surfel_meshing.h"

using namespace vis;

TEST(Triangulation, CheckSurfelState) {
  constexpr bool kShowVisualization = true;
  
  constexpr int kSurfelCount = 1000;
  constexpr float kSurfelRange = 1.0f;
  constexpr float kSurfelRadius = 0.1f;
  
  shared_ptr<SurfelMeshingRenderWindow> render_window;
  shared_ptr<RenderWindow> generic_render_window;
  if (kShowVisualization) {
    render_window.reset(new SurfelMeshingRenderWindow(
        /*render_new_surfels_as_splats*/ false,
        /*splat_half_extent_in_pixels*/ 2.5f,
        /*triangle_normal_shading*/ false,
        /*render_camera_frustum*/ false));
    generic_render_window =
        RenderWindow::CreateWindow("SurfelFusion Triangulation Test", RenderWindow::API::kOpenGL, render_window);
  }
  
  SurfelMeshing reconstruction(
      50,
      M_PI / 180.0f * 90.0f,
      M_PI / 180.0f * 10.0f,
      M_PI / 180.0f * 170.0f,
      2.0,
      1.5,
      30,
      render_window);
  
  if (kShowVisualization) {
    render_window->SetReconstructionForDebugging(&reconstruction);
  }
  
  CUDASurfelsCPU input(kSurfelCount);
  CUDASurfelBuffersCPU* b = input.write_buffers();
  
  srand(0);
  
  input.LockWriteBuffers();
  b->frame_index = 1;
  b->surfel_count = kSurfelCount;
  for (usize i = 0; i < kSurfelCount; ++ i) {
    Vec3f surfel_position = 0.5f * kSurfelRange * Vec3f::Random();
    Vec3f surfel_normal = Vec3f(1, 0, 0); // Vec3f::Random().normalized();
    
    b->surfel_x_buffer[i] = surfel_position.x();
    b->surfel_y_buffer[i] = surfel_position.y();
    b->surfel_z_buffer[i] = surfel_position.z();
    b->surfel_radius_squared_buffer[i] = kSurfelRadius * kSurfelRadius;
    b->surfel_normal_x_buffer[i] = surfel_normal.x();
    b->surfel_normal_y_buffer[i] = surfel_normal.y();
    b->surfel_normal_z_buffer[i] = surfel_normal.z();
    b->surfel_last_update_stamp_buffer[i] = 1;
  }
  input.UnlockWriteBuffers();
  
  input.WaitForLockAndSwapBuffers();
  reconstruction.IntegrateCUDABuffers(
      input.read_buffers().frame_index,
      input);
  reconstruction.CheckRemeshing();
  reconstruction.Triangulate();
  
  for (usize surfel_index = 0; surfel_index < kSurfelCount; ++ surfel_index) {
    reconstruction.CheckSurfelState(surfel_index);
  }
  
  if (kShowVisualization) {
    shared_ptr<Point3fCloud> visualization_cloud(new Point3fCloud());
    reconstruction.ConvertToPoint3fCloud(visualization_cloud.get());
    shared_ptr<Point3fC3u8Cloud> color_cloud(new Point3fC3u8Cloud(visualization_cloud->size()));
    for (usize i = 0; i < visualization_cloud->size(); ++ i) {
      color_cloud->at(i).position() = visualization_cloud->at(i).position();
      color_cloud->at(i).color() = Vec3u8(0, 0, 0);
    }
    render_window->UpdateVisualizationCloud(color_cloud);
    
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    reconstruction.ConvertToMesh3fCu8(visualization_mesh.get());
    unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
    render_window->UpdateVisualizationMesh(visualization_mesh);
    render_mutex_lock.unlock();
    
    std::getchar();
  }
  
  // Remove some triangles and test again.
  LOG(INFO) << "--- Testing triangle removal ---";
  for (int i = 0; i < 10; ++ i) {
    reconstruction.RemeshTrianglesAt(
        const_cast<Surfel*>(&reconstruction.surfels()[i]),
        (2 * 2) * reconstruction.surfels()[i].radius_squared());
  }
  
  // Code copied from above.
  for (usize surfel_index = 0; surfel_index < kSurfelCount; ++ surfel_index) {
    reconstruction.CheckSurfelState(surfel_index);
  }
  
  if (kShowVisualization) {
    shared_ptr<Point3fCloud> visualization_cloud(new Point3fCloud());
    reconstruction.ConvertToPoint3fCloud(visualization_cloud.get());
    shared_ptr<Point3fC3u8Cloud> color_cloud(new Point3fC3u8Cloud(visualization_cloud->size()));
    for (usize i = 0; i < visualization_cloud->size(); ++ i) {
      color_cloud->at(i).position() = visualization_cloud->at(i).position();
      color_cloud->at(i).color() = Vec3u8(0, 0, 0);
    }
    render_window->UpdateVisualizationCloud(color_cloud);
    
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    reconstruction.ConvertToMesh3fCu8(visualization_mesh.get());
    unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
    render_window->UpdateVisualizationMesh(visualization_mesh);
    render_mutex_lock.unlock();
    
    std::getchar();
  }
}
