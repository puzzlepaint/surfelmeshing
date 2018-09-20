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


// Must be included before surfel_meshing_render_window.h to avoid errors
#include <QApplication>
#include <QClipboard>

#include "surfel_meshing/surfel_meshing_render_window.h"

#include <GL/glew.h>
#include <GL/glx.h>

#include "surfel_meshing/cuda_surfel_reconstruction.cuh"
#include <libvis/cuda/cuda_util.h>

namespace vis {

// If true the viewing direction is used as light direction, otherwise the
// light comes from the top (which might not always be positioned correctly).
constexpr bool kUseViewingDirAsLightSource = true;

// Activate debugging functions which are not thread-safe?
constexpr bool kUnsafeDebugging = false;


SurfelMeshingRenderWindow::SurfelMeshingRenderWindow(
    bool render_new_surfels_as_splats,
    float splat_half_extent_in_pixels,
    bool triangle_normal_shading,
    bool render_camera_frustum)
    : have_visualization_cloud_(false), have_visualization_mesh_(false) {
  width_ = 0;
  height_ = 0;
  
  dragging_ = false;
  pressed_mouse_buttons_ = 0;
  m_pressed_ = false;
  
  // Set default view parameters.
  min_depth_ = 0.01f;
  max_depth_ = 50.0f;
  
  camera_free_orbit_theta_ = 0.5;
  camera_free_orbit_phi_ = -1.57;
  camera_free_orbit_radius_ = 6;
  camera_free_orbit_offset_ = Vec3f(0, 0, 0);
  
  use_camera_matrix_ = false;
  
  render_new_surfels_as_splats_ = render_new_surfels_as_splats;
  splat_half_extent_in_pixels_ = splat_half_extent_in_pixels;
  triangle_normal_shading_ = triangle_normal_shading;
  render_camera_frustum_ = render_camera_frustum;
  
  // up_direction_rotation_ = Mat3f::Identity();
  up_direction_rotation_ = AngleAxisf(M_PI, Vec3f(0, 0, 1)) * AngleAxisf(-M_PI / 2, Vec3f(1, 0, 0));
  
  camera_frustum_set_ = false;
  
  visualization_cloud_size_ = numeric_limits<usize>::max();
  new_visualization_cloud_size_ = numeric_limits<usize>::max();
  mesh_surfel_count_ = numeric_limits<u32>::max();
  new_mesh_surfel_count_ = 0;
  
  neighbor_index_buffer_ = 0;
  normal_vertex_buffer_ = 0;
  
  init_max_point_count_ = 0;
  
  reconstruction_ = nullptr;
  selected_surfel_index_ = 0;
  
  render_as_wireframe_ = false;
  show_surfels_ = false;
  show_mesh_ = true;
}

void SurfelMeshingRenderWindow::Initialize() {
  GLenum glew_init_result = glewInit();
  CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
  glGetError();  // Ignore GL_INVALID_ENUM​ error caused by glew
  CHECK_OPENGL_NO_ERROR();
  
  CreateSplatProgram();
  CHECK_OPENGL_NO_ERROR();
  
  CreateMeshProgram();
  CHECK_OPENGL_NO_ERROR();
  
  CreateConstantColorProgram();
  CHECK_OPENGL_NO_ERROR();
  
  CreateTriangleNormalShadedProgram();
  CHECK_OPENGL_NO_ERROR();
  
  // TODO: Inserted to make it work on my laptop with Intel graphics chip. It
  //       would probably be preferable to handle this in a sane way instead of
  //       simply creating a global VAO at the beginning and then forgetting
  //       about it.
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
}

void SurfelMeshingRenderWindow::Resize(int width, int height) {
  width_ = width;
  height_ = height;
}

void SurfelMeshingRenderWindow::Render() {
  CHECK_OPENGL_NO_ERROR();
  
  unique_lock<mutex> lock(render_mutex_);
  
  
  // ### Setup ###
  
  // Setup the render_camera_.
  SetCamera();
  
  unique_lock<mutex> camera_mutex_lock(camera_mutex_);
  
  // Compute projection_matrix_ from the camera.
  ComputeProjectionMatrix();
  
  // Set the rendering bounds (viewport) according to the camera dimensions.
  SetupViewport();
  
  // Set the camera_T_world_ transformation according to an orbiting scheme.
  SetViewpoint();
  
  // Cache the following at the same time the view pose is retrieved to get a consistent state.
  bool cached_camera_frustum_set = camera_frustum_set_;
  SE3f cached_global_T_camera_frustum = global_T_camera_frustum_;
  Vec3f cached_viewing_dir = camera_T_world_.rotationMatrix().transpose() * Vec3f(0, 0, 1);
  
  camera_mutex_lock.unlock();
  
  CHECK_OPENGL_NO_ERROR();
  
  // Compute the model-view-projection matrix.
  Mat4f model_matrix = Mat4f::Identity();
  Mat4f model_view_matrix = camera_T_world_.matrix() * model_matrix;
  model_view_projection_matrix_ = projection_matrix_ * model_view_matrix;
  
  // Set states for rendering.
  glClearColor(0.9f, 0.9f, 0.9f, 1.0f);  // background color
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  // Render.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  //glShadeModel(GL_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  CHECK_OPENGL_NO_ERROR();
  
  
  // ### Rendering ###
  
  // Point cloud rendering of surfels.
  RenderPointSplats();
  
  // Mesh rendering.
  if (show_mesh_) {
    RenderMesh(model_matrix, cached_viewing_dir);
  }
  
  // Camera frustum rendering.
  if (render_camera_frustum_ && cached_camera_frustum_set) {
    RenderCameraFrustum(cached_global_T_camera_frustum);
  }
  
  // Neighbor debug rendering.
  RenderNeighbors();
  
  // Normal debug rendering.
  RenderNormals();
  
  
  // InitializeForCUDAInterop() body
  if (init_max_point_count_ > 0) {
    InitializeForCUDAInteropInRenderingThread();
  }
  
  
  // Take screenshot?
  unique_lock<mutex> screenshot_lock(screenshot_mutex_);
  if (!screenshot_path_.empty()) {
    Image<Vec3u8> image(width_, height_, width_ * sizeof(Vec3u8), 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    CHECK_OPENGL_NO_ERROR();
    
    image.FlipY();
    image.Write(screenshot_path_);
    
    screenshot_path_ = "";
    screenshot_lock.unlock();
    screenshot_condition_.notify_all();
  } else {
    screenshot_lock.unlock();
  }
};

void SurfelMeshingRenderWindow::RenderPointSplats() {
  unique_lock<mutex> cloud_lock(visualization_cloud_mutex_);
  if (new_visualization_cloud_size_ != numeric_limits<usize>::max()) {
    // Option 1: new_visualization_cloud_size_ is specified, the vertex data is
    //           on the GPU already.
    have_visualization_cloud_ = true;
    visualization_cloud_size_ = new_visualization_cloud_size_;
    mesh_surfel_count_ = new_mesh_surfel_count_;
    new_visualization_cloud_size_ = numeric_limits<usize>::max();
  }
  cloud_lock.unlock();
  
  if (show_surfels_ ||
      (render_new_surfels_as_splats_ && mesh_surfel_count_ != numeric_limits<u32>::max())) {
    unique_lock<mutex> cloud_lock(visualization_cloud_mutex_);
    if (new_visualization_cloud_) {
      // Option 2: new_visualization_cloud_ is specified, the vertex data must
      //           be transferred to the GPU.
      have_visualization_cloud_ = true;
      
      visualization_cloud_.TransferToGPU(*new_visualization_cloud_, GL_DYNAMIC_DRAW);
      CHECK_OPENGL_NO_ERROR();
      
      current_visualization_cloud_ = new_visualization_cloud_;
      new_visualization_cloud_.reset();
    }
    cloud_lock.unlock();
    
    // Render the visualization cloud if a cloud is available.
    if (have_visualization_cloud_) {
      splat_program_.UseProgram();
      splat_program_.SetUniformMatrix4f(
          splat_u_model_view_projection_matrix_location_,
          model_view_projection_matrix_);
      splat_program_.SetUniform1f(splat_u_point_size_x_location_, splat_half_extent_in_pixels_ / width_);
      splat_program_.SetUniform1f(splat_u_point_size_y_location_, splat_half_extent_in_pixels_ / height_);
      CHECK_OPENGL_NO_ERROR();
      
      if (visualization_cloud_size_ != numeric_limits<usize>::max()) {
        visualization_cloud_.SetAttributes(&splat_program_);
        if (show_surfels_) {
          // Render all surfels.
          glDrawArrays(GL_POINTS, 0, visualization_cloud_size_);
        } else {
          // Render only new surfels which are not in the mesh yet. These have
          // the index range [mesh_surfel_count_, visualization_cloud_size_[.
          if (visualization_cloud_size_ - mesh_surfel_count_ > 0) {
            glDrawArrays(GL_POINTS, mesh_surfel_count_, visualization_cloud_size_ - mesh_surfel_count_);
          }
        }
      } else {
        visualization_cloud_.Render(&splat_program_);
      }
      CHECK_OPENGL_NO_ERROR();
    }
  }
}

void SurfelMeshingRenderWindow::RenderMesh(const Mat4f& model_matrix, const Vec3f& viewing_dir) {
  // Transfer a new mesh to the GPU?
  unique_lock<mutex> mesh_lock(visualization_mesh_mutex_);
  if (new_visualization_mesh_) {
    have_visualization_mesh_ = true;
    
    visualization_mesh_.TransferToGPU(*new_visualization_mesh_, GL_DYNAMIC_DRAW);
    CHECK_OPENGL_NO_ERROR();
    
    new_visualization_mesh_.reset();
  }
  mesh_lock.unlock();
  
  // Render mesh if one is available.
  if (have_visualization_mesh_) {
    ShaderProgramOpenGL* render_program;
    if (triangle_normal_shading_) {
      render_program = &tri_normal_shaded_program_;
      render_program->UseProgram();
      render_program->SetUniformMatrix4f(
          tri_normal_shaded_u_model_matrix_location_,
          model_matrix);
      render_program->SetUniformMatrix4f(
          tri_normal_shaded_u_model_view_projection_matrix_location_,
          model_view_projection_matrix_);
      if (kUseViewingDirAsLightSource) {
        render_program->SetUniform3f(
            tri_normal_shaded_u_light_source_location_,
            -viewing_dir.x(),
            -viewing_dir.y(),
            -viewing_dir.z());
      }
    } else {
      render_program = &mesh_program_;
      render_program->UseProgram();
      render_program->SetUniformMatrix4f(
          mesh_u_model_view_projection_matrix_location_,
          model_view_projection_matrix_);
    }
    
    if (render_as_wireframe_) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    CHECK_OPENGL_NO_ERROR();
    
    if (visualization_cloud_size_ != numeric_limits<usize>::max()) {
      // Render using the mesh's indices, but the vertex buffer from visualization_cloud_.
      visualization_cloud_.SetAttributes(render_program);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, visualization_mesh_.index_buffer_name());
      glDrawElements(GL_TRIANGLES, visualization_mesh_.index_count(), GL_UNSIGNED_INT,
                      reinterpret_cast<char*>(0) + 0);
    } else {
      visualization_mesh_.Render(render_program);
    }
    CHECK_OPENGL_NO_ERROR();
    
    if (render_as_wireframe_) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
  }
}

void SurfelMeshingRenderWindow::RenderCameraFrustum(const SE3f& global_T_camera_frustum) {
  constexpr float kScaling = 0.1f;  // Equals the z (forward) extent of the frustum.
  Mat4f scaling_matrix = Mat4f::Identity();
  scaling_matrix(0, 0) = kScaling;
  scaling_matrix(1, 1) = kScaling;
  scaling_matrix(2, 2) = kScaling;
  Mat4f frustum_model_view_projection_matrix =
      projection_matrix_ * camera_T_world_.matrix() * global_T_camera_frustum.matrix() * scaling_matrix;
  
  constant_color_program_.UseProgram();
  constant_color_program_.SetUniformMatrix4f(
        constant_color_u_model_view_projection_matrix_location_,
        frustum_model_view_projection_matrix);
  constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 0.1f, 0.1f, 0.1f);
  
  glLineWidth(2);
  camera_frustum_.Render(&constant_color_program_);
  glLineWidth(1);
}

void SurfelMeshingRenderWindow::RenderNeighbors() {
  if (neighbor_index_buffer_ > 0) {
    constant_color_program_.UseProgram();
    constant_color_program_.SetUniformMatrix4f(
          constant_color_u_model_view_projection_matrix_location_,
          model_view_projection_matrix_);
    constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 1, 0, 0);
    
    visualization_cloud_.SetAttributes(&constant_color_program_);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, neighbor_index_buffer_);
    glDrawElements(GL_LINES, 2 * kSurfelNeighborCount * visualization_cloud_size_, GL_UNSIGNED_INT,
                   reinterpret_cast<char*>(0) + 0);
  }
}

void SurfelMeshingRenderWindow::RenderNormals() {
  if (normal_vertex_buffer_ > 0) {
    constant_color_program_.UseProgram();
    constant_color_program_.SetUniformMatrix4f(
          constant_color_u_model_view_projection_matrix_location_,
          model_view_projection_matrix_);
    constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 0, 0, 1);
    
    glBindBuffer(GL_ARRAY_BUFFER, normal_vertex_buffer_);
    constant_color_program_.SetPositionAttribute(3, GL_FLOAT, 3 * sizeof(float), 0);
    
    glDrawArrays(GL_LINES, 0, 2 * visualization_cloud_size_);
  }
}

void SurfelMeshingRenderWindow::MouseDown(MouseButton button, int x, int y) {
  pressed_mouse_buttons_ |= static_cast<int>(button);
  
  if (button == MouseButton::kLeft ||
      button == MouseButton::kMiddle) {
    dragging_ = true;
    last_drag_x_ = x;
    last_drag_y_ = y;
  } else if (button == MouseButton::kRight) {
    if (kUnsafeDebugging && reconstruction_) {
      // NOTE: Accessing reconstruction_ without any thread-safety.
      //       This is for debugging only.
      
      Mat3f camera_R_world = camera_T_world_.rotationMatrix();
      Vec3f camera_T_world = camera_T_world_.translation();
      usize surfel_index = 0;
      float closest_vertex_distance_sq = numeric_limits<float>::infinity();
      for (usize i = 0; i < reconstruction_->surfels().size(); ++ i) {
        Vec3f camera_point = camera_R_world * reconstruction_->surfels().at(i).position() + camera_T_world;
        Vec2f image_point;
        if (render_camera_.ProjectToPixelCenterConvIfVisible(camera_point, 0, &image_point)) {
          float vertex_distance_sq = (image_point - Vec2f(x, y)).squaredNorm();
          if (vertex_distance_sq < closest_vertex_distance_sq) {
            closest_vertex_distance_sq = vertex_distance_sq;
            surfel_index = i;
          }
        }
      }
      
      if (!isinf(closest_vertex_distance_sq)) {
        selected_surfel_index_ = surfel_index;
        
        const Surfel& surfel = reconstruction_->surfels().at(surfel_index);
        LOG(INFO) << "Closest clicked surfel index: " << surfel_index;
        LOG(INFO) << "Surfel attributes: " << surfel;
        
        std::set<u32> surfel_front_neighbors;
        if (surfel.meshing_state() == Surfel::MeshingState::kFront) {
          auto surfel_front = &surfel.fronts();
          LOG(INFO) << "Number of fronts: " << surfel_front->size();
          for (const Front& front : *surfel_front) {
            surfel_front_neighbors.insert(front.left);
            surfel_front_neighbors.insert(front.right);
          }
        }
        
        shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
        reconstruction_->ConvertToMesh3fCu8(visualization_mesh.get());
        
        (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
        if (surfel.meshing_state() == Surfel::MeshingState::kFront) {
          for (u32 front_surfel_index : surfel_front_neighbors) {
            (*visualization_mesh->vertices_mutable())->at(front_surfel_index).color() = Vec3u8(255, 255, 60);
          }
        }
        
        UpdateVisualizationMesh(visualization_mesh);
      }
    } else {
      unique_lock<mutex> cloud_lock(visualization_cloud_mutex_);
      if (!current_visualization_cloud_) {
        return;
      }
      
      Mat3f camera_R_world = camera_T_world_.rotationMatrix();
      Vec3f camera_T_world = camera_T_world_.translation();
      usize closest_vertex_index = 0;
      float closest_vertex_distance_sq = numeric_limits<float>::infinity();
      for (usize i = 0; i < current_visualization_cloud_->size(); ++ i) {
        Vec3f camera_point = camera_R_world * current_visualization_cloud_->at(i).position() + camera_T_world;
        Vec2f image_point;
        if (render_camera_.ProjectToPixelCenterConvIfVisible(camera_point, 0, &image_point)) {
          float vertex_distance_sq = (image_point - Vec2f(x, y)).squaredNorm();
          if (vertex_distance_sq < closest_vertex_distance_sq) {
            closest_vertex_distance_sq = vertex_distance_sq;
            closest_vertex_index = i;
          }
        }
      }
      if (!isinf(closest_vertex_distance_sq)) {
        LOG(INFO) << "Closest clicked vertex index: " << closest_vertex_index;
      }
    }
  }
}

void SurfelMeshingRenderWindow::MouseMove(int x, int y) {
  if (dragging_) {
    bool move_camera = false;
    bool rotate_camera = false;
    
    move_camera = m_pressed_ ||
                  (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kMiddle)) ||
                  ((pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft)) &&
                    (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kRight)));
    rotate_camera = pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft);
    
    int x_distance = x - last_drag_x_;
    int y_distance = y - last_drag_y_;

    if (move_camera) {
      const float right_phi = camera_free_orbit_phi_ + 0.5f * M_PI;
      const Eigen::Vector3f right_vector =
          Eigen::Vector3f(cosf(right_phi), sinf(right_phi), 0.f);
      const float up_theta = camera_free_orbit_theta_ + 0.5f * M_PI;
      const float phi = camera_free_orbit_phi_;
      const Eigen::Vector3f up_vector =
          -1 * Eigen::Vector3f(sinf(up_theta) * cosf(phi),
                                sinf(up_theta) * sinf(phi), cosf(up_theta));
      
      // Camera move speed in units per pixel for 1 unit orbit radius.
      constexpr float kCameraMoveSpeed = 0.001f;
      unique_lock<mutex> lock(camera_mutex_);
      camera_free_orbit_offset_ -= x_distance * kCameraMoveSpeed *
                                    camera_free_orbit_radius_ * right_vector;
      camera_free_orbit_offset_ += y_distance * kCameraMoveSpeed *
                                    camera_free_orbit_radius_ * up_vector;
      lock.unlock();
      
      window_->RenderFrame();
    } else if (rotate_camera) {
      unique_lock<mutex> lock(camera_mutex_);
      camera_free_orbit_theta_ -= y_distance * 0.01f;
      camera_free_orbit_phi_ -= x_distance * 0.01f;

      camera_free_orbit_theta_ = fmin(camera_free_orbit_theta_, 3.14f);
      camera_free_orbit_theta_ = fmax(camera_free_orbit_theta_, 0.01f);
      lock.unlock();
      
      window_->RenderFrame();
    }
  }
  
  last_drag_x_ = x;
  last_drag_y_ = y;
}

void SurfelMeshingRenderWindow::MouseUp(MouseButton button, int /*x*/, int /*y*/) {
  pressed_mouse_buttons_ &= ~static_cast<int>(button);
  
  if (button == MouseButton::kLeft) {
    dragging_ = false;
  }
}

void SurfelMeshingRenderWindow::WheelRotated(float degrees, Modifier /*modifiers*/) {
  double num_steps = -1 * (degrees / 15.0);
  
  // Zoom camera.
  double scale_factor = powf(powf(2.0, 1.0 / 5.0), num_steps);
  camera_free_orbit_radius_ *= scale_factor;
  
  window_->RenderFrame();
}

void SurfelMeshingRenderWindow::KeyPressed(char key, Modifier /*modifiers*/) {
  if (key == 'w') {
    render_as_wireframe_ = !render_as_wireframe_;
    window_->RenderFrame();
  } else if (key == 's') {
    show_surfels_ = !show_surfels_;
    window_->RenderFrame();
  } else if (key == 'h') {
    show_mesh_ = !show_mesh_;
    window_->RenderFrame();
  } else if (key == 'm') {
    m_pressed_ = true;
  } else if (key == 'c') {
    // Copy camera pose (as text).
    unique_lock<mutex> lock(camera_mutex_);
    
    QClipboard* clipboard = QApplication::clipboard();
    clipboard->setText(
        QString::number(camera_free_orbit_offset_.x()) + " " +
        QString::number(camera_free_orbit_offset_.y()) + " " +
        QString::number(camera_free_orbit_offset_.z()) + " " +
        QString::number(camera_free_orbit_radius_) + " " +
        QString::number(camera_free_orbit_theta_) + " " +
        QString::number(camera_free_orbit_phi_));
  } else if (key == 'v') {
    // Paste copied camera pose.
    QClipboard* clipboard = QApplication::clipboard();
    QString text = clipboard->text();
    QStringList list = text.split(' ');
    if (list.size() != 6) {
      LOG(ERROR) << "Cannot parse clipboard content as camera pose!";
    } else {
      unique_lock<mutex> lock(camera_mutex_);
      
      camera_free_orbit_offset_.x() = list[0].toFloat();
      camera_free_orbit_offset_.y() = list[1].toFloat();
      camera_free_orbit_offset_.z() = list[2].toFloat();
      camera_free_orbit_radius_ = list[3].toFloat();
      camera_free_orbit_theta_ = list[4].toFloat();
      camera_free_orbit_phi_ = list[5].toFloat();
      
      lock.unlock();
      window_->RenderFrame();
    }
  } else if (key == 'd') {
    max_depth_ /= 2.0f;
    LOG(INFO) << "max_depth_: " << max_depth_;
    window_->RenderFrame();
  } else if (key == 'i') {
    max_depth_ *= 2.0f;
    LOG(INFO) << "max_depth_: " << max_depth_;
    window_->RenderFrame();
  }
}

void SurfelMeshingRenderWindow::KeyReleased(char key, Modifier /*modifiers*/) {
  if (key == 'm') {
    m_pressed_ = false;
  }
}

void SurfelMeshingRenderWindow::InitializeForCUDAInterop(
    usize max_point_count,
    cudaGraphicsResource** vertex_buffer_resource,
    OpenGLContext* context,
    const Camera& camera,
    bool debug_neighbor_rendering,
    bool debug_normal_rendering,
    cudaGraphicsResource** neighbor_index_buffer_resource,
    cudaGraphicsResource** normal_vertex_buffer_resource) {
  // Unfortunately, it is not possible to make a QOpenGLContext current in
  // another thread, so we have to send these variables to the render thread.
  // This thread will then call InitializeForCUDAInteropInRenderingThread()
  // while the context is active to do the actial initialization.
  
  unique_lock<mutex> render_lock(render_mutex_);
  init_max_point_count_ = max_point_count;
  init_vertex_buffer_resource_ = vertex_buffer_resource;
  init_camera_ = &camera;
  init_debug_neighbor_rendering_ = debug_neighbor_rendering;
  init_debug_normal_rendering_ = debug_normal_rendering;
  init_neighbor_index_buffer_resource_ = neighbor_index_buffer_resource;
  init_normal_vertex_buffer_resource_ = normal_vertex_buffer_resource;
  init_done_ = false;
  render_lock.unlock();
  
  window_->RenderFrame();
  
  unique_lock<mutex> init_lock(init_mutex_);
  while (!init_done_) {
    init_condition_.wait(init_lock);
  }
  init_lock.unlock();
  
  // Initialize a windowless OpenGL context which shares names with the Qt
  // OpenGL context.
  context->InitializeWindowless(&qt_gl_context_);
  qt_gl_context_.Detach();
}

void SurfelMeshingRenderWindow::InitializeForCUDAInteropInRenderingThread() {
  // Initialize vertex buffer for surfels.
  visualization_cloud_.AllocateBuffer(
      init_max_point_count_ * sizeof(Point3fC3u8), GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  // Register the vertex buffer with CUDA.
  cudaGraphicsGLRegisterBuffer(init_vertex_buffer_resource_,
                               visualization_cloud_.buffer_name(),
                               cudaGraphicsMapFlagsWriteDiscard);
  CHECK_CUDA_NO_ERROR();
  
  // Create camera frustum.
  camera_frustum_.Create(*init_camera_);
  
  // Create and register the neighbor index buffer with CUDA (if debugging).
  if (init_debug_neighbor_rendering_) {
    glGenBuffers(1, &neighbor_index_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, neighbor_index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * kSurfelNeighborCount * init_max_point_count_ * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    cudaGraphicsGLRegisterBuffer(init_neighbor_index_buffer_resource_, neighbor_index_buffer_, cudaGraphicsMapFlagsWriteDiscard);
    CHECK_CUDA_NO_ERROR();
  }
  
  // Create and register the normal buffers (if debugging).
  if (init_debug_normal_rendering_) {
    // vertex buffer
    glGenBuffers(1, &normal_vertex_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, normal_vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, 2 * 3 * init_max_point_count_ * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    cudaGraphicsGLRegisterBuffer(init_normal_vertex_buffer_resource_, normal_vertex_buffer_, cudaGraphicsMapFlagsWriteDiscard);
    CHECK_CUDA_NO_ERROR();
  }
  
  // Store the context.
  qt_gl_context_.AttachToCurrent();
  
  
  // Reset init_max_point_count_ such that this function is not called again.
  init_max_point_count_ = 0;
  
  // Signal completion.
  unique_lock<mutex> init_lock(init_mutex_);
  init_done_ = true;
  init_lock.unlock();
  init_condition_.notify_all();
}

void SurfelMeshingRenderWindow::UpdateVisualizationCloud(const shared_ptr<Point3fC3u8Cloud>& cloud) {
  unique_lock<mutex> lock(visualization_cloud_mutex_);
  new_visualization_cloud_ = cloud;
  new_visualization_cloud_size_ = numeric_limits<usize>::max();
  window_->RenderFrame();
}

void SurfelMeshingRenderWindow::UpdateVisualizationCloudCUDA(u32 surfel_count, u32 latest_mesh_surfel_count) {
  unique_lock<mutex> lock(visualization_cloud_mutex_);
  new_visualization_cloud_ = nullptr;
  new_visualization_cloud_size_ = surfel_count;
  new_mesh_surfel_count_ = latest_mesh_surfel_count;
  window_->RenderFrame();
}

void SurfelMeshingRenderWindow::UpdateVisualizationMesh(const std::shared_ptr<vis::Mesh3fCu8>& mesh) {
  unique_lock<mutex> lock(visualization_mesh_mutex_);
  new_visualization_mesh_ = mesh;
  if (new_visualization_mesh_->vertices()) {
    visualization_cloud_size_ = numeric_limits<usize>::max();
  }
  window_->RenderFrame();
}

void SurfelMeshingRenderWindow::UpdateVisualizationMeshCUDA(const std::shared_ptr<vis::Mesh3fCu8>& mesh) {
  UpdateVisualizationMesh(mesh);  // No difference to the non-CUDA variant
}

void SurfelMeshingRenderWindow::UpdateVisualizationCloudAndMeshCUDA(u32 surfel_count, const std::shared_ptr<vis::Mesh3fCu8>& mesh) {
  unique_lock<mutex> cloud_lock(visualization_cloud_mutex_);
  unique_lock<mutex> mesh_lock(visualization_mesh_mutex_);
  
  new_visualization_cloud_ = nullptr;
  new_visualization_cloud_size_ = surfel_count;
  
  new_visualization_mesh_ = mesh;
  if (new_visualization_mesh_->vertices()) {
    visualization_cloud_size_ = numeric_limits<usize>::max();
  }
  
  window_->RenderFrame();
}

void SurfelMeshingRenderWindow::SetUpDirection(const Vec3f& direction) {
  unique_lock<mutex> lock(camera_mutex_);
  up_direction_rotation_ = Quaternionf::FromTwoVectors(direction, Vec3f(0, 0, 1)).toRotationMatrix();
}

void SurfelMeshingRenderWindow::CenterViewOn(const Vec3f& position) {
  unique_lock<mutex> lock(camera_mutex_);
  SE3f up_direction_rotation_transformation =
      SE3f(up_direction_rotation_, Vec3f::Zero());
  camera_free_orbit_offset_ = up_direction_rotation_transformation * position;
}

void SurfelMeshingRenderWindow::SetView(const Vec3f& look_at, const Vec3f& camera_pos, const SE3f& global_T_camera) {
  unique_lock<mutex> lock(camera_mutex_);
  
  SE3f up_direction_rotation_transformation =
      SE3f(up_direction_rotation_, Vec3f::Zero());
  
  use_camera_matrix_ = false;
  
  camera_free_orbit_offset_ = up_direction_rotation_transformation * look_at;
  
  Vec3f look_at_to_camera = up_direction_rotation_ * (camera_pos - look_at);
  camera_free_orbit_radius_ = look_at_to_camera.norm();
  camera_free_orbit_theta_ = acos(look_at_to_camera.z() / camera_free_orbit_radius_);
  camera_free_orbit_phi_ = atan2(look_at_to_camera.y(), look_at_to_camera.x());
  
  global_T_camera_frustum_ = global_T_camera;
  camera_frustum_set_ = true;
}

void SurfelMeshingRenderWindow::SetView2(const Vec3f& x, const Vec3f& y, const Vec3f& z, const Vec3f& eye, const SE3f& global_T_camera) {
  unique_lock<mutex> lock(camera_mutex_);
  
  use_camera_matrix_ = true;
  camera_matrix_ << x(0),  x(1),  x(2),  -(x.dot(eye)),
                    y(0),  y(1),  y(2),  -(y.dot(eye)),
                    z(0),  z(1),  z(2),  -(z.dot(eye)),
                       0,     0,     0,              1;
  
  global_T_camera_frustum_ = global_T_camera;
  camera_frustum_set_ = true;
}

void SurfelMeshingRenderWindow::SetViewParameters(
    const Vec3f& camera_free_orbit_offset,
    float camera_free_orbit_radius,
    float camera_free_orbit_theta,
    float camera_free_orbit_phi,
    float max_depth,
    const SE3f& global_T_camera) {
  unique_lock<mutex> lock(camera_mutex_);
  
  use_camera_matrix_ = false;
  
  camera_free_orbit_offset_ = camera_free_orbit_offset;
  camera_free_orbit_radius_ = camera_free_orbit_radius;
  camera_free_orbit_theta_ = camera_free_orbit_theta;
  camera_free_orbit_phi_ = camera_free_orbit_phi;
  
  max_depth_ = max_depth;
  
  global_T_camera_frustum_ = global_T_camera;
  camera_frustum_set_ = true;
}

void SurfelMeshingRenderWindow::SetCameraFrustumPose(const SE3f& global_T_camera) {
  unique_lock<mutex> lock(camera_mutex_);
  global_T_camera_frustum_ = global_T_camera;
  camera_frustum_set_ = true;
}

void SurfelMeshingRenderWindow::RenderFrame() {
  window_->RenderFrame();
}

void SurfelMeshingRenderWindow::SaveScreenshot(const char* filepath) {
  // Use render_lock to make sure a new frame is rendered for the screenshot.
  // This way any previous calls to update the camera pose, for example, should
  // take effect.
  unique_lock<mutex> render_lock(render_mutex_);
  unique_lock<mutex> lock(screenshot_mutex_);
  screenshot_path_ = filepath;
  lock.unlock();
  render_lock.unlock();
  
  window_->RenderFrame();
  
  unique_lock<mutex> lock2(screenshot_mutex_);
  while (!screenshot_path_.empty()) {
    screenshot_condition_.wait(lock2);
  }
  lock2.unlock();
}

void SurfelMeshingRenderWindow::GetCameraPoseParameters(
    Vec3f* camera_free_orbit_offset,
    float* camera_free_orbit_radius,
    float* camera_free_orbit_theta,
    float* camera_free_orbit_phi) {
  unique_lock<mutex> lock(camera_mutex_);
  *camera_free_orbit_offset = camera_free_orbit_offset_;
  *camera_free_orbit_radius = camera_free_orbit_radius_;
  *camera_free_orbit_theta = camera_free_orbit_theta_;
  *camera_free_orbit_phi = camera_free_orbit_phi_;
}

void SurfelMeshingRenderWindow::SetReconstructionForDebugging(SurfelMeshing* reconstruction) {
  reconstruction_ = reconstruction;
}

void SurfelMeshingRenderWindow::SetCamera() {
  float camera_parameters[4];
  camera_parameters[0] = height_;  // fx
  camera_parameters[1] = height_;  // fy
  camera_parameters[2] = 0.5 * width_ - 0.5f;  // cx
  camera_parameters[3] = 0.5 * height_ - 0.5f;  // cy
  render_camera_ = PinholeCamera4f(width_, height_, camera_parameters);
}

void SurfelMeshingRenderWindow::SetViewpoint() {
  if (use_camera_matrix_) {
    camera_T_world_ = SE3f(camera_matrix_);
  } else {
    Vec3f look_at = camera_free_orbit_offset_;
    float r = camera_free_orbit_radius_;
    float t = camera_free_orbit_theta_;
    float p = camera_free_orbit_phi_;
    Vec3f look_from =
        look_at + Vec3f(r * sinf(t) * cosf(p), r * sinf(t) * sinf(p),
                                  r * cosf(t));
    
    Vec3f forward = (look_at - look_from).normalized();
    Vec3f up_temp = Vec3f(0, 0, 1);
    Vec3f right = forward.cross(up_temp).normalized();
    Vec3f up = right.cross(forward);
    
    Mat3f world_R_camera;
    world_R_camera.col(0) = right;
    world_R_camera.col(1) = -up;  // Y will be mirrored by the projection matrix to remove the discrepancy between OpenGL's and our coordinate system.
    world_R_camera.col(2) = forward;
    
    SE3f world_T_camera(world_R_camera, look_from);
    camera_T_world_ = world_T_camera.inverse();
    
    SE3f up_direction_rotation_transformation =
        SE3f(up_direction_rotation_, Vec3f::Zero());
    camera_T_world_ = camera_T_world_ * up_direction_rotation_transformation;
  }
}

void SurfelMeshingRenderWindow::ComputeProjectionMatrix() {
  CHECK_GT(max_depth_, min_depth_);
  CHECK_GT(min_depth_, 0);

  const float fx = render_camera_.parameters()[0];
  const float fy = render_camera_.parameters()[1];
  const float cx = render_camera_.parameters()[2];
  const float cy = render_camera_.parameters()[3];

  // Row-wise projection matrix construction.
  projection_matrix_(0, 0) = (2 * fx) / render_camera_.width();
  projection_matrix_(0, 1) = 0;
  projection_matrix_(0, 2) = 2 * (0.5f + cx) / render_camera_.width() - 1.0f;
  projection_matrix_(0, 3) = 0;
  
  projection_matrix_(1, 0) = 0;
  projection_matrix_(1, 1) = -1 * ((2 * fy) / render_camera_.height());
  projection_matrix_(1, 2) = -1 * (2 * (0.5f + cy) / render_camera_.height() - 1.0f);
  projection_matrix_(1, 3) = 0;
  
  projection_matrix_(2, 0) = 0;
  projection_matrix_(2, 1) = 0;
  projection_matrix_(2, 2) = (max_depth_ + min_depth_) / (max_depth_ - min_depth_);
  projection_matrix_(2, 3) = -(2 * max_depth_ * min_depth_) / (max_depth_ - min_depth_);
  
  projection_matrix_(3, 0) = 0;
  projection_matrix_(3, 1) = 0;
  projection_matrix_(3, 2) = 1;
  projection_matrix_(3, 3) = 0;
}

void SurfelMeshingRenderWindow::SetupViewport() {
  glViewport(0, 0, render_camera_.width(), render_camera_.height());
}

void SurfelMeshingRenderWindow::CreateSplatProgram() {
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out vec3 var1_color;\n"
      "void main() {\n"
      "  var1_color = in_color;\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "#extension GL_EXT_geometry_shader : enable\n"
      "layout(points) in;\n"
      "layout(triangle_strip, max_vertices = 4) out;\n"
      "\n"
      "uniform float u_point_size_x;\n"
      "uniform float u_point_size_y;\n"
      "\n"
      "in vec3 var1_color[];\n"
      "out vec3 var2_color;\n"
      "\n"
      "void main() {\n"
      "  var2_color = var1_color[0];\n"
      "  vec4 base_pos = vec4(gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w, 1.0);\n"
      "  gl_Position = base_pos + vec4(-u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(-u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  \n"
      "  EndPrimitive();\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kGeometryShader));
  
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "#extension GL_ARB_explicit_attrib_location : enable\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "in lowp vec3 var2_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = var2_color;\n"
      // For highlighting the splats in red:
//      "  out_color = vec3(1.0, 0.0, 0.0);\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(splat_program_.LinkProgram());
  
  splat_program_.UseProgram();
  
  splat_u_model_view_projection_matrix_location_ =
      splat_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  splat_u_point_size_x_location_ =
      splat_program_.GetUniformLocationOrAbort("u_point_size_x");
  splat_u_point_size_y_location_ =
      splat_program_.GetUniformLocationOrAbort("u_point_size_y");
}

void SurfelMeshingRenderWindow::CreateMeshProgram() {
  CHECK(mesh_program_.AttachShader(
      "#version 300 es\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out vec3 var_color;\n"
      "void main() {\n"
      "  var_color = in_color;\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(mesh_program_.AttachShader(
      "#version 300 es\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "in lowp vec3 var_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = var_color;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(mesh_program_.LinkProgram());
  
  mesh_program_.UseProgram();
  
  mesh_u_model_view_projection_matrix_location_ =
      mesh_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
}

void SurfelMeshingRenderWindow::CreateConstantColorProgram() {
  CHECK(constant_color_program_.AttachShader(
      "#version 300 es\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "void main() {\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(constant_color_program_.AttachShader(
      "#version 300 es\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "\n"
      "uniform lowp vec3 u_constant_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = u_constant_color;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(constant_color_program_.LinkProgram());
  
  constant_color_program_.UseProgram();
  
  constant_color_u_model_view_projection_matrix_location_ =
      constant_color_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  constant_color_u_constant_color_location_ =
      constant_color_program_.GetUniformLocationOrAbort("u_constant_color");
}

void SurfelMeshingRenderWindow::CreateTriangleNormalShadedProgram() {
  if (kUseViewingDirAsLightSource) {
    CHECK(tri_normal_shaded_program_.AttachShader(
        "#version 300 es\n"
        "uniform mat4 u_model_matrix;\n"
        "uniform mat4 u_model_view_projection_matrix;\n"
        "in vec4 in_position;\n"
        "in vec3 in_color;\n"
        "out vec3 var_pos;\n"
        "void main() {\n"
        "  gl_Position = u_model_view_projection_matrix * in_position;\n"
        "  var_pos = (u_model_matrix * in_position).xyz;\n"
        "}\n",
        ShaderProgramOpenGL::ShaderType::kVertexShader));
    
    CHECK(tri_normal_shaded_program_.AttachShader(
        "#version 300 es\n"
        "layout(location = 0) out lowp vec3 out_color;\n"
        "\n"
        "uniform highp vec3 u_light_source;\n"
        "\n"
        "in highp vec3 var_pos;\n"
        "\n"
        "void main() {\n"
        "  highp vec3 model_normal = normalize(cross(dFdx(var_pos), dFdy(var_pos)));\n"
        "  highp float angle = dot(model_normal, u_light_source);\n"
        "  lowp float intensity = 0.8 * angle + 0.2;"
        "  out_color = vec3(intensity, intensity, intensity);\n"
        "}\n",
        ShaderProgramOpenGL::ShaderType::kFragmentShader));
    
    CHECK(tri_normal_shaded_program_.LinkProgram());
    
    tri_normal_shaded_program_.UseProgram();
    
    tri_normal_shaded_u_model_matrix_location_ =
        tri_normal_shaded_program_.GetUniformLocationOrAbort("u_model_matrix");
    tri_normal_shaded_u_model_view_projection_matrix_location_ =
        tri_normal_shaded_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
    tri_normal_shaded_u_light_source_location_ =
        tri_normal_shaded_program_.GetUniformLocationOrAbort("u_light_source");
  } else {
    CHECK(tri_normal_shaded_program_.AttachShader(
        "#version 300 es\n"
        "uniform mat4 u_model_matrix;\n"
        "uniform mat4 u_model_view_projection_matrix;\n"
        "in vec4 in_position;\n"
        "in vec3 in_color;\n"
        "out vec3 var_pos;\n"
        "void main() {\n"
        "  gl_Position = u_model_view_projection_matrix * in_position;\n"
        "  var_pos = (u_model_matrix * in_position).xyz;\n"
        "}\n",
        ShaderProgramOpenGL::ShaderType::kVertexShader));
    
    CHECK(tri_normal_shaded_program_.AttachShader(
        "#version 300 es\n"
        "layout(location = 0) out lowp vec3 out_color;\n"
        "\n"
        "in highp vec3 var_pos;\n"
        "\n"
        "void main() {\n"
        "  highp float model_normal_y = normalize(cross(dFdx(var_pos), dFdy(var_pos))).y;\n"
        "  lowp float intensity = 0.5 * (-model_normal_y + 1.0);\n"
        "  out_color = vec3(intensity, intensity, intensity);\n"
        "}\n",
        ShaderProgramOpenGL::ShaderType::kFragmentShader));
    
    CHECK(tri_normal_shaded_program_.LinkProgram());
    
    tri_normal_shaded_program_.UseProgram();
    
    tri_normal_shaded_u_model_matrix_location_ =
        tri_normal_shaded_program_.GetUniformLocationOrAbort("u_model_matrix");
    tri_normal_shaded_u_model_view_projection_matrix_location_ =
        tri_normal_shaded_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  }
}

}
