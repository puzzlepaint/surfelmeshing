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

#include <atomic>
#include <iomanip>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <cuda_runtime.h>
#include <libvis/logging.h>
#include <libvis/command_line_parser.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/mesh_opengl.h>
#include <libvis/opengl.h>
#include <libvis/opengl_context.h>
#include <libvis/point_cloud.h>
#include <libvis/point_cloud_opengl.h>
#include <libvis/render_window.h>
#include <libvis/rgbd_video.h>
#include <libvis/rgbd_video_io_tum_dataset.h>
#include <libvis/shader_program_opengl.h>
#include <libvis/sophus.h>
#include <libvis/timing.h>
#include <signal.h>
#include <spline_library/splines/uniform_cr_spline.h>
#include <termios.h>
#include <unistd.h>

#include "surfel_meshing/asynchronous_meshing.h"
#include "surfel_meshing/cuda_depth_processing.cuh"
#include "surfel_meshing/cuda_surfel_reconstruction.h"
#include "surfel_meshing/surfel_meshing_render_window.h"
#include "surfel_meshing/surfel.h"
#include "surfel_meshing/surfel_meshing.h"

using namespace vis;


// Get a key press from the terminal without requiring the Return key to confirm.
// From https://stackoverflow.com/questions/421860
char getch() {
  char buf = 0;
  struct termios old = {0};
  if (tcgetattr(0, &old) < 0) {
    perror("tcsetattr()");
  }
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &old) < 0) {
    perror("tcsetattr ICANON");
  }
  if (read(0, &buf, 1) < 0) {
    perror ("read()");
  }
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if (tcsetattr(0, TCSADRAIN, &old) < 0) {
    perror ("tcsetattr ~ICANON");
  }
  return (buf);
}


// Helper to use splines from the used spline library with single-dimension values.
struct FloatForSpline {
  FloatForSpline(float value)
      : value(value) {}
  
  float length() const {
    return std::fabs(value);
  }
  
  operator float() const {
    return value;
  }
  
  float value;
};


// Saves the reconstructed mesh as an OBJ file.
bool SaveMeshAsOBJ(
    CUDASurfelReconstruction& reconstruction,
    SurfelMeshing& surfel_meshing,
    const std::string& export_mesh_path,
    cudaStream_t stream) {
  CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size());
  
  shared_ptr<Mesh3fCu8> mesh(new Mesh3fCu8());
  
  // Also use the positions from the surfel_meshing such that positions
  // and the mesh are from a consistent state.
  surfel_meshing.ConvertToMesh3fCu8(mesh.get());
  
  CUDABuffer<float> position_buffer(1, 3 * reconstruction.surfels_size());
  CUDABuffer<u8> color_buffer(1, 3 * reconstruction.surfels_size());
  reconstruction.ExportVertices(stream, &position_buffer, &color_buffer);
  float* position_buffer_cpu = new float[3 * reconstruction.surfels_size()];
  u8* color_buffer_cpu = new u8[3 * reconstruction.surfels_size()];
  position_buffer.DownloadAsync(stream, position_buffer_cpu);
  color_buffer.DownloadAsync(stream, color_buffer_cpu);
  cudaStreamSynchronize(stream);
  usize index = 0;
  CHECK_EQ(mesh->vertices()->size(), reconstruction.surfel_count());
  for (usize i = 0; i < reconstruction.surfels_size(); ++ i) {
    if (isnan(position_buffer_cpu[3 * i + 0])) {
      continue;
    }
    
    Point3fC3u8* point = &(*mesh->vertices_mutable())->at(index);
    point->color() = Vec3u8(color_buffer_cpu[3 * i + 0],
                            color_buffer_cpu[3 * i + 1],
                            color_buffer_cpu[3 * i + 2]);
    ++ index;
  }
  CHECK_EQ(index, mesh->vertices()->size());
  delete[] color_buffer_cpu;
  delete[] position_buffer_cpu;
  
  // DEBUG:
  // CHECK(mesh->CheckIndexValidity());
  
  if (mesh->WriteAsOBJ(export_mesh_path.c_str())) {
    LOG(INFO) << "Wrote " << export_mesh_path << ".";
    return true;
  } else {
    LOG(ERROR) << "Writing the mesh failed.";
    return false;
  }
}


// Saves the reconstructed surfels as a point cloud in PLY format.
bool SavePointCloudAsPLY(
    CUDASurfelReconstruction& reconstruction,
    SurfelMeshing& surfel_meshing,
    const std::string& export_point_cloud_path) {
  CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size());
  
  Point3fCloud cloud;
  surfel_meshing.ConvertToPoint3fCloud(&cloud);
  
  Point3fC3u8NfCloud final_cloud(cloud.size());
  usize index = 0;
  for (usize i = 0; i < cloud.size(); ++ i) {
    final_cloud[i].position() = cloud[i].position();
    final_cloud[i].color() = Vec3u8(255, 255, 255);  // TODO: Export colors as well.
    while (surfel_meshing.surfels()[index].node() == nullptr) {
      ++ index;
    }
    final_cloud[i].normal() = surfel_meshing.surfels()[index].normal();
    ++ index;
  }
  final_cloud.WriteAsPLY(export_point_cloud_path);
  LOG(INFO) << "Wrote " << export_point_cloud_path << ".";
  return true;
}


// Runs a median filter on the depth map to perform denoising and fill-in.
void MedianFilterAndDensifyDepthMap(const Image<u16>& input, Image<u16>* output) {
  vector<u16> values;
  
  constexpr int kRadius = 1;
  constexpr int kMinNeighbors = 2;
  
  for (int y = 0; y < static_cast<int>(input.height()); ++ y) {
    for (int x = 0; x < static_cast<int>(input.width()); ++ x) {
      values.clear();
      
      int dy_end = std::min<int>(input.height() - 1, y + kRadius);
      for (int dy = std::max<int>(0, static_cast<int>(y) - kRadius);
           dy <= dy_end;
           ++ dy) {
        int dx_end = std::min<int>(input.width() - 1, x + kRadius);
        for (int dx = std::max<int>(0, static_cast<int>(x) - kRadius);
             dx <= dx_end;
             ++ dx) {
          if (input(dx, dy) != 0) {
            values.push_back(input(dx, dy));
          }
        }
      }
      
      if (values.size() >= kMinNeighbors) {
        std::sort(values.begin(), values.end());  // NOTE: slow, need to get center element only
        if (values.size() % 2 == 0) {
          // Take the element which is closer to the average.
          float sum = 0;
          for (u16 value : values) {
            sum += value;
          }
          float average = sum / values.size();
          
          float prev_diff = std::fabs(values[values.size() / 2 - 1] - average);
          float next_diff = std::fabs(values[values.size() / 2] - average);
          (*output)(x, y) = (prev_diff < next_diff) ? values[values.size() / 2 - 1] : values[values.size() / 2];
        } else {
          (*output)(x, y) = values[values.size() / 2];
        }
      } else {
        (*output)(x, y) = input(x, y);
      }
    }
  }
}


int LIBVIS_MAIN(int argc, char** argv) {
  // Ignore SIGTTIN and SIGTTOU. I am not sure why they occurred: it seems that
  // they should only occur for background processes trying to interact with the
  // terminal, but they seemingly happened to me while there was no background
  // process and they interfered with using gdb.
  // TODO: Find out the reason for those signals
  signal(SIGTTIN, SIG_IGN);
  signal(SIGTTOU, SIG_IGN);
  
  
  // ### Parse parameters ###
  
  CommandLineParser cmd_parser(argc, argv);
  
  // Dataset playback parameters.
  float depth_scaling = 5000;  // The default is for TUM RGB-D datasets.
  cmd_parser.NamedParameter(
      "--depth_scaling", &depth_scaling, /*required*/ false,
      "Input depth scaling: input_depth = depth_scaling * depth_in_meters. The default is for TUM RGB-D benchmark datasets.");
  
  int start_frame = 0;
  cmd_parser.NamedParameter(
      "--start_frame", &start_frame, /*required*/ false,
      "First frame of the video to process.");
  
  int end_frame = numeric_limits<int>::max();
  cmd_parser.NamedParameter(
      "--end_frame", &end_frame, /*required*/ false,
      "If the video is longer, processing stops after end_frame.");
  
  int pyramid_level = 0;
  cmd_parser.NamedParameter(
      "--pyramid_level", &pyramid_level, /*required*/ false,
      "Specify the scale-space pyramid level to use. 0 uses the original sized images, 1 uses half the original resolution, etc.");
  
  int fps_restriction = 30;
  cmd_parser.NamedParameter(
      "--restrict_fps_to", &fps_restriction, /*required*/ false,
      "Restrict the frames per second to at most the given number.");
  
  bool step_by_step_playback = cmd_parser.Flag(
      "--step_by_step_playback",
      "Play back video frames step-by-step (do a step by pressing the Return key in the terminal).");
  
  bool invert_quaternions = cmd_parser.Flag(
      "--invert_quaternions",
      "Invert the quaternions loaded from the poses file.");
  
  // Surfel reconstruction parameters.
  int max_surfel_count = 20 * 1000 * 1000;  // 20 million.
  cmd_parser.NamedParameter(
      "--max_surfel_count", &max_surfel_count, /*required*/ false,
      "Maximum number of surfels. Determines the GPU memory requirements.");
  
  float sensor_noise_factor = 0.05f;
  cmd_parser.NamedParameter(
      "--sensor_noise_factor", &sensor_noise_factor, /*required*/ false,
      "Sensor noise range extent as \"factor times the measured depth\". The real measurement is assumed to be in [(1 - sensor_noise_factor) * depth, (1 + sensor_noise_factor) * depth].");
  
  float max_surfel_confidence = 5.0f;
  cmd_parser.NamedParameter(
      "--max_surfel_confidence", &max_surfel_confidence, /*required*/ false,
      "Maximum value for the surfel confidence. Higher values enable more denoising, lower values faster adaptation to changes.");
  
  float regularizer_weight = 10.0f;
  cmd_parser.NamedParameter(
      "--regularizer_weight", &regularizer_weight, /*required*/ false,
      "Weight for the regularization term (w_{reg} in the paper).");
  
  float normal_compatibility_threshold_deg = 40;
  cmd_parser.NamedParameter(
      "--normal_compatibility_threshold_deg", &normal_compatibility_threshold_deg, /*required*/ false,
      "Angle threshold (in degrees) for considering a measurement normal and a surfel normal to be compatible.");
  
  int regularization_frame_window_size = 30;
  cmd_parser.NamedParameter(
      "--regularization_frame_window_size", &regularization_frame_window_size, /*required*/ false,
      "Number of frames for which the regularization of a surfel is continued after it goes out of view.");
  
  bool do_blending = !cmd_parser.Flag(
      "--disable_blending",
      "Disable observation boundary blending.");
  
  int measurement_blending_radius = 12;
  cmd_parser.NamedParameter(
      "--measurement_blending_radius", &measurement_blending_radius, /*required*/ false,
      "Radius for measurement blending in pixels.");
  
  int regularization_iterations_per_integration_iteration = 1;
  cmd_parser.NamedParameter(
      "--regularization_iterations_per_integration_iteration",
      &regularization_iterations_per_integration_iteration, /*required*/ false,
      "Number of regularization (gradient descent) iterations performed per depth integration iteration. Set this to zero to disable regularization.");
  
  float radius_factor_for_regularization_neighbors = 2;
  cmd_parser.NamedParameter(
      "--radius_factor_for_regularization_neighbors", &radius_factor_for_regularization_neighbors, /*required*/ false,
      "Factor on the surfel radius for how far regularization neighbors can be away from a surfel.");
  
  int surfel_integration_active_window_size = numeric_limits<int>::max();
  cmd_parser.NamedParameter(
      "--surfel_integration_active_window_size", &surfel_integration_active_window_size, /*required*/ false,
      "Number of frames which need to pass before a surfel becomes inactive. If there are no loop closures, set this to a value larger than the dataset frame count to disable surfel deactivation.");
  
  // Meshing parameters.
  float max_angle_between_normals_deg = 90.0f;
  cmd_parser.NamedParameter(
      "--max_angle_between_normals_deg", &max_angle_between_normals_deg, /*required*/ false,
      "Maximum angle between normals of surfels that are connected by triangulation.");
  const float max_angle_between_normals = M_PI / 180.0f * max_angle_between_normals_deg;
  
  float min_triangle_angle_deg = 10.0f;
  cmd_parser.NamedParameter(
      "--min_triangle_angle_deg", &min_triangle_angle_deg, /*required*/ false,
      "The meshing algorithm attempts to keep triangle angles larger than this.");
  const float min_triangle_angle = M_PI / 180.0 * min_triangle_angle_deg;
  
  float max_triangle_angle_deg = 170.0f;
  cmd_parser.NamedParameter(
      "--max_triangle_angle_deg", &max_triangle_angle_deg, /*required*/ false,
      "The meshing algorithm attempts to keep triangle angles smaller than this.");
  const float max_triangle_angle = M_PI / 180.0 * max_triangle_angle_deg;
  
  float max_neighbor_search_range_increase_factor = 2.0f;
  cmd_parser.NamedParameter(
      "--max_neighbor_search_range_increase_factor", &max_neighbor_search_range_increase_factor, /*required*/ false,
      "Maximum factor by which the surfel neighbor search range can be increased if the front neighbors are far away.");
  
  float long_edge_tolerance_factor = 1.5f;
  cmd_parser.NamedParameter(
      "--long_edge_tolerance_factor", &long_edge_tolerance_factor, /*required*/ false,
      "Tolerance factor over 'max_neighbor_search_range_increase_factor * surfel_radius' for deciding whether to remesh a triangle with long edges.");
  
  bool asynchronous_triangulation = !cmd_parser.Flag(
      "--synchronous_meshing",
      "Makes the meshing proceed synchronously to the surfel integration (instead of asynchronously).");
  
  bool full_meshing_every_frame = cmd_parser.Flag(
      "--full_meshing_every_frame",
      "Instead of partial remeshing, performs full meshing in every frame. Only implemented for using together with --synchronous_meshing.");
  
  bool full_retriangulation_at_end = cmd_parser.Flag(
      "--full_retriangulation_at_end",
      "Performs a full retriangulation in the end (after the viewer closes, before the mesh is saved).");
  
  // Depth preprocessing parameters.
  float max_depth = 3.0f;
  cmd_parser.NamedParameter(
      "--max_depth", &max_depth, /*required*/ false,
      "Maximum input depth in meters.");
  
  float depth_valid_region_radius = 333;
  cmd_parser.NamedParameter(
      "--depth_valid_region_radius", &depth_valid_region_radius, /*required*/ false,
      "Radius of a circle (centered on the image center) with valid depth. Everything outside the circle is considered to be invalid. Used to discard biased depth at the corners of Kinect v1 depth images.");
  
  float observation_angle_threshold_deg = 85;
  cmd_parser.NamedParameter(
      "--observation_angle_threshold_deg", &observation_angle_threshold_deg, /*required*/ false,
      "If the angle between the inverse observation direction and the measured surface normal is larger than this setting, the surface is discarded.");
  
  int depth_erosion_radius = 2;
  cmd_parser.NamedParameter(
      "--depth_erosion_radius", &depth_erosion_radius, /*required*/ false,
      "Radius for depth map erosion (in [0, 3]). Useful to combat foreground fattening artifacts.");
  
  int median_filter_and_densify_iterations = 0;
  cmd_parser.NamedParameter(
      "--median_filter_and_densify_iterations", &median_filter_and_densify_iterations, /*required*/ false,
      "Number of iterations of median filtering with hole filling. Disabled by default. Can be useful for noisy time-of-flight data.");
  
  int outlier_filtering_frame_count = 8;
  cmd_parser.NamedParameter(
      "--outlier_filtering_frame_count", &outlier_filtering_frame_count, /*required*/ false,
      "Number of other depth frames to use for outlier filtering of a depth frame. Supported values: 2, 4, 6, 8. Should be reduced if using low-frequency input.");
  
  int outlier_filtering_required_inliers = -1;
  cmd_parser.NamedParameter(
      "--outlier_filtering_required_inliers", &outlier_filtering_required_inliers, /*required*/ false,
      "Number of required inliers for accepting a depth value in outlier filtering. With the default value of -1, all other frames (outlier_filtering_frame_count) must be inliers.");
  
  float bilateral_filter_sigma_xy = 3;
  cmd_parser.NamedParameter(
      "--bilateral_filter_sigma_xy", &bilateral_filter_sigma_xy, /*required*/ false,
      "sigma_xy for depth bilateral filtering, in pixels.");
  
  float bilateral_filter_radius_factor = 2.0f;
  cmd_parser.NamedParameter(
      "--bilateral_filter_radius_factor", &bilateral_filter_radius_factor, /*required*/ false,
      "Factor on bilateral_filter_sigma_xy to define the kernel radius for depth bilateral filtering.");
  
  float bilateral_filter_sigma_depth_factor = 0.05;
  cmd_parser.NamedParameter(
      "--bilateral_filter_sigma_depth_factor", &bilateral_filter_sigma_depth_factor, /*required*/ false,
      "Factor on the depth to compute sigma_depth for depth bilateral filtering.");
  
  float outlier_filtering_depth_tolerance_factor = 0.02f;
  cmd_parser.NamedParameter(
      "--outlier_filtering_depth_tolerance_factor", &outlier_filtering_depth_tolerance_factor, /*required*/ false,
      "Factor on the depth to define the size of the inlier region for outlier filtering.");
  
  float point_radius_extension_factor = 1.5f;
  cmd_parser.NamedParameter(
      "--point_radius_extension_factor", &point_radius_extension_factor, /*required*/ false,
      "Factor by which a point's radius is extended beyond the distance to its farthest neighbor.");
  
  float point_radius_clamp_factor = numeric_limits<float>::infinity();
  cmd_parser.NamedParameter(
      "--point_radius_clamp_factor", &point_radius_clamp_factor, /*required*/ false,
      "Factor by which a point's radius can be larger than the distance to its closest neighbor (times sqrt(2)). Larger radii are clamped to this distance.");
  
  // Octree parameters.
  int max_surfels_per_node = 50;
  cmd_parser.NamedParameter(
      "--max_surfels_per_node", &max_surfels_per_node, /*required*/ false,
      "Maximum number of surfels per octree node. Should only affect the runtime.");
  
  // File export parameters.
  std::string export_mesh_path;
  cmd_parser.NamedParameter(
      "--export_mesh", &export_mesh_path, /*required*/ false,
      "Save the final mesh to the given path (as an OBJ file).");
  
  std::string export_point_cloud_path;
  cmd_parser.NamedParameter(
      "--export_point_cloud", &export_point_cloud_path, /*required*/ false,
      "Save the final (surfel) point cloud to the given path (as a PLY file).");
  
  // Visualization parameters.
  bool render_camera_frustum = !cmd_parser.Flag(
      "--hide_camera_frustum",
      "Hides the input camera frustum rendering.");
  
  bool render_new_surfels_as_splats = !cmd_parser.Flag(
      "--hide_new_surfel_splats",
      "Hides the splat rendering of new surfels which are not meshed yet.");
  
  float splat_half_extent_in_pixels = 3.0f;
  cmd_parser.NamedParameter(
      "--splat_half_extent_in_pixels", &splat_half_extent_in_pixels, /*required*/ false,
      "Half splat quad extent in pixels.");
  
  bool triangle_normal_shading = cmd_parser.Flag(
      "--triangle_normal_shading",
      "Colors the mesh triangles based on their triangle normal.");
  
  bool show_input_images = !cmd_parser.Flag(
      "--hide_input_images",
      "Hides the input images (which are normally shown in separate windows).");
  
  int render_window_default_width = 1280;
  cmd_parser.NamedParameter(
      "--render_window_default_width", &render_window_default_width, /*required*/ false,
      "Default width of the 3D visualization window.");
  
  int render_window_default_height = 720;
  cmd_parser.NamedParameter(
      "--render_window_default_height", &render_window_default_height, /*required*/ false,
      "Default height of the 3D visualization window.");
  
  bool show_result = !cmd_parser.Flag(
      "--exit_after_processing",
      "After processing the video, exit immediately instead of continuing to show the reconstruction.");
  
  bool follow_input_camera = !step_by_step_playback;
  std::string follow_input_camera_str;
  cmd_parser.NamedParameter(
      "--follow_input_camera", &follow_input_camera_str, /*required*/ false,
      "Make the visualization camera follow the input camera (true / false).");
  if (follow_input_camera_str == "true") {
    follow_input_camera = true;
  } else if (follow_input_camera_str == "false") {
    follow_input_camera = false;
  } else if (!follow_input_camera_str.empty()) {
    LOG(FATAL) << "Unknown value given for --follow_input_camera parameter: " << follow_input_camera_str;
    return EXIT_FAILURE;
  }
  
  std::string record_keyframes_path;
  cmd_parser.NamedParameter(
      "--record_keyframes", &record_keyframes_path, /*required*/ false,
      "Record keyframes for video recording to the given file. It is recommended to also set --step_by_step_playback and --show_result.");
  
  std::string playback_keyframes_path;
  cmd_parser.NamedParameter(
      "--playback_keyframes", &playback_keyframes_path, /*required*/ false,
      "Play back keyframes for video recording from the given file.");
  
  // Debug and evaluation parameters.
  bool create_video = cmd_parser.Flag(
      "--create_video",
      "Records a video by writing screenshots frame-by-frame to the current working directory.");
  
  bool debug_depth_preprocessing = cmd_parser.Flag(
      "--debug_depth_preprocessing",
      "Activates debug display of the depth maps at various stages of pre-processing.");
  
  bool debug_neighbor_rendering = cmd_parser.Flag(
      "--debug_neighbor_rendering",
      "Activates debug rendering of surfel regularization neighbors.");
  
  bool debug_normal_rendering = cmd_parser.Flag(
      "--debug_normal_rendering",
      "Activates debug rendering of surfel normal vectors.");
  
  bool visualize_last_update_timestamp = cmd_parser.Flag(
      "--visualize_last_update_timestamp",
      "Show a visualization of the surfel last update timestamps.");
  
  bool visualize_creation_timestamp = cmd_parser.Flag(
      "--visualize_creation_timestamp",
      "Show a visualization of the surfel creation timestamps.");
  
  bool visualize_radii = cmd_parser.Flag(
      "--visualize_radii",
      "Show a visualization of the surfel radii.");
  
  bool visualize_surfel_normals = cmd_parser.Flag(
      "--visualize_surfel_normals",
      "Show a visualization of the surfel normals.");
  
  std::string timings_log_path;
  cmd_parser.NamedParameter(
      "--log_timings", &timings_log_path, /*required*/ false,
      "Log the timings to the given file.");
  
  // Required input paths.
  string dataset_folder_path;
  cmd_parser.SequentialParameter(
      &dataset_folder_path, "dataset_folder_path", true,
      "Path to the dataset in TUM RGB-D format.");
  
  string trajectory_filename;
  cmd_parser.SequentialParameter(
      &trajectory_filename, "trajectory_filename", true,
      "Filename of the trajectory file in TUM RGB-D format within the dataset_folder_path (for example, 'trajectory.txt').");
  
  if (!cmd_parser.CheckParameters()) {
    return EXIT_FAILURE;
  }
  
  
  // ### Initialization ###
  
  // Create render window.
  shared_ptr<SurfelMeshingRenderWindow> render_window(
      new SurfelMeshingRenderWindow(render_new_surfels_as_splats,
                                    splat_half_extent_in_pixels,
                                    triangle_normal_shading,
                                    render_camera_frustum));
  shared_ptr<RenderWindow> generic_render_window =
      RenderWindow::CreateWindow("SurfelMeshing", render_window_default_width, render_window_default_height, RenderWindow::API::kOpenGL, render_window);
  
  // Load dataset.
  RGBDVideo<Vec3u8, u16> rgbd_video;
  
  if (!ReadTUMRGBDDatasetAssociatedAndCalibrated(dataset_folder_path.c_str(), trajectory_filename.c_str(), &rgbd_video)) {
    LOG(FATAL) << "Could not read dataset.";
  } else {
    CHECK_EQ(rgbd_video.depth_frames_mutable()->size(), rgbd_video.color_frames_mutable()->size());
    LOG(INFO) << "Read dataset with " << rgbd_video.frame_count() << " frames";
  }
  
  if (invert_quaternions) {
    for (usize frame_index = 0; frame_index < rgbd_video.frame_count(); ++ frame_index) {
      SE3f global_T_frame = rgbd_video.color_frame_mutable(frame_index)->global_T_frame();
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
      rgbd_video.color_frame_mutable(frame_index)->SetGlobalTFrame(global_T_frame);
      
      global_T_frame = rgbd_video.depth_frame_mutable(frame_index)->global_T_frame();
      global_T_frame.setQuaternion(global_T_frame.unit_quaternion().inverse());
      rgbd_video.depth_frame_mutable(frame_index)->SetGlobalTFrame(global_T_frame.inverse());
    }
  }
  
  // Some heuristics to get a reasonable up direction, does not always work.
  if (trajectory_filename == string("groundtruth.txt")) {
    // Up direction for TUM RGB-D datasets with groundtruth poses.
    render_window->SetUpDirection(Vec3f(0, 0, 1));
  } else {
    // Load the up direction from groundtruth.txt if it exists.
    RGBDVideo<Vec3u8, u16> rgbd_video_temp;
    if (boost::filesystem::exists(boost::filesystem::path(dataset_folder_path) / "groundtruth.txt") &&
        ReadTUMRGBDDatasetAssociatedAndCalibrated(dataset_folder_path.c_str(), "groundtruth.txt", &rgbd_video_temp)) {
      render_window->SetUpDirection(
          rgbd_video.depth_frame_mutable(0)->frame_T_global().rotationMatrix().transpose() * rgbd_video_temp.depth_frame_mutable(0)->frame_T_global().rotationMatrix() * Vec3f(0, 0, 1));
    } else {
      // Set the up direction of the first frame as the global up direction.
      render_window->SetUpDirection(rgbd_video.depth_frame_mutable(0)->frame_T_global().rotationMatrix().transpose() * Vec3f(0, 1, 0));
    }
  }
  render_window->CenterViewOn(rgbd_video.depth_frame_mutable(0)->global_T_frame() * Vec3f(0, 0, 2));
  
  // Check that the RGB-D dataset uses the same intrinsics for color and depth.
  if (!AreCamerasEqual(*rgbd_video.color_camera(), *rgbd_video.depth_camera())) {
    LOG(FATAL) << "The color and depth camera of the RGB-D video must be equal.";
  }
  
  // If end_frame is non-zero, remove all frames which would extend beyond
  // this length.
  if (end_frame > 0 &&
      rgbd_video.color_frames_mutable()->size() > static_cast<usize>(end_frame)) {
    rgbd_video.color_frames_mutable()->resize(end_frame);
    rgbd_video.depth_frames_mutable()->resize(end_frame);
  }
  
  // Handle keyframe recording or playback.
  std::ofstream keyframes_write_file;
  unique_ptr<UniformCRSpline<FloatForSpline>> offset_x_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> offset_y_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> offset_z_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> radius_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> theta_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> phi_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> max_depth_spline;
  std::vector<float> spline_frame_indices;
  if (!record_keyframes_path.empty()) {
    keyframes_write_file.open(record_keyframes_path, std::ios::out);
  } else if (!playback_keyframes_path.empty()) {
    std::ifstream keyframes_read_file;
    keyframes_read_file.open(playback_keyframes_path, std::ios::in);
    if (!keyframes_read_file) {
      LOG(FATAL) << "Cannot open " << playback_keyframes_path;
    }
    
    std::vector<FloatForSpline> offset_x_spline_points;
    std::vector<FloatForSpline> offset_y_spline_points;
    std::vector<FloatForSpline> offset_z_spline_points;
    std::vector<FloatForSpline> radius_spline_points;
    std::vector<FloatForSpline> theta_spline_points;
    std::vector<FloatForSpline> phi_spline_points;
    std::vector<FloatForSpline> max_depth_spline_points;
    
    while (!keyframes_read_file.eof() && !keyframes_read_file.bad()) {
      std::string line;
      std::getline(keyframes_read_file, line);
      if (line.size() == 0 || line[0] == '#') {
        continue;
      }
      
      std::istringstream line_stream(line);
      std::string word;
      usize frame_index;
      Vec3f camera_free_orbit_offset;
      float camera_free_orbit_radius;
      float camera_free_orbit_theta;
      float camera_free_orbit_phi;
      float max_depth;
      line_stream >> word >> frame_index >> camera_free_orbit_offset.x() >>
                     camera_free_orbit_offset.y() >>
                     camera_free_orbit_offset.z() >>
                     camera_free_orbit_radius >> camera_free_orbit_theta >>
                     camera_free_orbit_phi >> max_depth;
      CHECK_EQ(word, "keyframe");
      spline_frame_indices.push_back(frame_index);
      offset_x_spline_points.push_back(FloatForSpline(camera_free_orbit_offset.x()));
      offset_y_spline_points.push_back(FloatForSpline(camera_free_orbit_offset.y()));
      offset_z_spline_points.push_back(FloatForSpline(camera_free_orbit_offset.z()));
      
      radius_spline_points.push_back(FloatForSpline(camera_free_orbit_radius));
      theta_spline_points.push_back(FloatForSpline(camera_free_orbit_theta));
      phi_spline_points.push_back(FloatForSpline(camera_free_orbit_phi));
      max_depth_spline_points.push_back(FloatForSpline(max_depth));
    }
    keyframes_read_file.close();
    
    offset_x_spline.reset(new UniformCRSpline<FloatForSpline>(offset_x_spline_points));
    offset_y_spline.reset(new UniformCRSpline<FloatForSpline>(offset_y_spline_points));
    offset_z_spline.reset(new UniformCRSpline<FloatForSpline>(offset_z_spline_points));
    radius_spline.reset(new UniformCRSpline<FloatForSpline>(radius_spline_points));
    theta_spline.reset(new UniformCRSpline<FloatForSpline>(theta_spline_points));
    phi_spline.reset(new UniformCRSpline<FloatForSpline>(phi_spline_points));
    max_depth_spline.reset(new UniformCRSpline<FloatForSpline>(max_depth_spline_points));
  }
  
  // Allocate image displays.
  shared_ptr<ImageDisplay> image_display(new ImageDisplay());
  shared_ptr<ImageDisplay> depth_display(new ImageDisplay());
  shared_ptr<ImageDisplay> downscaled_depth_display(new ImageDisplay());
  
  // Get potentially scaled depth camera as pinhole camera, determine input size.
  const Camera& generic_depth_camera = *rgbd_video.depth_camera();
  unique_ptr<Camera> scaled_camera(generic_depth_camera.Scaled(1.0f / powf(2, pyramid_level)));
  
  CHECK_EQ(scaled_camera->type_int(), static_cast<int>(Camera::Type::kPinholeCamera4f));
  const PinholeCamera4f& depth_camera = reinterpret_cast<const PinholeCamera4f&>(*scaled_camera);
  
  int width = depth_camera.width();
  int height = depth_camera.height();
  
  // Initialize CUDA streams.
  cudaStream_t stream;
  cudaStream_t upload_stream;
  cudaStreamCreate(&stream);
  cudaStreamCreate(&upload_stream);
  
  // Initialize CUDA events.
  cudaEvent_t depth_image_upload_pre_event;
  cudaEvent_t depth_image_upload_post_event;
  cudaEvent_t color_image_upload_pre_event;
  cudaEvent_t color_image_upload_post_event;
  cudaEvent_t frame_start_event;
  cudaEvent_t bilateral_filtering_post_event;
  cudaEvent_t outlier_filtering_post_event;
  cudaEvent_t depth_erosion_post_event;
  cudaEvent_t normal_computation_post_event;
  cudaEvent_t preprocessing_end_event;
  cudaEvent_t frame_end_event;
  cudaEvent_t surfel_transfer_start_event;
  cudaEvent_t surfel_transfer_end_event;
  
  cudaEvent_t upload_finished_event;
  
  cudaEventCreate(&depth_image_upload_pre_event);
  cudaEventCreate(&depth_image_upload_post_event);
  cudaEventCreate(&color_image_upload_pre_event);
  cudaEventCreate(&color_image_upload_post_event);
  cudaEventCreate(&frame_start_event);
  cudaEventCreate(&bilateral_filtering_post_event);
  cudaEventCreate(&outlier_filtering_post_event);
  cudaEventCreate(&depth_erosion_post_event);
  cudaEventCreate(&normal_computation_post_event);
  cudaEventCreate(&preprocessing_end_event);
  cudaEventCreate(&frame_end_event);
  cudaEventCreate(&surfel_transfer_start_event);
  cudaEventCreate(&surfel_transfer_end_event);
  
  cudaEventCreate(&upload_finished_event);
  
  // Allocate CUDA buffers.
  unordered_map<int, u16*> frame_index_to_depth_buffer_pagelocked;
  unordered_map<int, CUDABufferPtr<u16>> frame_index_to_depth_buffer;
  CUDABuffer<u16> filtered_depth_buffer_A(height, width);
  CUDABuffer<u16> filtered_depth_buffer_B(height, width);
  
  CUDABuffer<float2> normals_buffer(height, width);
  CUDABuffer<float> radius_buffer(height, width);
  
  Vec3u8* color_buffer_pagelocked;
  Vec3u8* next_color_buffer_pagelocked;
  cudaHostAlloc(reinterpret_cast<void**>(&color_buffer_pagelocked), height * width * sizeof(Vec3u8), cudaHostAllocWriteCombined);
  cudaHostAlloc(reinterpret_cast<void**>(&next_color_buffer_pagelocked), height * width * sizeof(Vec3u8), cudaHostAllocWriteCombined);
  shared_ptr<CUDABuffer<Vec3u8>> color_buffer(new CUDABuffer<Vec3u8>(height, width));
  shared_ptr<CUDABuffer<Vec3u8>> next_color_buffer(new CUDABuffer<Vec3u8>(height, width));
  
  std::vector<u16*> depth_buffers_pagelocked_cache;
  std::vector<CUDABufferPtr<u16>> depth_buffers_cache;
  
  // Initialize CUDA-OpenGL interoperation.
  OpenGLContext opengl_context;
  cudaGraphicsResource_t vertex_buffer_resource = nullptr;
  cudaGraphicsResource_t neighbor_index_buffer_resource = nullptr;
  cudaGraphicsResource_t normal_vertex_buffer_resource = nullptr;
  render_window->InitializeForCUDAInterop(
      max_surfel_count,
      &vertex_buffer_resource,
      &opengl_context,
      *scaled_camera,
      debug_neighbor_rendering,
      debug_normal_rendering,
      &neighbor_index_buffer_resource,
      &normal_vertex_buffer_resource);
  OpenGLContext no_opengl_context;
  SwitchOpenGLContext(opengl_context, &no_opengl_context);
  
  // Allocate reconstruction objects.
  CUDASurfelReconstruction reconstruction(
      max_surfel_count, depth_camera, vertex_buffer_resource,
      neighbor_index_buffer_resource, normal_vertex_buffer_resource, render_window);
  CUDASurfelsCPU cuda_surfels_cpu_buffers(max_surfel_count);
  SurfelMeshing surfel_meshing(
      max_surfels_per_node,
      max_angle_between_normals,
      min_triangle_angle,
      max_triangle_angle,
      max_neighbor_search_range_increase_factor,
      long_edge_tolerance_factor,
      regularization_frame_window_size,
      render_window);
  
  // Start background thread if using asynchronous meshing.
  unique_ptr<AsynchronousMeshing> triangulation_thread;
  if (asynchronous_triangulation) {
    triangulation_thread.reset(new AsynchronousMeshing(
        &surfel_meshing,
        &cuda_surfels_cpu_buffers,
        !timings_log_path.empty(),
        render_window));
  }
  
  // Show memory usage of GPU
  size_t free_bytes;
  size_t total_bytes;
  CUDA_CHECKED_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t used_bytes = total_bytes - free_bytes;
  
  constexpr double kBytesToMiB = 1.0 / (1024.0 * 1024.0);
  LOG(INFO) << "GPU memory usage after initialization: used = " <<
               kBytesToMiB * used_bytes << " MiB, free = " <<
               kBytesToMiB * free_bytes << " MiB, total = " <<
               kBytesToMiB * total_bytes << " MiB\n";
  
  
  // ### Main loop ###
  
  u32 latest_mesh_frame_index = 0;
  u32 latest_mesh_surfel_count = 0;
  usize latest_mesh_triangle_count = 0;
  bool triangulation_in_progress = false;
  
  ostringstream timings_log;
  ostringstream meshing_timings_log;
  
  constexpr int kStatsLogInterval = 200;
  
  bool quit = false;
  for (usize frame_index = start_frame; frame_index < rgbd_video.frame_count() - outlier_filtering_frame_count / 2 && !quit; ++ frame_index) {
    Timer frame_rate_timer("");  // "Frame rate timer (with I/O!)"
    
    
    // ### Input data loading ###
    
    // Since we do not want to measure the time for disk I/O, pre-load the
    // new images for this frame from disk here before starting the frame timer.
    if (frame_index + outlier_filtering_frame_count / 2 + 1 < rgbd_video.frame_count()) {
      rgbd_video.depth_frame_mutable(frame_index + outlier_filtering_frame_count / 2 + 1)->GetImage();
    }
    if (frame_index + 1 < rgbd_video.frame_count()) {
      rgbd_video.color_frame_mutable(frame_index + 1)->GetImage();
    }
    
    ConditionalTimer complete_frame_timer("[Integration frame - measured on CPU]");
    
    cudaEventRecord(upload_finished_event, upload_stream);
    
    // Upload all frames up to (frame_index + outlier_filtering_frame_count / 2) to the GPU.
    for (usize test_frame_index = frame_index;
         test_frame_index <= std::min(rgbd_video.frame_count() - 1, frame_index + outlier_filtering_frame_count / 2 + 1);
         ++ test_frame_index) {
      if (frame_index_to_depth_buffer.count(test_frame_index)) {
        continue;
      }
      
      u16** pagelocked_ptr = &frame_index_to_depth_buffer_pagelocked[test_frame_index];
      CUDABufferPtr<u16>* buffer_ptr = &frame_index_to_depth_buffer[test_frame_index];
      
      if (depth_buffers_cache.empty()) {
        cudaHostAlloc(reinterpret_cast<void**>(pagelocked_ptr), height * width * sizeof(u16), cudaHostAllocWriteCombined);
        
        buffer_ptr->reset(new CUDABuffer<u16>(height, width));
      } else {
        *pagelocked_ptr = depth_buffers_pagelocked_cache.back();
        depth_buffers_pagelocked_cache.pop_back();
        
        *buffer_ptr = depth_buffers_cache.back();
        depth_buffers_cache.pop_back();
      }
      
      // Perform median filtering and densification.
      // TODO: Do this on the GPU for better performance.
      const Image<u16>* depth_map = rgbd_video.depth_frame_mutable(test_frame_index)->GetImage().get();
      Image<u16> temp_depth_map;
      Image<u16> temp_depth_map_2;
      for (int iteration = 0; iteration < median_filter_and_densify_iterations; ++ iteration) {
        Image<u16>* target_depth_map = (depth_map == &temp_depth_map) ? &temp_depth_map_2 : &temp_depth_map;
        
        target_depth_map->SetSize(depth_map->size());
        MedianFilterAndDensifyDepthMap(*depth_map, target_depth_map);
        
        depth_map = target_depth_map;
      }
      
      if (pyramid_level == 0) {
        memcpy(*pagelocked_ptr,
               depth_map->data(),
               height * width * sizeof(u16));
      } else {
        if (median_filter_and_densify_iterations > 0) {
          LOG(ERROR) << "Simultaneous downscaling and median filtering of depth maps is not implemented.";
          return EXIT_FAILURE;
        }
        
        Image<u16> downscaled_image(width, height);
        rgbd_video.depth_frame_mutable(test_frame_index)->GetImage()->DownscaleUsingMedianWhileExcluding(0, width, height, &downscaled_image);
        
        // DEBUG: Show downsampled image.
        if (debug_depth_preprocessing) {
          downscaled_depth_display->Update(
              downscaled_image, "downscaled depth", static_cast<u16>(0),
              static_cast<u16>(depth_scaling * max_depth));
        }
        
        memcpy(*pagelocked_ptr,
               downscaled_image.data(),
               height * width * sizeof(u16));
      }
      cudaEventRecord(depth_image_upload_pre_event, upload_stream);
      (*buffer_ptr)->UploadAsync(upload_stream, *pagelocked_ptr);
      cudaEventRecord(depth_image_upload_post_event, upload_stream);
    }
    
    // Swap color image pointers and upload the next color frame to the GPU.
    std::swap(next_color_buffer, color_buffer);
    std::swap(next_color_buffer_pagelocked, color_buffer_pagelocked);
    if (pyramid_level == 0) {
      memcpy(next_color_buffer_pagelocked,
             rgbd_video.color_frame_mutable(frame_index + 1)->GetImage()->data(),
             width * height * sizeof(Vec3u8));
    } else {
      memcpy(next_color_buffer_pagelocked,
             ImagePyramid(rgbd_video.color_frame_mutable(frame_index + 1).get(), pyramid_level).GetOrComputeResult()->data(),
             width * height * sizeof(Vec3u8));
    }
    cudaEventRecord(color_image_upload_pre_event, upload_stream);
    next_color_buffer->UploadAsync(upload_stream, next_color_buffer_pagelocked);
    cudaEventRecord(color_image_upload_post_event, upload_stream);
    
    // If not enough neighboring frames are available for outlier filtering, go to the next frame.
    if (frame_index < static_cast<usize>(start_frame + outlier_filtering_frame_count / 2) ||
        frame_index >= rgbd_video.frame_count() - outlier_filtering_frame_count / 2) {
      frame_rate_timer.Stop(false);
      complete_frame_timer.Stop(false);
      continue;
    }
    
    // In the processing stream, wait for this frame's buffers to finish uploading in the upload stream.
    cudaStreamWaitEvent(stream, upload_finished_event, 0);
    
    
    // ### Depth pre-processing ###
    
    // Get and display input images.
    ImageFramePtr<Vec3u8, SE3f> color_frame = rgbd_video.color_frame_mutable(frame_index);
    ImageFramePtr<u16, SE3f> input_depth_frame = rgbd_video.depth_frame_mutable(frame_index);
    
    if (show_input_images) {
      image_display->Update(*color_frame->GetImage(), "image");
      depth_display->Update(*input_depth_frame->GetImage(), "depth",
                            static_cast<u16>(0), static_cast<u16>(depth_scaling * max_depth));
    }
    
    cudaEventRecord(frame_start_event, stream);
    
    CUDABufferPtr<u16> depth_buffer = frame_index_to_depth_buffer.at(frame_index);
    
    // Bilateral filtering and depth cutoff.
    BilateralFilteringAndDepthCutoffCUDA(
        stream,
        bilateral_filter_sigma_xy,
        bilateral_filter_sigma_depth_factor,
        /*value_to_ignore*/ 0,
        bilateral_filter_radius_factor,
        depth_scaling * max_depth,
        depth_valid_region_radius,
        depth_buffer->ToCUDA(),
        &filtered_depth_buffer_A.ToCUDA());
    cudaEventRecord(bilateral_filtering_post_event, stream);
    
    // DEBUG: Show bilateral filtering result.
    if (debug_depth_preprocessing) {
      Image<u16> filtered_depth(width, height);
      filtered_depth_buffer_A.DownloadAsync(stream, &filtered_depth);
      cudaStreamSynchronize(stream);
      static shared_ptr<ImageDisplay> filtered_depth_display(new ImageDisplay());
      filtered_depth_display->Update(filtered_depth, "CUDA bilateral filtered and cutoff depth",
                                     static_cast<u16>(0), static_cast<u16>(depth_scaling * max_depth));
    }
    
    // Depth outlier filtering.
    // Scale the poses to match the depth scaling. This is faster than scaling the depths of all pixels to match the poses.
    SE3f input_depth_frame_scaled_frame_T_global = input_depth_frame->frame_T_global();
    input_depth_frame_scaled_frame_T_global.translation() = depth_scaling * input_depth_frame_scaled_frame_T_global.translation();
    
    const CUDABuffer_<u16>* other_depths[outlier_filtering_frame_count];
    SE3f global_TR_others[outlier_filtering_frame_count];
    CUDAMatrix3x4 others_TR_reference[outlier_filtering_frame_count];
    for (int i = 0; i < outlier_filtering_frame_count / 2; ++ i) {
      int offset = i + 1;
      
      other_depths[i] = &frame_index_to_depth_buffer.at(frame_index - offset)->ToCUDA();
      global_TR_others[i] = rgbd_video.depth_frame_mutable(frame_index - offset)->global_T_frame();
      global_TR_others[i].translation() = depth_scaling * global_TR_others[i].translation();
      others_TR_reference[i] = CUDAMatrix3x4((input_depth_frame_scaled_frame_T_global * global_TR_others[i]).inverse().matrix3x4());
      
      int k = outlier_filtering_frame_count / 2 + i;
      other_depths[k] = &frame_index_to_depth_buffer.at(frame_index + offset)->ToCUDA();
      global_TR_others[k] = rgbd_video.depth_frame_mutable(frame_index + offset)->global_T_frame();
      global_TR_others[k].translation() = depth_scaling * global_TR_others[k].translation();
      others_TR_reference[k] = CUDAMatrix3x4((input_depth_frame_scaled_frame_T_global * global_TR_others[k]).inverse().matrix3x4());
    }
    
    if (outlier_filtering_required_inliers == -1 ||
        outlier_filtering_required_inliers == outlier_filtering_frame_count) {
      // Use a macro to pre-compile several versions of the template function.
      #define CALL_OUTLIER_FUSION(other_frame_count) \
          OutlierDepthMapFusionCUDA<other_frame_count + 1, u16>( \
              stream, \
              outlier_filtering_depth_tolerance_factor, \
              filtered_depth_buffer_A.ToCUDA(), \
              depth_camera.parameters()[0], \
              depth_camera.parameters()[1], \
              depth_camera.parameters()[2], \
              depth_camera.parameters()[3], \
              other_depths, \
              others_TR_reference, \
              &filtered_depth_buffer_B.ToCUDA())
      if (outlier_filtering_frame_count == 2) {
        CALL_OUTLIER_FUSION(2);
      } else if (outlier_filtering_frame_count == 4) {
        CALL_OUTLIER_FUSION(4);
      } else if (outlier_filtering_frame_count == 6) {
        CALL_OUTLIER_FUSION(6);
      } else if (outlier_filtering_frame_count == 8) {
        CALL_OUTLIER_FUSION(8);
      } else {
        LOG(FATAL) << "Unsupported value for outlier_filtering_frame_count: " << outlier_filtering_frame_count;
      }
      #undef CALL_OUTLIER_FUSION
    } else {
      // Use a macro to pre-compile several versions of the template function.
      #define CALL_OUTLIER_FUSION(other_frame_count) \
          OutlierDepthMapFusionCUDA<other_frame_count + 1, u16>( \
              stream, \
              outlier_filtering_required_inliers, \
              outlier_filtering_depth_tolerance_factor, \
              filtered_depth_buffer_A.ToCUDA(), \
              depth_camera.parameters()[0], \
              depth_camera.parameters()[1], \
              depth_camera.parameters()[2], \
              depth_camera.parameters()[3], \
              other_depths, \
              others_TR_reference, \
              &filtered_depth_buffer_B.ToCUDA())
      if (outlier_filtering_frame_count == 2) {
        CALL_OUTLIER_FUSION(2);
      } else if (outlier_filtering_frame_count == 4) {
        CALL_OUTLIER_FUSION(4);
      } else if (outlier_filtering_frame_count == 6) {
        CALL_OUTLIER_FUSION(6);
      } else if (outlier_filtering_frame_count == 8) {
        CALL_OUTLIER_FUSION(8);
      } else {
        LOG(FATAL) << "Unsupported value for outlier_filtering_frame_count: " << outlier_filtering_frame_count;
      }
      #undef CALL_OUTLIER_FUSION
    }
    cudaEventRecord(outlier_filtering_post_event, stream);
    
    // DEBUG: Show outlier filtering result.
    if (debug_depth_preprocessing) {
      Image<u16> filtered_depth(width, height);
      filtered_depth_buffer_B.DownloadAsync(stream, &filtered_depth);
      cudaStreamSynchronize(stream);
      static shared_ptr<ImageDisplay> filtered_depth_display(new ImageDisplay());
      filtered_depth_display->Update(filtered_depth, "CUDA outlier filtered depth",
                                    static_cast<u16>(0), static_cast<u16>(depth_scaling * max_depth));
    }
    
    // Depth map erosion.
    if (depth_erosion_radius > 0) {
      ErodeDepthMapCUDA(
          stream,
          depth_erosion_radius,
          filtered_depth_buffer_B.ToCUDA(),
          &filtered_depth_buffer_A.ToCUDA());
    } else {
      CopyWithoutBorderCUDA(
          stream,
          filtered_depth_buffer_B.ToCUDA(),
          &filtered_depth_buffer_A.ToCUDA());
    }
    
    cudaEventRecord(depth_erosion_post_event, stream);
    
    // DEBUG: Show erosion result.
    if (debug_depth_preprocessing) {
      Image<u16> filtered_depth(width, height);
      filtered_depth_buffer_A.DownloadAsync(stream, &filtered_depth);
      cudaStreamSynchronize(stream);
      static shared_ptr<ImageDisplay> filtered_depth_display(new ImageDisplay());
      filtered_depth_display->Update(filtered_depth, "CUDA eroded depth",
                                     static_cast<u16>(0), static_cast<u16>(depth_scaling * max_depth));
    }
    
    ComputeNormalsAndDropBadPixelsCUDA(
        stream,
        observation_angle_threshold_deg,
        depth_scaling,
        depth_camera.parameters()[0],
        depth_camera.parameters()[1],
        depth_camera.parameters()[2],
        depth_camera.parameters()[3],
        filtered_depth_buffer_A.ToCUDA(),
        &filtered_depth_buffer_B.ToCUDA(),
        &normals_buffer.ToCUDA());
    
    cudaEventRecord(normal_computation_post_event, stream);
    
    // DEBUG: Show current depth map result.
    if (debug_depth_preprocessing) {
      Image<u16> filtered_depth(width, height);
      filtered_depth_buffer_B.DownloadAsync(stream, &filtered_depth);
      cudaStreamSynchronize(stream);
      static shared_ptr<ImageDisplay> filtered_depth_display(new ImageDisplay());
      filtered_depth_display->Update(filtered_depth, "CUDA bad normal dropped depth",
                                      static_cast<u16>(0), static_cast<u16>(depth_scaling * max_depth));
    }
    
    cudaEventRecord(preprocessing_end_event, stream);
    
    ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
        stream,
        point_radius_extension_factor,
        point_radius_clamp_factor,
        depth_scaling,
        depth_camera.parameters()[0],
        depth_camera.parameters()[1],
        depth_camera.parameters()[2],
        depth_camera.parameters()[3],
        filtered_depth_buffer_B.ToCUDA(),
        &radius_buffer.ToCUDA(),
        &filtered_depth_buffer_A.ToCUDA());
    
    
    // ### Loop closures ###
    
    // Perform surfel deformation if needed.
    // NOTE: This component has been removed to avoid license issues.
    // if (loop_closure) {
    //   // Deform surfels ...
    // }
    
    
    // ### Surfel reconstruction ###
    
    reconstruction.Integrate(
        stream,
        frame_index,
        depth_scaling,
        &filtered_depth_buffer_A,
        normals_buffer,
        radius_buffer,
        *color_buffer,
        rgbd_video.depth_frame_mutable(frame_index)->global_T_frame(),
        sensor_noise_factor,
        max_surfel_confidence,
        regularizer_weight,
        regularization_frame_window_size,
        do_blending,
        measurement_blending_radius,
        regularization_iterations_per_integration_iteration,
        radius_factor_for_regularization_neighbors,
        normal_compatibility_threshold_deg,
        surfel_integration_active_window_size);
    
    cudaEventRecord(frame_end_event, stream);
    
    
    // ### Surfel meshing handling ###
    
    // Transfer surfels to the CPU if no meshing is in progress,
    // if we expect that the next iteration will start very soon,
    // and for the last frame if the final result is needed.
    bool did_surfel_transfer = false;
    
    bool no_meshing_in_progress =
        !asynchronous_triangulation || !triangulation_in_progress;
    bool next_meshing_expected_soon = false;
    if (!no_meshing_in_progress) {
      double time_since_last_meshing_start =
          1e-9 * chrono::duration<double, nano>(
              chrono::steady_clock::now() -
              triangulation_thread->latest_triangulation_start_time()).count();
      next_meshing_expected_soon =
          time_since_last_meshing_start >
          triangulation_thread->latest_triangulation_duration() - 0.05f;
    }
    bool final_result_required =
        show_result || !export_mesh_path.empty() || !export_point_cloud_path.empty();
    bool is_last_frame =
        frame_index == rgbd_video.frame_count() - outlier_filtering_frame_count / 2 - 1;
    
    if (no_meshing_in_progress ||
        next_meshing_expected_soon ||
        (final_result_required && is_last_frame)) {
      cudaEventRecord(surfel_transfer_start_event, stream);
      if (asynchronous_triangulation) {
        triangulation_thread->LockInputData();
      }
      cuda_surfels_cpu_buffers.LockWriteBuffers();
      
      reconstruction.TransferAllToCPU(
          stream,
          frame_index,
          &cuda_surfels_cpu_buffers);
      
      cudaEventRecord(surfel_transfer_end_event, stream);
      cudaStreamSynchronize(stream);
      
      // Notify the triangulation thread about new input data.
      // NOTE: It must be avoided to send this notification after the thread
      //       has already started working on the input (due to a previous
      //       notification), so do it while the write buffers are locked.
      //       Otherwise, the thread might later continue its
      //       next iteration before the write buffer was updated again,
      //       resulting in wrong data being used, in particular many surfels
      //       might be at (0, 0, 0).
      if (asynchronous_triangulation) {
        triangulation_thread->NotifyAboutNewInputSurfelsAlreadyLocked();
      }
      triangulation_in_progress = true;
      
      cuda_surfels_cpu_buffers.UnlockWriteBuffers();
      if (asynchronous_triangulation) {
        triangulation_thread->UnlockInputData();
      }
      did_surfel_transfer = true;
    }
    cudaStreamSynchronize(stream);
    complete_frame_timer.Stop();
    
    // Update the visualization if a new mesh is available.
    if (asynchronous_triangulation) {
      shared_ptr<Mesh3fCu8> output_mesh;
      
      if (final_result_required && is_last_frame) {
        // No need for efficiency here, use simple polling waiting
        LOG(INFO) << "Waiting for final mesh ...";
        while (!triangulation_thread->all_work_done()) {
          usleep(0);
        }
        triangulation_thread->RequestExitAndWaitForIt();
        LOG(INFO) << "Got final mesh";
      }
      
      // Get new mesh from the triangulation thread?
      u32 output_frame_index;
      u32 output_surfel_count;
      triangulation_thread->GetOutput(&output_frame_index, &output_surfel_count, &output_mesh);
      
      if (output_mesh) {
        // There is a new mesh.
        latest_mesh_frame_index = output_frame_index;
        latest_mesh_surfel_count = output_surfel_count;
        latest_mesh_triangle_count = output_mesh->triangles().size();
      }
      
      // Update visualization.
      unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
      reconstruction.UpdateVisualizationBuffers(
          stream,
          frame_index,
          latest_mesh_frame_index,
          latest_mesh_surfel_count,
          surfel_integration_active_window_size,
          visualize_last_update_timestamp,
          visualize_creation_timestamp,
          visualize_radii,
          visualize_surfel_normals);
      render_window->UpdateVisualizationCloudCUDA(reconstruction.surfels_size(), latest_mesh_surfel_count);
      if (output_mesh) {
        render_window->UpdateVisualizationMeshCUDA(output_mesh);
      }
      cudaStreamSynchronize(stream);
      render_mutex_lock.unlock();
      
      if (frame_index % kStatsLogInterval == 0) {
        LOG(INFO) << "[frame " << frame_index << "] #surfels: " << reconstruction.surfel_count() << ", #triangles (of latest mesh): " << latest_mesh_triangle_count;
      }
    } else {
      // Synchronous triangulation.
      cuda_surfels_cpu_buffers.WaitForLockAndSwapBuffers();
      surfel_meshing.IntegrateCUDABuffers(frame_index, cuda_surfels_cpu_buffers);
      
      if (full_meshing_every_frame) {
        double full_retriangulation_seconds = surfel_meshing.FullRetriangulation();
        
        if (!timings_log_path.empty()) {
          timings_log << "frame " << frame_index << std::endl;
          timings_log << "-full_meshing " << (1000 * full_retriangulation_seconds) << std::endl;
        }
      } else {
        ConditionalTimer check_remeshing_timer("CheckRemeshing()");
        surfel_meshing.CheckRemeshing();
        double remeshing_seconds = check_remeshing_timer.Stop();
        
        ConditionalTimer triangulate_timer("Triangulate()");
        surfel_meshing.Triangulate();
        double meshing_seconds = triangulate_timer.Stop();
        
        if (!timings_log_path.empty()) {
          timings_log << "frame " << frame_index << std::endl;
          timings_log << "-remeshing " << (1000 * remeshing_seconds) << std::endl;
          timings_log << "-meshing " << (1000 * meshing_seconds) << std::endl;
        }
      }
      
      // Update cloud and mesh in the display.
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
      unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
      reconstruction.UpdateVisualizationBuffers(
          stream,
          frame_index,
          frame_index,
          surfel_meshing.surfels().size(),
          surfel_integration_active_window_size,
          visualize_last_update_timestamp,
          visualize_creation_timestamp,
          visualize_radii,
          visualize_surfel_normals);
      render_window->UpdateVisualizationCloudAndMeshCUDA(reconstruction.surfel_count(), visualization_mesh);
      cudaStreamSynchronize(stream);
      render_mutex_lock.unlock();
      LOG(INFO) << "[frame " << frame_index << "] #surfels: " << reconstruction.surfel_count() << ", #triangles: " << visualization_mesh->triangles().size();
    }
    
    
    // ### Visualization camera pose handling ###
    
    SE3f global_T_frame = rgbd_video.depth_frame_mutable(frame_index + outlier_filtering_frame_count / 2)->global_T_frame();
    if (!playback_keyframes_path.empty()) {
      // Determine camera pose from spline-based keyframe playback.
      usize first_keyframe_index = spline_frame_indices.size() - 1;
      for (usize i = 1; i < spline_frame_indices.size(); ++ i) {
        if (spline_frame_indices[i] >= frame_index) {
          first_keyframe_index = i - 1;
          break;
        }
      }
      usize prev_frame_index = spline_frame_indices[first_keyframe_index];
      usize next_frame_index = spline_frame_indices[first_keyframe_index + 1];
      float t = -1 + first_keyframe_index + (frame_index - prev_frame_index) * 1.0f / (next_frame_index - prev_frame_index);
      
      Vec3f camera_free_orbit_offset;
      camera_free_orbit_offset.x() = offset_x_spline->getPosition(t);
      camera_free_orbit_offset.y() = offset_y_spline->getPosition(t);
      camera_free_orbit_offset.z() = offset_z_spline->getPosition(t);
      float camera_free_orbit_radius = radius_spline->getPosition(t);
      float camera_free_orbit_theta = theta_spline->getPosition(t);
      float camera_free_orbit_phi = phi_spline->getPosition(t);
      float camera_max_depth = max_depth_spline->getPosition(t);
      
      render_window->SetViewParameters(camera_free_orbit_offset, camera_free_orbit_radius, camera_free_orbit_theta, camera_free_orbit_phi, camera_max_depth, global_T_frame);
    } else if (follow_input_camera) {
      // Use camera pose of frame where all used image data is available.
      Vec3f eye = global_T_frame * Vec3f(0, 0, -0.25f);
      Vec3f look_at = global_T_frame * Vec3f(0, 0, 1.0f);
      Vec3f up = global_T_frame.rotationMatrix() * Vec3f(0, -1.0f, 0);
      
      Vec3f z = (look_at - eye).normalized();  // Forward
      Vec3f x = z.cross(up).normalized(); // Right
      Vec3f y = z.cross(x);
      
      render_window->SetView2(x, y, z, eye, global_T_frame);
    } else {
      // Do not set the visualization camera pose, only tell the input camera
      // pose to the visualization.
      render_window->SetCameraFrustumPose(global_T_frame);
    }
    
    // Create screenshot for video?
    if (create_video) {
      ostringstream frame_path;
      frame_path << "frame" << std::setw(6) << std::setfill('0') << frame_index << ".png";
      render_window->SaveScreenshot(frame_path.str().c_str());
    }
    
    // For debugging purposes only, notify the render window about the surfel_meshing.
    render_window->SetReconstructionForDebugging(&surfel_meshing);
    
    
    // ### Profiling ###
    
    float elapsed_milliseconds;
    float frame_time_milliseconds = 0;
    float preprocessing_milliseconds = 0;
    float surfel_transfer_milliseconds = 0;
    
    // Synchronize with latest event
    if (did_surfel_transfer) {
      cudaEventSynchronize(surfel_transfer_end_event);
    } else {
      cudaEventSynchronize(frame_end_event);
    }
    
    cudaEventSynchronize(depth_image_upload_post_event);
    cudaEventElapsedTime(&elapsed_milliseconds, depth_image_upload_pre_event, depth_image_upload_post_event);
    Timing::addTime(Timing::getHandle("Upload depth image"), 0.001 * elapsed_milliseconds);
    
    cudaEventSynchronize(color_image_upload_post_event);
    cudaEventElapsedTime(&elapsed_milliseconds, color_image_upload_pre_event, color_image_upload_post_event);
    Timing::addTime(Timing::getHandle("Upload color image"), 0.001 * elapsed_milliseconds);
    
    cudaEventElapsedTime(&elapsed_milliseconds, frame_start_event, bilateral_filtering_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Depth bilateral filtering"), 0.001 * elapsed_milliseconds);
    
    cudaEventElapsedTime(&elapsed_milliseconds, bilateral_filtering_post_event, outlier_filtering_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Depth outlier filtering"), 0.001 * elapsed_milliseconds);
    
    cudaEventElapsedTime(&elapsed_milliseconds, outlier_filtering_post_event, depth_erosion_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Depth erosion"), 0.001 * elapsed_milliseconds);
    
    cudaEventElapsedTime(&elapsed_milliseconds, depth_erosion_post_event, normal_computation_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Normal computation"), 0.001 * elapsed_milliseconds);
    
    cudaEventElapsedTime(&elapsed_milliseconds, normal_computation_post_event, preprocessing_end_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Radius computation"), 0.001 * elapsed_milliseconds);
    
    cudaEventElapsedTime(&elapsed_milliseconds, preprocessing_end_event, frame_end_event);
    frame_time_milliseconds += elapsed_milliseconds;
    Timing::addTime(Timing::getHandle("Integration"), 0.001 * elapsed_milliseconds);
    
    Timing::addTime(Timing::getHandle("[CUDA frame]"), 0.001 * frame_time_milliseconds);
    
    if (did_surfel_transfer) {
      cudaEventElapsedTime(&surfel_transfer_milliseconds, surfel_transfer_start_event, surfel_transfer_end_event);
      Timing::addTime(Timing::getHandle("Surfel transfer to CPU"), 0.001 * surfel_transfer_milliseconds);
    }
    
    float data_association;
    float surfel_merging;
    float measurement_blending;
    float integration;
    float neighbor_update;
    float new_surfel_creation;
    float regularization;
    reconstruction.GetTimings(
        &data_association,
        &surfel_merging,
        &measurement_blending,
        &integration,
        &neighbor_update,
        &new_surfel_creation,
        &regularization);
    Timing::addTime(Timing::getHandle("Integration - data_association"), 0.001 * data_association);
    Timing::addTime(Timing::getHandle("Integration - surfel_merging"), 0.001 * surfel_merging);
    Timing::addTime(Timing::getHandle("Integration - measurement_blending"), 0.001 * measurement_blending);
    Timing::addTime(Timing::getHandle("Integration - integration"), 0.001 * integration);
    Timing::addTime(Timing::getHandle("Integration - neighbor_update"), 0.001 * neighbor_update);
    Timing::addTime(Timing::getHandle("Integration - new_surfel_creation"), 0.001 * new_surfel_creation);
    Timing::addTime(Timing::getHandle("Integration - regularization"), 0.001 * regularization);
    
    if (frame_index % kStatsLogInterval == 0) {
      LOG(INFO) << Timing::print(kSortByTotal);
    }
    
    if (!timings_log_path.empty()) {
      timings_log << "frame " << frame_index << std::endl;
      timings_log << "-preprocessing " << preprocessing_milliseconds << std::endl;
      timings_log << "-data_association " << data_association << std::endl;
      timings_log << "-surfel_merging " << surfel_merging << std::endl;
      timings_log << "-measurement_blending " << measurement_blending << std::endl;
      timings_log << "-integration " << integration << std::endl;
      timings_log << "-neighbor_update " << neighbor_update << std::endl;
      timings_log << "-new_surfel_creation " << new_surfel_creation << std::endl;
      timings_log << "-regularization " << regularization << std::endl;
      if (did_surfel_transfer) {
        timings_log << "-surfel_transfer " << surfel_transfer_milliseconds << std::endl;
      }
      timings_log << "-surfel_count " << reconstruction.surfel_count() << std::endl;
    }
    
    
    // ### Handle key presses (in the terminal) ###
    
    if (step_by_step_playback || (show_result && is_last_frame)) {
      while (true) {
        int key = getch();
        
        if (key == 10) {
          // Return key.
          if (!(show_result && is_last_frame)) {
            break;
          }
        }
        
        if (key == 'q' || key == 'Q') {
          // Quit the program.
          quit = true;
          break;
        } else if (key == 'r' || key == 'R') {
          // Run (disable step-by-step playback).
          step_by_step_playback = !step_by_step_playback;
          break;
        } else if (key == 'a' || key =='A') {
          // Stronger regularization.
          regularizer_weight *= 1.1f;
          LOG(INFO) << "regularizer_weight: " << regularizer_weight;
        } else if (key == 'd' || key =='D') {
          // Perform a regularization iteratin.
          LOG(INFO) << "Regularization iteration ...";
          reconstruction.Regularize(
              stream, frame_index, regularizer_weight,
              radius_factor_for_regularization_neighbors,
              regularization_frame_window_size);
          
          unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
          reconstruction.UpdateVisualizationBuffers(
              stream,
              frame_index,
              asynchronous_triangulation ? latest_mesh_frame_index : frame_index,
              asynchronous_triangulation ? latest_mesh_surfel_count : surfel_meshing.surfels().size(),
              surfel_integration_active_window_size,
              visualize_last_update_timestamp,
              visualize_creation_timestamp,
              visualize_radii,
              visualize_surfel_normals);
          render_window->UpdateVisualizationCloudCUDA(
              reconstruction.surfels_size(),
              asynchronous_triangulation ? latest_mesh_surfel_count : 0);
          cudaStreamSynchronize(stream);
          render_mutex_lock.unlock();
        } else if (key == 's' || key =='S') {
          // Weaker regularization.
          regularizer_weight *= 1 / 1.1f;
          LOG(INFO) << "regularizer_weight: " << regularizer_weight;
        } else if (key == 't' || key == 'T') {
          // Full re-triangulation of all surfels.
          surfel_meshing.FullRetriangulation();
          
          shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
          surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
          render_window->UpdateVisualizationMesh(visualization_mesh);
          LOG(INFO) << "[frame " << frame_index << " full retriangulation] Triangle count: " << visualization_mesh->triangles().size();
        } else if (key == 'y' || key == 'Y') {
          // Try to triangulate the selected surfel in debug mode.
          LOG(INFO) << "Trying to triangulate surfel " << render_window->selected_surfel_index() << " ...";
          surfel_meshing.SetSurfelToRemesh(render_window->selected_surfel_index());
          surfel_meshing.Triangulate(/* force_debug */ true);
          
          shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
          surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
          render_window->UpdateVisualizationMesh(visualization_mesh);
        } else if (key == 'e' || key == 'E') {
          // Retriangulate the selected surfel in debug mode.
          const Surfel* surfel = &surfel_meshing.surfels().at(render_window->selected_surfel_index());
          LOG(INFO) << "Retriangulating surfel " << render_window->selected_surfel_index() << " (radius_squared: " << surfel->radius_squared() << ") ...";
          surfel_meshing.RemeshTrianglesAt(const_cast<Surfel*>(surfel), surfel->radius_squared());  // TODO: avoid const_cast
          surfel_meshing.Triangulate(/* force_debug */ true);
          
          shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
          surfel_meshing.ConvertToMesh3fCu8(visualization_mesh.get(), true);
          render_window->UpdateVisualizationMesh(visualization_mesh);
        } else if (key == 'p' || key == 'P') {
          // Save the mesh.
          SaveMeshAsOBJ(reconstruction, surfel_meshing, export_mesh_path, stream);
        } else if (key == 'k' || key == 'K') {
          // Record keyframe.
          Vec3f camera_free_orbit_offset;
          float camera_free_orbit_radius;
          float camera_free_orbit_theta;
          float camera_free_orbit_phi;
          render_window->GetCameraPoseParameters(
              &camera_free_orbit_offset,
              &camera_free_orbit_radius,
              &camera_free_orbit_theta,
              &camera_free_orbit_phi);
          
          keyframes_write_file << "keyframe " << frame_index
                               << " " << camera_free_orbit_offset.transpose()
                               << " " << camera_free_orbit_radius
                               << " " << camera_free_orbit_theta
                               << " " << camera_free_orbit_phi
                               << " " << 50  // TODO: output max_depth
                               << std::endl;
          keyframes_write_file.flush();
        }
      }
    }
    
    
    // ### End-of-frame handling ###
    
    // Release frames which are no longer needed.
    int last_frame_in_window = frame_index - outlier_filtering_frame_count / 2;
    if (last_frame_in_window >= 0) {
      rgbd_video.color_frame_mutable(last_frame_in_window)->ClearImageAndDerivedData();
      rgbd_video.depth_frame_mutable(last_frame_in_window)->ClearImageAndDerivedData();
      depth_buffers_pagelocked_cache.push_back(frame_index_to_depth_buffer_pagelocked.at(last_frame_in_window));
      frame_index_to_depth_buffer_pagelocked.erase(last_frame_in_window);
      depth_buffers_cache.push_back(frame_index_to_depth_buffer.at(last_frame_in_window));
      frame_index_to_depth_buffer.erase(last_frame_in_window);
    }
    
    // Restrict frame time.
    double min_frame_time = 1.0 / fps_restriction;
    static u32 times_target_fps_reached = 0;
    static u32 times_target_fps_not_reached = 0;
    
    double actual_frame_time = frame_rate_timer.Stop(false);
    if (actual_frame_time <= min_frame_time) {
      ++ times_target_fps_reached;
    } else {
      ++ times_target_fps_not_reached;
    }
    if (frame_index % kStatsLogInterval == 0) {
      LOG(INFO) << "Target FPS of " << fps_restriction << " for integration reached " << times_target_fps_reached << " times, failed " << times_target_fps_not_reached << " times";
    }
    
    if (actual_frame_time < min_frame_time) {
      constexpr float kSecondsToMicroSeconds = 1000 * 1000;
      usize microseconds = kSecondsToMicroSeconds * (min_frame_time - actual_frame_time);
      usleep(microseconds);
    }
  }  // End of main loop
  
  
  // ### Save results and cleanup ###
  
  if (asynchronous_triangulation && !(show_result || !export_mesh_path.empty() || !export_point_cloud_path.empty())) {
    triangulation_thread->RequestExitAndWaitForIt();
  }
  
  if (!timings_log_path.empty()) {
    FILE* file = fopen(timings_log_path.c_str(), "wb");
    string str = timings_log.str();
    fwrite(str.c_str(), 1, str.size(), file);
    fclose(file);
  }
  
  // Perform retriangulation at end?
  if (full_retriangulation_at_end) {
    surfel_meshing.FullRetriangulation();
  }
  
  // Save the final mesh.
  if (!export_mesh_path.empty()) {
    LOG(INFO) << "Saving the final mesh ...";
    SaveMeshAsOBJ(reconstruction, surfel_meshing, export_mesh_path, stream);
  }
  
  // Save the final point cloud.
  if (!export_point_cloud_path.empty()) {
    LOG(INFO) << "Saving the final point cloud ...";
    SavePointCloudAsPLY(reconstruction, surfel_meshing, export_point_cloud_path);
  }
  
  // Cleanup
  for (u16* ptr : depth_buffers_pagelocked_cache) {
    cudaFreeHost(ptr);
  }
  for (pair<int, u16*> item : frame_index_to_depth_buffer_pagelocked) {
    cudaFreeHost(item.second);
  }
  
  cudaFreeHost(color_buffer_pagelocked);
  cudaFreeHost(next_color_buffer_pagelocked);
  
  cudaEventDestroy(depth_image_upload_pre_event);
  cudaEventDestroy(depth_image_upload_post_event);
  cudaEventDestroy(color_image_upload_pre_event);
  cudaEventDestroy(color_image_upload_post_event);
  cudaEventDestroy(frame_start_event);
  cudaEventDestroy(bilateral_filtering_post_event);
  cudaEventDestroy(outlier_filtering_post_event);
  cudaEventDestroy(depth_erosion_post_event);
  cudaEventDestroy(normal_computation_post_event);
  cudaEventDestroy(preprocessing_end_event);
  cudaEventDestroy(frame_end_event);
  cudaEventDestroy(surfel_transfer_start_event);
  cudaEventDestroy(surfel_transfer_end_event);
  
  cudaEventDestroy(upload_finished_event);
  
  cudaStreamDestroy(stream);
  cudaStreamDestroy(upload_stream);
  
  
  // Print final timings.
  LOG(INFO) << Timing::print(kSortByTotal);
  
  return EXIT_SUCCESS;
}
