# SurfelMeshing #

![Screenshot](screenshot.jpg?raw=true)

## Overview ##

SurfelMeshing is an approach for real-time surfel-based mesh reconstruction from
RGB-D video, described in the following paper:

T. Schöps, T. Sattler, M. Pollefeys, "SurfelMeshing: Online Surfel-Based Mesh Reconstruction", ArXiv, 2018. \[link will be added soon\]

If you use the SurfelMeshing code for research, please cite this paper.

Compared to the version of the code used to run the experiments for the paper,
we removed the loop closure handling component for the open source version in this
repository. This is because this component was written with the help of
proprietary-licensed code, which we wanted to avoid. If you would like to add
your own loop closure handling, see the corresponding section below.

The repository contains the
[SurfelMeshing application](https://github.com/puzzlepaint/surfelmeshing/tree/master/applications/surfel_meshing)
and the library it is based on,
[libvis](https://github.com/puzzlepaint/surfelmeshing/tree/master/libvis).
The library is work-in-progress and it is not recommended to use it for other projects at this point.

The application and library code is licensed under the BSD license, but please
also notice the licenses of the included or externally used third-party components.


## Building ##

Building has been tested on Ubuntu 14.04 only. It is expected that later
versions of Ubuntu also work with little effort.

The following external dependencies are required (the versions in brackets are
known to work):

* Boost
* CUDA (8)
* Eigen
* GLEW
* GLog
* PCL (1.7)
* Qt (5.2.1)
* zlib

Notice that the versions of CUDA and Eigen must be compatible since some Eigen headers are included in
code compiled by CUDA's nvcc compiler. At the time of writing, this meant that for a recent CUDA version (9.1),
the development version of Eigen had to be used instead of the latest release. PCL also depends on Eigen and
thus it might be good to ensure that it uses the same version.

After obtaining all dependencies, the application can be built with CMake, for example as follows:
```
mkdir build_RelWithDebInfo
cd build_RelWithDebInfo
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j SurfelMeshing
```


## Running ##

The program supports datasets in the format of the
[TUM RGB-D benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset)
with two small additions:

* The original format does not specify the intrinsic camera calibration.
  SurfelMeshing thus additionally expects a file `calibration.txt` in the
  dataset directory, consisting of a single line of text structured as follows:
  ```
  fx fy cx cy
  ```
  These values specify the parameters for the pinhole projection (fx * x + cx, fy * y + cy).
  For example, `525.0 525.0 319.5 239.5` works reasonably well for the Kinect
  v1 (these values can be used for the TUM RGB-D benchmark datasets).
  The coordinate system convention for cx and cy is that the origin is at
  the center of the top-left pixel in the image.
* The [associate.py](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py)
  tool from the benchmark must be run as follows to associate
  the color and depth images:
  ```
  python associate.py rgb.txt depth.txt > associated.txt
  ```

Without specifying any optional parameters, SurfelMeshing can be run as follows:
```
./build_RelWithDebInfo/applications/surfel_meshing/SurfelMeshing <dataset> <trajectory>
```
Here, `<dataset>` is the path to the dataset folder, and `<trajectory>` is the
filename of the trajectory file within this folder (excluding the rest of the
file's path). The `groundtruth.txt` files provided with the TUM RGB-D benchmark can
be used for testing (but keep in mind that for some datasets, parts of the ground truth trajectory
are missing, which might lead to issues).

For example, to run the reconstruction with the ground truth trajectory (and use
a freely movable camera):
```
./build_RelWithDebInfo/applications/surfel_meshing/SurfelMeshing /path/to/some_tum_rgbd_dataset groundtruth.txt --follow_input_camera false
```

Notice that the first time the program runs on a dataset, the performance is
usually limited by the time it takes to read the image files from the hard disk
(unless the dataset is on an SSD, or is already cached because the files were written recently).
Subsequent runs should be faster as long as the files remain cached.

In case you encounter issues with insufficient GPU memory, try decreasing the
maximum surfel count with the `--max_surfel_count` option (default: 20000000).
However, the program will abort once this surfel count is exceeded.



## 3D window controls ##

By default, the 3D window view is fixed to the input camera's trajectory. Use
`--follow_input_camera false` to get a freely movable camera, which can be
controlled as follows:

* Left click and drag: rotate camera around look-at point
* Mouse wheel: zoom
* Middle mouse button click and drag: move look-at point. Alternatively, you can hold the left and right mouse button at the same time, or drag with the left mouse button and the m key pressed.
* C key: copy camera pose (can be stored as text)
* V key: paste camera pose

Furthermore, the following keys can be used to alter the 3D display:

* W key: toggle wireframe rendering (notice that you need to have a close-up look at the surfaces to see the wireframe since the mesh is extremely dense)
* S key: toggle splat rendering for all surfels
* H key: toggle mesh rendering
* D, I keys: decrease or increase maximum render distance



## Terminal controls ##

These controls can be used in the terminal after the dataset processing finished or if the `--step_by_step_playback` option is used
(i.e., when the reconstruction is not running).

* q: quit program
* r: run (disable step-by-step playback)
* a: increase regularization strength
* s: decrease regularization strength
* d: perform a regularization iteration
* t: perform full re-triangulation of all surfels
* p: Save the current mesh (the filename must be specified with the `--export_mesh` option); the mesh will also be saved when the program exits
* k: Record a keyframe (see `--record_keyframes` and `--playback_keyframes` options)



## Adding loop closure handling ##

As mentioned above, the loop closure component was removed from this version of the code to
avoid license issues. If you would like to add your own loop closure handling,
you could start by adding the call to it at main.cc:1162 (line "// ### Loop closures ###").
You would need to write a CUDA kernel analogously to the existing ones in cuda_surfel_reconstruction.cu
and modify the following surfel attributes:

```c++
surfels(kSurfelX, surfel_index) += position_offset.x;
surfels(kSurfelY, surfel_index) += position_offset.y;
surfels(kSurfelZ, surfel_index) += position_offset.z;
surfels(kSurfelSmoothX, surfel_index) += position_offset.x;
surfels(kSurfelSmoothY, surfel_index) += position_offset.y;
surfels(kSurfelSmoothZ, surfel_index) += position_offset.z;
surfels(kSurfelNormalX, surfel_index) = new_normal.x;
surfels(kSurfelNormalY, surfel_index) = new_normal.y;
surfels(kSurfelNormalZ, surfel_index) = new_normal.z;
```

The timestamp for old surfels which shall be activated for measurement integration again can be updated as follows:

```c++
*reinterpret_cast<u32*>(&surfels(kSurfelLastUpdateStamp, surfel_index)) = frame_index;
```



## Program arguments ##

A list of optional program arguments follows, grouped by category:

#### Dataset playback ####

* `--depth_scaling` (default 5000): Input depth scaling: input_depth = depth_scaling * depth_in_meters. The default is for TUM RGB-D benchmark datasets.
* `--start_frame` (default 0): First frame of the video to process.
* `--end_frame` (default: 2147483647): If the video is longer, processing stops after end_frame.
* `--pyramid_level` (default: 0): Specify the scale-space pyramid level to use. 0 uses the original sized images, 1 uses half the original resolution, etc.
* `--restrict_fps_to` (default: 30): Restrict the frames per second to at most the given number.
* `--step_by_step_playback`: Play back video frames step-by-step (do a step by pressing the Return key in the terminal).
* `--invert_quaternions`: Invert the quaternions loaded from the poses file.

#### Surfel reconstruction ####

* `--max_surfel_count` (default: 20000000): Maximum number of surfels. Determines the GPU memory requirements.
* `--sensor_noise_factor` (default: 0.05): Sensor noise range extent as "factor times the measured depth". The real measurement is assumed to be in [(1 - sensor_noise_factor) * depth, (1 + sensor_noise_factor) * depth].
* `--max_surfel_confidence` (default: 5): Maximum value for the surfel confidence. Higher values enable more denoising, lower values enable faster adaptation to changes.
* `--regularizer_weight` (default: 10): Weight for the regularization term (w_{\text{reg}} in the paper).
* `--normal_compatibility_threshold_deg` (default: 40): Angle threshold (in degrees) for considering a measurement normal and a surfel normal to be compatible.
* `--regularization_frame_window_size` (default: 30): Number of frames for which the regularization of a surfel is continued after it goes out of view.
* `--disable_blending`: Disable observation boundary blending.
* `--measurement_blending_radius` (default: 12): Radius for measurement blending in pixels.
* `--regularization_iterations_per_integration_iteration` (default: 1): Number of regularization (gradient descent) iterations performed per depth integration iteration.
* `--radius_factor_for_regularization_neighbors` (default: 2): Factor on the surfel radius for how far regularization neighbors can be away from a surfel.
* `--surfel_integration_active_window_size` (default: numeric_limits&lt;int>::max()): Number of frames which need to pass before a surfel becomes inactive. If there are no loop closures, set this to a value larger than the dataset frame count to disable surfel deactivation.

#### Meshing ####

* `--max_angle_between_normals_deg` (default: 90): Maximum angle between normals of surfels that are connected by triangulation.
* `--min_triangle_angle_deg` (default: 10): The meshing algorithm attempts to keep triangle angles larger than this.
* `--max_triangle_angle_deg` (default: 170): The meshing algorithm attempts to keep triangle angles smaller than this.
* `--max_neighbor_search_range_increase_factor` (default: 2): Maximum factor by which the surfel neighbor search range can be increased if the front neighbors are far away.
* `--long_edge_tolerance_factor` (default: 1.5): Tolerance factor over 'max_neighbor_search_range_increase_factor * surfel_radius' for deciding whether to remesh a triangle with long edges.
* `--synchronous_meshing`: Makes the meshing proceed synchronously to the surfel integration (instead of asynchronously).

#### Depth preprocessing ####

* `--max_depth` (default: 3): Maximum input depth in meters.
* `--depth_valid_region_radius` (default: 333): Radius of a circle (centered on the image center) with valid depth. Everything outside the circle is considered to be invalid and not used. This setting is useful to discard the highly biased depth values at the corners of Kinect v1 depth images which seem to come from bad calibration.
* `--observation_angle_threshold_deg` (default: 75): If the angle between the inverse observation direction and the measured surface normal is larger than this setting, the surface is discarded.
* `--depth_erosion_radius` (default: 2): Radius for depth map erosion (in [0, 3]). Useful to combat foreground fattening artifacts.
* `--median_filter_and_densify_iterations` (default: 0): Number of iterations of median filtering with hole filling. Disabled by default. Can be useful for noisy time-of-flight data.
* `--outlier_filtering_frame_count` (default: 8): Number of other depth frames to use for outlier filtering of a depth frame. Supported values: 2, 4, 6, 8. Should be reduced if using low-frequency input.
* `--outlier_filtering_required_inliers` (default: -1): Number of required inliers for accepting a depth value in outlier filtering. With the default value of -1, all other frames (outlier_filtering_frame_count) must be inliers.
* `--bilateral_filter_sigma_xy` (default: 3): sigma_xy for depth bilateral filtering, in pixels.
* `--bilateral_filter_radius_factor` (default: 2): Factor on bilateral_filter_sigma_xy to define the kernel radius for depth bilateral filtering.
* `--bilateral_filter_sigma_depth_factor` (default: 0.05): Factor on the depth to compute sigma_depth for depth bilateral filtering.
* `--outlier_filtering_depth_tolerance_factor` (default: 0.02): Factor on the depth to define the size of the inlier region for outlier filtering.
* `--point_radius_extension_factor` (default: 1.5): Factor by which a point's radius is extended beyond the distance to its farthest neighbor.
* `--point_radius_clamp_factor` (default: inf): Factor by which a point's radius can be larger than the distance to its closest neighbor (times sqrt(2)). Larger radii are clamped to this distance.

#### Octree ####

* `--max_surfels_per_node` (default: 50): Maximum number of surfels per octree node. Should only affect the runtime.

#### File export ####

* `--export_mesh` (default: ""): Save the final mesh to the given path (as an OBJ file).
* `--export_point_cloud` (default: ""): Save the final (surfel) point cloud to the given path (as a PLY file).

#### Visualization ####

* `--hide_camera_frustum`: Hides the input camera frustum rendering.
* `--hide_new_surfel_splats`: Hides the splat rendering of new surfels which are not meshed yet.
* `--splat_half_extent_in_pixels` (default: 3.0): Half splat quad extent in pixels.
* `--triangle_normal_shading`: Colors the mesh triangles based on their triangle normal.
* `--hide_input_images`: Hides the input images (which are normally shown in separate windows). Might speed up the system as well since showing the images can be slow.
* `--render_window_default_width` (default: 1280): Default width of the 3D visualization window.
* `--render_window_default_height` (default: 720): Default height of the 3D visualization window.
* `--exit_after_processing`: After processing the video, exit immediately instead of continuing to show the reconstruction.
* `--follow_input_camera` (default: ""): Make the visualization camera follow the input camera (true / false).
* `--record_keyframes` (default: ""): Record keyframes for video recording to the given file. It is recommended to also set --step_by_step_playback and --show_result.
* `--playback_keyframes` (default: ""): Play back keyframes for video recording from the given file.

#### Debug and evaluation ####

* `--create_video`: Records a video by writing screenshots frame-by-frame to the current working directory.
* `--debug_depth_preprocessing`: Activates debug display of the depth maps at various stages of pre-processing.
* `--debug_neighbor_rendering`: Activates debug rendering of surfel regularization neighbors.
* `--debug_normal_rendering`: Activates debug rendering of surfel normal vectors.
* `--visualize_last_update_timestamp`: Show a visualization of the surfel last update timestamps.
* `--visualize_creation_timestamp`: Show a visualization of the surfel creation timestamps.
* `--visualize_radii`: Show a visualization of the surfel radii.
* `--visualize_surfel_normals`: Show a visualization of the surfel normals.
* `--log_timings` (default: ""): Log the timings to the given file.
