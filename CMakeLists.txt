cmake_minimum_required(VERSION 2.8.11)

project(libvis)

# Make CMake find the Find<Package>.cmake files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")


################################################################################
# Dependencies and settings.

# CUDA (external)
find_package(CUDA REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)
# if(CUDA_MULTI_ARCH)
#     CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS All)
# else()
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
# endif()
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-use_fast_math")

# Eigen (external)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# # OpenMP (external)
# find_package(OpenMP)
# if (OPENMP_FOUND)
#   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

# Boost (external)
find_package(Boost COMPONENTS serialization filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Loguru (packaged)
add_library(loguru SHARED
  libvis/third_party/loguru/loguru.cpp
)
set_property(TARGET loguru PROPERTY FOLDER "libvis/third_party")
target_include_directories(loguru PUBLIC
  libvis/third_party/loguru/
)
set_target_properties(loguru PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(loguru PRIVATE
  LOGURU_WITH_STREAMS
)
if(MSVC)
  set(LOGURU_DLL_EXPORT "__declspec(dllexport)")
  #set(LOGURU_DLL_IMPORT "__declspec(dllimport)")
endif()
target_compile_definitions(loguru PRIVATE
  LOGURU_EXPORT=${LOGURU_DLL_EXPORT}
)
target_link_libraries(loguru
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

# GTest (packaged)
add_subdirectory(libvis/third_party/gtest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# libpng (packaged)
add_subdirectory(libvis/third_party/libpng)
include_directories(libvis/third_party)

# Sophus (packaged)
include_directories(libvis/third_party/sophus)

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# Vulkan (optional)
set(VULKAN_PATH "" CACHE PATH "Set this to the path of the Vulkan SDK installation (the directory containing include and lib) if installed locally, otherwise leave this empty.")
if(NOT (VULKAN_PATH STREQUAL ""))
  add_definitions(-DLIBVIS_HAVE_VULKAN)
  include_directories("${VULKAN_PATH}/include")
  link_directories("${VULKAN_PATH}/lib")
  set(VULKAN_FOUND TRUE)
else()
  set(VULKAN_FOUND FALSE)
endif()

# Qt5 (external, required)
# TODO: Qt should be an optional dependency.
# Find includes in corresponding build directories.
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# Instruct CMake to run rcc (resource compiler) automatically when needed.
set(CMAKE_AUTORCC ON)
find_package(Qt5Widgets)
find_package(Qt5X11Extras)
find_package(Qt5OpenGL)

if(Qt5Widgets_FOUND)
  add_definitions(-DLIBVIS_HAVE_QT)
endif()


# Settings.
add_definitions("-Wall -Wextra -O2 -msse2 -msse3 -std=c++11")

# TODO: It seems gcc cannot disable this locally, therefore added it here as a
#       workaround.
add_definitions("-Wno-unknown-pragmas")

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUXX)
  add_definitions("-Wno-missing-field-initializers")
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})


################################################################################
# Helper application bin2c required for building libvis.

add_executable(bin2c
  ./libvis/src/bin2c/main.cc
)
target_link_libraries(bin2c
  ${Boost_LIBRARIES}
)


################################################################################
# Function for generating headers with SPIR-V bytecode from GLSL code.

# Sets up compile and header-conversion commands for Vulkan shaders. The
# resulting header file paths are appended to the variable whose name is passed
# in for _GENERATED_HEADERS. These headers must be specified in a target to
# trigger the shader build steps.
# NOTE: _GENERATED_HEADERS must differ from the variable name being passed in,
#       otherwise the dereferencing returns the wrong variable.
function(add_vulkan_shader TARGETNAME INPUT_FILEPATH _GENERATED_HEADERS)
  get_filename_component(INPUT_FILENAME ${INPUT_FILEPATH} NAME)
  
  # Vertex shader.
  add_custom_command (
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.vert.h
    COMMAND glslangValidator -V -S vert -o ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.vert ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILEPATH}.vert
    COMMAND bin2c -n vis -H ${CMAKE_CURRENT_SOURCE_DIR}/libvis/src/libvis/shaders/license_header.h ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.vert
    DEPENDS bin2c ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILEPATH}.vert
    )
  
  # Fragment shader.
  add_custom_command (
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.frag.h
    COMMAND glslangValidator -V -S frag -o ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.frag ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILEPATH}.frag
    COMMAND bin2c -n vis -H ${CMAKE_CURRENT_SOURCE_DIR}/libvis/src/libvis/shaders/license_header.h ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.frag
    DEPENDS bin2c ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILEPATH}.frag
    )
  
  # Return generated headers in the variable named ${_GENERATED_HEADERS} in the
  # parent scope.
  set(${_GENERATED_HEADERS}
    ${${_GENERATED_HEADERS}}
    ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.vert.h
    ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILENAME}.frag.h
    PARENT_SCOPE
  )
endfunction()


################################################################################
# libvis.

include_directories(
  ./libvis/src
)

if(VULKAN_FOUND)
  add_vulkan_shader(libvis
    libvis/src/libvis/shaders/shader
    GENERATED_HEADERS
  )
endif()

set(LIBVIS_FILES
  libvis/src/libvis/camera.h
  libvis/src/libvis/camera_frustum_opengl.h
  libvis/src/libvis/command_line_parser.cc
  libvis/src/libvis/command_line_parser.h
  libvis/src/libvis/dlt.h
  libvis/src/libvis/eigen.h
  libvis/src/libvis/geometry.h
  libvis/src/libvis/glew.cc
  libvis/src/libvis/glew.h
  libvis/src/libvis/image.cc
  libvis/src/libvis/image.h
  libvis/src/libvis/image_cache.h
  libvis/src/libvis/image_display.cc
  libvis/src/libvis/image_display.h
  libvis/src/libvis/image_display_qt_widget.cc
  libvis/src/libvis/image_display_qt_widget.h
  libvis/src/libvis/image_display_qt_window.cc
  libvis/src/libvis/image_display_qt_window.h
  libvis/src/libvis/image_frame.h
  libvis/src/libvis/image_io.cc
  libvis/src/libvis/image_io.h
  libvis/src/libvis/image_io_libpng.cc
  libvis/src/libvis/image_io_libpng.h
  libvis/src/libvis/image_io_netpbm.cc
  libvis/src/libvis/image_io_netpbm.h
  libvis/src/libvis/image_io_qt.cc
  libvis/src/libvis/image_io_qt.h
  libvis/src/libvis/libvis.cc
  libvis/src/libvis/libvis.h
  libvis/src/libvis/lm_optimizer.h
  libvis/src/libvis/lm_optimizer_impl.h
  libvis/src/libvis/lm_optimizer_update_accumulator.h
  libvis/src/libvis/loss_functions.h
  libvis/src/libvis/mesh.h
  libvis/src/libvis/mesh_opengl.h
  libvis/src/libvis/opengl.cc
  libvis/src/libvis/opengl.h
  libvis/src/libvis/opengl_context.cc
  libvis/src/libvis/opengl_context.h
  libvis/src/libvis/opengl_context_glx.cc
  libvis/src/libvis/opengl_context_glx.h
  libvis/src/libvis/opengl_context_qt.cc
  libvis/src/libvis/opengl_context_qt.h
  libvis/src/libvis/patch_match_stereo.cc
  libvis/src/libvis/patch_match_stereo.h
  libvis/src/libvis/point_cloud.h
  libvis/src/libvis/point_cloud_opengl.h
  libvis/src/libvis/qt_thread.cc
  libvis/src/libvis/qt_thread.h
  libvis/src/libvis/render_display.cc
  libvis/src/libvis/render_display.h
  libvis/src/libvis/render_window.cc
  libvis/src/libvis/render_window.h
  libvis/src/libvis/render_window_qt.cc
  libvis/src/libvis/render_window_qt.h
  libvis/src/libvis/render_window_qt_opengl.cc
  libvis/src/libvis/render_window_qt_opengl.h
  libvis/src/libvis/renderer.cc
  libvis/src/libvis/renderer.h
  libvis/src/libvis/rgbd_video.h
  libvis/src/libvis/rgbd_video_io_tum_dataset.h
  libvis/src/libvis/shader_program_opengl.cc
  libvis/src/libvis/shader_program_opengl.h
  libvis/src/libvis/sophus.h
  libvis/src/libvis/statistics.h
  libvis/src/libvis/timing.cc
  libvis/src/libvis/timing.h
  libvis/src/libvis/util.h
  
  ${GENERATED_HEADERS}
  libvis/resources/resources.qrc
)
if(VULKAN_FOUND)
  set(LIBVIS_FILES
    libvis/src/libvis/render_window_qt_vulkan.cc
    libvis/src/libvis/render_window_qt_vulkan.h
    libvis/src/libvis/vulkan.cc
    libvis/src/libvis/vulkan.h
    ${LIBVIS_FILES}
  )
endif()
add_library(libvis SHARED
  ${LIBVIS_FILES}
)

set(BASE_LIB_LIBRARIES
  ${Boost_LIBRARIES}
  ${GLEW_LIBRARIES}
  GL
  loguru
  Qt5::Widgets
  Qt5::X11Extras
  Qt5::OpenGL
  png
  z
  pthread
)
if(VULKAN_FOUND)
  set(BASE_LIB_LIBRARIES
    vulkan
    ${BASE_LIB_LIBRARIES}
  )
endif()
target_link_libraries(libvis
  ${BASE_LIB_LIBRARIES}
)
set(BASE_LIB_LIBRARIES
  libvis
  ${BASE_LIB_LIBRARIES}
)


# libvis optional library: libvis_cuda.
# Contains CUDA functionality, which is only useful with NVIDIA graphics cards.
cuda_add_library(libvis_cuda SHARED
  libvis/src/libvis/cuda/cuda_auto_tuner.h
  libvis/src/libvis/cuda/cuda_buffer.cu
  libvis/src/libvis/cuda/cuda_buffer.cuh
  libvis/src/libvis/cuda/cuda_buffer.h
  libvis/src/libvis/cuda/cuda_buffer_inl.h
  libvis/src/libvis/cuda/cuda_matrix.cuh
  libvis/src/libvis/cuda/cuda_unprojection_lookup.cuh
  libvis/src/libvis/cuda/cuda_unprojection_lookup.h
  libvis/src/libvis/cuda/cuda_util.h
  libvis/src/libvis/cuda/patch_match_stereo.cc
  libvis/src/libvis/cuda/patch_match_stereo.cu
  libvis/src/libvis/cuda/patch_match_stereo.cuh
  libvis/src/libvis/cuda/patch_match_stereo.h
  libvis/src/libvis/cuda/pixel_corner_projector.cuh
  libvis/src/libvis/cuda/pixel_corner_projector.h
)
target_link_libraries(libvis_cuda
  libvis
)


# Applications.
add_subdirectory(applications)
