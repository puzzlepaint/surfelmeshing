cmake_minimum_required(VERSION 2.8.11)
project(libvis LANGUAGES C CXX)


# Enable C++11 globally
set(CMAKE_CXX_STANDARD 11)

# Support IDEs: https://cliutils.gitlab.io/modern-cmake/chapters/features/ides.html
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake-default-targets")

if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  message ("Enabling parallel builds in visual studio")
endif()

# Make CMake find the Find<Package>.cmake files in the cmake subdirectory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(CheckLanguage)


################################################################################
# Dependencies and settings.

# CUDA (external)
# Enable C++11 globally
# NOTE: The user of this CMake file should set the CUDA arch when invoking
#       CMake, for example with: -DCMAKE_CUDA_FLAGS="-arch=sm_61"
#       Common settings are either the 'native' arch for which the binary will
#       be known to run on exclusively, or a broader set of PTX architectures
#       in case the binary is intended for distribution.
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  # Enable C++11 globally for CUDA code
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
else()
  message(FATAL_ERROR "CMake did not find CUDA, which is required to build this program.")
endif()

# Cross-platform threading. See:
# https://cmake.org/cmake/help/latest/module/FindThreads.html
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)

# Eigen (external)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# Boost (external)
if (POLICY CMP0074) # get rid of annoying warning
  cmake_policy(SET CMP0074 NEW)
endif()
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

if (UNIX AND NOT APPLE)
  find_package(Qt5X11Extras REQUIRED)
endif()
if (TARGET Qt5::Core AND UNIX)
  # HACK, see: https://gitlab.kitware.com/cmake/cmake/issues/16915
  get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
  string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
  set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
  set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
  set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()
find_package(Qt5OpenGL)

if(Qt5Widgets_FOUND)
  add_definitions(-DLIBVIS_HAVE_QT)
endif()


# Settings.
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUXX)
  set(LIBVIS_WARNING_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-Wextra>")
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-O2>")
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-msse2>")
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-msse3>")
  # TODO: It seems gcc cannot disable this locally, therefore added it here as a
  #       workaround.
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>")
  
  # NOTE: Disabled this since it triggered in a lot of places, including external headers, creating a lot of warning spam.
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>")
  set(LIBVIS_WARNING_OPTIONS "${LIBVIS_WARNING_OPTIONS};$<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>")
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})


################################################################################
# Helper application bin2c required for building libvis.

add_executable(bin2c
  ./libvis/src/bin2c/main.cc
)
set_property(TARGET bin2c PROPERTY FOLDER "libvis")

target_include_directories(bin2c PRIVATE
  ${Boost_INCLUDE_DIR}
)

target_compile_options(bin2c PRIVATE
  "${LIBVIS_WARNING_OPTIONS}"
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

add_library(libvis 
  ${LIBVIS_FILES}
)
set_property(TARGET libvis PROPERTY FOLDER "libvis")

target_compile_options(libvis PRIVATE
  "${LIBVIS_WARNING_OPTIONS}"
)

if(MSVC)
  target_compile_definitions(libvis PUBLIC NOMINMAX) # to get rid of max define
endif()

set(BASE_LIB_LIBRARIES
  ${Boost_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${OPENGL_LIBRARY}
  loguru
  ${Qt5Widgets_LIBRARIES}
  ${Qt5X11Extras_LIBRARIES}
  ${Qt5Opengl_LIBRARIES}
  png
  Threads::Threads
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
add_library(libvis_cuda 
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
set_property(TARGET libvis_cuda PROPERTY FOLDER "libvis")
target_include_directories(libvis_cuda PUBLIC
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_compile_options(libvis_cuda PRIVATE
  "${LIBVIS_WARNING_OPTIONS}"
)
target_link_libraries(libvis_cuda
  libvis
)
target_compile_options(libvis_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)


# Applications.
add_subdirectory(applications)
