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

#include <queue>
#include <memory>
#include <set>
#include <unordered_map>

#include <libvis/camera.h>
#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/image_frame.h>
#include <libvis/libvis.h>
#include <libvis/mesh.h>
#include <libvis/point_cloud.h>

#include "surfel_meshing/cuda_surfels_cpu.h"
#include "surfel_meshing/octree.h"
#include "surfel_meshing/surfel.h"

namespace vis {

class SurfelMeshingRenderWindow;
struct Neighbor;
struct EdgeData;
struct SkinnySurfel;

// Performs meshing of surfels on the CPU.
class SurfelMeshing {
 friend class SurfelMeshingRenderWindow;
 public:
  // Constructor, does nothing.
  // The render window is passed in for debug purposes only.
  SurfelMeshing(
      int max_surfels_per_node,
      float max_angle_between_normals,
      float min_triangle_angle,
      float max_triangle_angle,
      float max_neighbor_search_range_increase_factor,
      float long_edge_tolerance_factor,
      int regularization_frame_window_size,
      const shared_ptr<SurfelMeshingRenderWindow>& render_window);
  
  // Updates this SurfelMeshing's CPU surfels to the contents of the buffers
  // coming from the CUDA surfels.
  void IntegrateCUDABuffers(
      int frame_index,
      const CUDASurfelsCPU& buffers);
  
  // Checks where remeshing must be done and deletes invalid triangles.
  void CheckRemeshing();
  
  // Iterates over surfels_to_remesh_ to perform a triangulation iteration.
  void Triangulate(bool force_debug = false);
  
  // Converts the surfels to a point cloud. All colors are set to black.
  void ConvertToPoint3fC3u8Cloud(Point3fC3u8Cloud* output);
  
  // Outputs the mesh as a Mesh3fCu8. Optionally, only outputs the indices.
  // TODO: indices_only has a hidden side effect wrt. including merged vertices in the indexing, document this (or better: split into two functions)
  void ConvertToMesh3fCu8(Mesh3fCu8* output, bool indices_only = false);
  
  // Provides raw (read) access to the surfels.
  inline const vector<Surfel>& surfels() const { return surfels_; }
  
  // Returns the number of valid triangles (which may be smaller than the triangles vector size).
  inline usize triangle_count() const {
    usize count = 0;
    for (usize triangle_index = 0, size = triangles_.size(); triangle_index < size; ++ triangle_index) {
      const SurfelTriangle* triangle = &triangles_[triangle_index];
      if (triangle->IsValid()) {
        ++ count;
      }
    }
    return count;
  }
  
  // Returns the number of triangles which were deleted in the last remeshing
  // iteration.
  inline usize deleted_triangle_count() const { return deleted_triangle_count_; }
  
  
  // Deletes triangles within (approximately) neighbor_search_radius_squared
  // around the surfel and makes the affected surfels be remeshed later.
  // Only in public such that it can be accessed from tests.
  void RemeshTrianglesAt(Surfel* surfel,
                         float neighbor_search_radius_squared);
  
  // For debugging: performs a full re-triangulation of all surfels. Returns
  // the time the meshing-from-scratch took in seconds. This excludes the time
  // required for deleting the existing mesh (which is not done in a performant
  // way and should not be included in the timings anyway).
  double FullRetriangulation();
  
  // For debugging: sets a single surfel to be remeshed.
  void SetSurfelToRemesh(u32 surfel_index);
  
  // For debugging: Determines the surfel state from its adjacent triangles and
  // checks whether it matches the state stored in the surfel struct.
  // NOTE: Does not check that the triangles have consistent orientations.
  void CheckSurfelState(u32 surfel_index);
  
 private:
  constexpr static u32 kNoFreeIndex = std::numeric_limits<u32>::max();
  
  // Attempts to triangulate the surfel with the given index.
  void TriangulateSurfel(
      u32 surfel_index,
      int max_neighbors,
      u32* neighbor_indices,
      float* neighbor_distances_squared,
      Neighbor* neighbors,
      Neighbor* selected_neighbors,
      std::vector<EdgeData>* double_edges,
      bool* gaps,
      bool* skinny,
      float* angle_diff,
      int* angle_indices,
      bool* to_erase,
      SkinnySurfel* skinny_surfels,
      bool force_debug = false,
      bool no_surfel_resets = false);
  
  // Deletes the triangle, adds the adjacent surfels to surfels_to_remesh_.
  void DeleteTriangleForRemeshing(u32 triangle_index);
  
  // Variant of DeleteTriangleForRemeshing() where one surfel is not handled
  // for better performance since it is supposed to get reset afterwards anyway.
  void DeleteTriangleForRemeshing(u32 triangle_index, u32 reset_surfel_index);
  
  // Updates the fronts of the surfel for a removed triangle.
  bool UpdateFrontsOnTriangleRemoval(u32 surfel_index,
                                     u32 left_surfel_index,
                                     u32 right_surfel_index);
  
  void ProjectNeighborsAndTestVisibility(
      u32 surfel_index,
      Surfel* surfel,
      const Vec3f& surfel_proj,
      int neighbor_count,
      u32* neighbor_indices,
      Neighbor* neighbors,
      const Vec3f& u,
      const Vec3f& v,
      std::vector<EdgeData>* double_edges,
      bool debug);
  
  void TryToAdvanceFront(
      u32 surfel_index, std::vector<Front>* surfel_front, int neighbor_count, u32* neighbor_indices,
      Neighbor* neighbors, std::vector<EdgeData>* double_edges, Neighbor* selected_neighbors,
      bool* gaps, bool* skinny, float* angle_diff, int* /*angle_indices*/, bool* to_erase, SkinnySurfel* skinny_surfels,
      int max_neighbor_count, bool no_surfel_resets, bool debug);
  
  // Left and right are meant from the stand point of the reference surfel,
  // looking outwards for triangulation.
  void UpdateSurfelFronts(
      u32 corner_surfel_index,
      u32 left_surfel_index,
      u32 right_surfel_index,
      const Vec3f& /*projection_normal*/,
      float corner_angle,
      const Vec3f& surfel_proj,
      const Vec2f& corner_uv,
      const Vec3f& u,
      const Vec3f& v,
      u32 reference_surfel_index);
  
  void CloseFront(
      u32 surfel_index,
      std::vector<Front>* surfel_fronts,
      usize front_index);
  
  bool TryToCreateInitialTriangle(
      u32 surfel_index, int neighbor_count, u32* neighbor_indices,
      Neighbor* neighbors, std::vector<EdgeData>* double_edges, bool debug);
  
  // Adds a triangle with the surfels with the given indices as corners.
  // Does not change any surfel attributes, this must be done separately.
  void AddTriangle(u32 a, u32 b, u32 c, bool debug);
  
  // Variant of IsVisible where the ray starts from the given origin point.
  template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
  bool IsVisible(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& S1, const MatrixBase<DerivedC>& S2, const MatrixBase<DerivedD>& origin);
  
  // Determines whether the line from the origin to X crosses the line segment
  // defined by the endpoints S1 and S2.
  template <typename DerivedA, typename DerivedB, typename DerivedC>
  bool IsVisible(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& S1, const MatrixBase<DerivedC>& S2);
  
  // Determines whether X is in front of the line having S1 and S2 on it, as
  // seen from the origin.
  template <typename DerivedA, typename DerivedB, typename DerivedC>
  bool IsInFrontOfLine(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& S1, const MatrixBase<DerivedC>& S2);
  
  // Deletes all triangles connected to a surfel.
  void DeleteAllTrianglesConnectedToSurfel(u32 surfel_index);
  
  // Deletes all triangles connected to the surfel and sets it to free. Puts it
  // (and the surfels next to the deleted triangles) on the remeshing queue.
  void ResetSurfelToFree(u32 surfel_index);
  
  
  // Unordered list of all surfels.
  vector<Surfel> surfels_;
  
  // Lazy compressed octree containing all surfels.
  CompressedOctree octree_;
  
  // Frame index used to provide the creation stamp for surfels and triangles.
  u32 frame_index_;
  
  // Set by Integrate() such that Triangulate() can run on only the new surfels.
  usize first_new_surfel_index_;
  
  // Queue of surfel indices to reconsider in the triangulation algorithm.
  vector<u32> surfels_to_remesh_;
  
  // Set of surfels to check for whether remeshing is necessary.
  vector<u32> surfels_to_check_;
  
  // Unordered list of all triangles.
  vector<SurfelTriangle> triangles_;
  
  // Free list index for the triangles_ vector.
  u32 next_free_triangle_index_;
  
  // Number of surfel entries which are inactive due to having been merged.
  u32 merged_surfel_count_;
  
  // Settings:
  float cos_max_angle_between_normals_;
  float min_triangle_angle_;
  float max_triangle_angle_;
  float max_neighbor_search_range_increase_factor_;
  float long_edge_tolerance_factor_;
  int regularization_frame_window_size_;
  
  float max_neighbor_search_range_increase_factor_squared_;
  float long_edge_total_factor_squared_;
  
  // Debug counters:
  usize holes_closed_counter_;
  usize front_neighbors_too_far_away_counter_;
  usize front_leads_to_completed_surfel_counter_;
  usize max_neighbor_count_exceeded_counter_;
  usize front_neighbors_not_visible_counter_;
  usize fronts_triangles_inconsistency_counter_;
  usize fronts_sharing_edge_counter_;
  usize connected_to_surfel_without_suitable_front_counter_;
  
  usize deleted_triangle_count_;
  
  // Temporary variables, stored here to avoid re-allocation:
  vector<float> surfel_distances_squared_;
  vector<u32> surfel_indices_;
  vector<Front> new_fronts_;
  
  // For debugging only:
  shared_ptr<SurfelMeshingRenderWindow> render_window_;
};

}
