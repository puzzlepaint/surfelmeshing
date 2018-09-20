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

#include "surfel_meshing/surfel_meshing.h"

#include <libvis/image_display.h>
#include <libvis/timing.h>

#include "surfel_meshing/surfel_meshing_render_window.h"

namespace vis {

// Stores temporary data about a neighbor surfel during triangulation.
struct Neighbor {
  // Coordinates on the tangent plane. Only computed if visible == true.
  Vec2f uv;
  
  // Angle relative to the reference surfel. Only computed if visible == true.
  float angle;
  
  // Global surfel index (to retain the index information when sorting).
  u32 surfel_index;
  
  // Index in nearest-neighbor list (to retain this information when sorting).
  u32 nearest_neighbor_index;
  
  // Whether the neighbor is visible from the reference surfel (considering
  // the mesh boundary projected onto the tangent plane).
  bool visible;
};


// Functor to sort neighbors by angle in ascending order.
class NeighborSortByAngleAsc {
 public:
  inline NeighborSortByAngleAsc() {};
  
  inline bool operator()(const Neighbor& a, const Neighbor& b) const {
    return a.angle < b.angle;
  }
};


// Stores an edge starting from a front point when using it in the
// triangulation code.
struct EdgeData {
  // Index of the neighbor point (within the neighbor list) to which this edge belongs.
  u32 neighbor_index;
  
  // Surfel index of the edge endpoint.
  u32 end_index;
  
  // In-plane coordinates of the edge endpoint.
  Vec2f end_pos;
};


// Stores information about surfels next to narrow triangles in the meshing.
struct SkinnySurfel {
  u8 selected_neighbor_index;
  u8 nearest_neighbor_index;
};


// Functor to sort SkinnySurfel by distance in descending order.
class SkinnySurfelSortByDistance {
 public:
  inline SkinnySurfelSortByDistance() {}
  
  inline bool operator()(const SkinnySurfel& a, const SkinnySurfel& b) {
    return a.nearest_neighbor_index > b.nearest_neighbor_index;
  }
};


// Used for finding connected components of triangles attached to a surfel.
// Stores the two surfels at the edges of the component.
struct ConnectedTriangleComponent {
  u32 surfel_index_a;
  u32 surfel_index_b;
};


// Fast atan2 approximation.
// Taken from: https://www.dsprelated.com/showarticle/1052.php
float ApproxAtan2(float y, float x) {
  constexpr float pi=3.141593f;
  constexpr float halfpi=1.570796f;

  constexpr float n1 = 0.97239411f;
  constexpr float n2 = -0.19194795f;    
  float result = 0.0f;
  if (x != 0.0f) {
    const union { float flVal; u32 nVal; } tYSign = { y };
    const union { float flVal; u32 nVal; } tXSign = { x };
    if (fabsf(x) >= fabsf(y)) {
      union { float flVal; u32 nVal; } tOffset = { pi };
      // Add or subtract PI based on y's sign.
      tOffset.nVal |= tYSign.nVal & 0x80000000u;
      // No offset if x is positive, so multiply by 0 or based on x's sign.
      tOffset.nVal *= tXSign.nVal >> 31;
      result = tOffset.flVal;
      const float z = y / x;
      result += (n1 + n2 * z * z) * z;
    } else { // Use atan(y/x) = pi/2 - atan(x/y) if |y/x| > 1.
      union { float flVal; u32 nVal; } tOffset = { halfpi };
      // Add or subtract PI/2 based on y's sign.
      tOffset.nVal |= tYSign.nVal & 0x80000000u;            
      result = tOffset.flVal;
      const float z = x / y;
      result -= (n1 + n2 * z * z) * z;            
    }
  } else if (y > 0.0f) {
    result = halfpi;
  } else if (y < 0.0f) {
    result = -halfpi;
  }
  return result;
}


SurfelMeshing::SurfelMeshing(
    int max_surfels_per_node,
    float max_angle_between_normals,
    float min_triangle_angle,
    float max_triangle_angle,
    float max_neighbor_search_range_increase_factor,
    float long_edge_tolerance_factor,
    int regularization_frame_window_size,
    const shared_ptr<SurfelMeshingRenderWindow>& render_window)
    : octree_(max_surfels_per_node, &surfels_, &triangles_),
      render_window_(render_window) {
  cos_max_angle_between_normals_ = cos(max_angle_between_normals);
  min_triangle_angle_ = min_triangle_angle;
  max_triangle_angle_ = max_triangle_angle;
  max_neighbor_search_range_increase_factor_ = max_neighbor_search_range_increase_factor;
  long_edge_tolerance_factor_ = long_edge_tolerance_factor;
  regularization_frame_window_size_ = regularization_frame_window_size;
  
  max_neighbor_search_range_increase_factor_squared_ =
      max_neighbor_search_range_increase_factor_ *
      max_neighbor_search_range_increase_factor_;
  long_edge_total_factor_squared_ =
      long_edge_tolerance_factor_ * long_edge_tolerance_factor_ *
      max_neighbor_search_range_increase_factor_squared_;
  
  frame_index_ = 0;
  next_free_triangle_index_ = kNoFreeIndex;
  merged_surfel_count_ = 0;
  
  front_neighbors_too_far_away_counter_ = 0;
  front_leads_to_completed_surfel_counter_ = 0;
  max_neighbor_count_exceeded_counter_ = 0;
  front_neighbors_not_visible_counter_ = 0;
  fronts_triangles_inconsistency_counter_ = 0;
  fronts_sharing_edge_counter_ = 0;
  holes_closed_counter_ = 0;
  connected_to_surfel_without_suitable_front_counter_ = 0;
}

void SurfelMeshing::IntegrateCUDABuffers(
    int frame_index,
    const CUDASurfelsCPU& buffers) {
  const CUDASurfelBuffersCPU& buffer = buffers.read_buffers();
  
  // Increase the frame index.
  u32 old_frame_index = frame_index_;
  frame_index_ = frame_index;
  
  // Update surfels which already exist on the CPU side.
  for (usize surfel_index = 0, size = surfels_.size();
        surfel_index < size;
        ++ surfel_index) {
    Surfel* surfel = &surfels_[surfel_index];
    
    // Was the surfel merged?
    if (surfel->node() == nullptr && buffer.surfel_radius_squared_buffer[surfel_index] < 0) {
      continue;  // Zombie surfel
    } else if (surfel->node() && buffer.surfel_radius_squared_buffer[surfel_index] < 0) {
      surfels_to_check_.push_back(surfel_index);
    } else if (surfel->node() == nullptr) {
      // This previous Zombie surfel was reactivated.
      LOG(FATAL) << "A merged surfel got reactivated, this is not supposed to happen.";
      -- merged_surfel_count_;
      octree_.AddSurfelActive(surfel_index, surfel);
    }
    
    // Did the surfel move?
    if (surfel->position().x() != buffer.surfel_x_buffer[surfel_index] ||
        surfel->position().y() != buffer.surfel_y_buffer[surfel_index] ||
        surfel->position().z() != buffer.surfel_z_buffer[surfel_index]) {
      Vec3f new_position(buffer.surfel_x_buffer[surfel_index],
                         buffer.surfel_y_buffer[surfel_index],
                         buffer.surfel_z_buffer[surfel_index]);
      octree_.MoveSurfel(surfel_index, surfel, new_position);
      surfel->SetPosition(new_position);
      
      // Only perform meshing / remeshing if the surfel was updated or regularized.
      // Notably, do not mesh / remesh if the surfel was moved only due to a loop closure.
      // This improves performance and reduces cracks.
      if (buffer.surfel_last_update_stamp_buffer[surfel_index] > surfel->last_update_stamp() ||
          static_cast<int>(old_frame_index) - static_cast<int>(surfel->last_update_stamp()) <= regularization_frame_window_size_) {
        // If this is a front or free surfel, always try to triangulate.
        if (surfel->meshing_state() != Surfel::MeshingState::kCompleted) {
          surfels_to_remesh_.push_back(surfel_index);
        }
        
        // If the surfel is not free, check for remeshing.
        if (surfel->meshing_state() != Surfel::MeshingState::kFree) {
          surfels_to_check_.push_back(surfel_index);
        }
      }
    }
    
    // Update remaining surfel attributes.
    surfel->SetRadiusSquared(buffer.surfel_radius_squared_buffer[surfel_index]);
    surfel->SetNormal(Vec3f(buffer.surfel_normal_x_buffer[surfel_index],
                            buffer.surfel_normal_y_buffer[surfel_index],
                            buffer.surfel_normal_z_buffer[surfel_index]));
    surfel->SetLastUpdateStamp(buffer.surfel_last_update_stamp_buffer[surfel_index]);
    surfel->SetFlags(true, true);
  }
  
  // Store the index of the first new surfel.
  first_new_surfel_index_ = surfels_.size();
  
  // Resize the buffers if necessary. Make sure that if the buffers are resized
  // they are resized by large steps, as this can be very slow.
  if (surfels_.capacity() < buffer.surfel_count) {
    constexpr usize kMinSurfelReserveCount = 3000000;
    surfels_.reserve(std::max(kMinSurfelReserveCount, 2 * buffer.surfel_count));
    triangles_.reserve(2.1f * surfels_.capacity());
  }
  
  // Create new CPU surfels for new CUDA surfels.
  for (usize surfel_index = surfels_.size();
       surfel_index < buffer.surfel_count;
       ++ surfel_index) {
    surfels_.emplace_back(
        Vec3f(buffer.surfel_x_buffer[surfel_index],
              buffer.surfel_y_buffer[surfel_index],
              buffer.surfel_z_buffer[surfel_index]),
        buffer.surfel_radius_squared_buffer[surfel_index],
        Vec3f(buffer.surfel_normal_x_buffer[surfel_index],
              buffer.surfel_normal_y_buffer[surfel_index],
              buffer.surfel_normal_z_buffer[surfel_index]),
        buffer.surfel_last_update_stamp_buffer[surfel_index]);
    surfels_.back().SetFlags(true, false);
    
    if (buffer.surfel_radius_squared_buffer[surfel_index] < 0) {
      // The surfel was already replaced. Do not add it to the octree.
      surfels_.back().SetOctreeNode(nullptr, 0);
      ++ merged_surfel_count_;
    } else {
      // Use the active version of surfel addition to the octree since the
      // triangulation will anyway perform neighbor searches around all new surfels.
      octree_.AddSurfelActive(surfels_.size() - 1, &surfels_.back());
    }
  }
}

void SurfelMeshing::TriangulateSurfel(
    u32 surfel_index,
    int max_neighbors,
    u32* neighbor_indices,
    float* neighbor_distances_squared,
    Neighbor* neighbors,
    Neighbor* selected_neighbors,
    std::vector<EdgeData>* edges,
    bool* gaps,
    bool* skinny,
    float* angle_diff,
    int* angle_indices,
    bool* to_erase,
    SkinnySurfel* skinny_surfels,
    bool force_debug,
    bool no_surfel_resets) {
  Surfel* surfel = &surfels_[surfel_index];
  
  bool debug = force_debug;
  // Use this to debug triangulation for a particular surfel:
  // debug |= surfel_index == 119708;
  if (debug) {
    LOG(INFO) << "DEBUG initial meshing state for surfel " << surfel_index << ": " << static_cast<int>(surfel->meshing_state());
  }
  
  // If the surfel is completed, there is nothing left to triangulate.
  if (surfel->meshing_state() == Surfel::MeshingState::kCompleted) {
    return;
  }
  
  // If this is a front surfel, determine the maximum distance to its
  // front-connected neighbors. If this exceeds the neighbor search radius,
  // enlarge it (up to a maximum factor) to also include those surfels.
  float neighbor_search_radius_squared = surfel->radius_squared();
  std::vector<Front>* surfel_front = nullptr;
  if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
    surfel_front = &surfel->fronts();
    
    // Determine the maximum distance of this surfel to a front neighbor, and
    // the set of front neighbors.
    float max_front_neighbor_squared_dist = 0;
    for (const Front& front : *surfel_front) {
      Surfel* left_surfel = &surfels_[front.left];
      Surfel* right_surfel = &surfels_[front.right];
      
      // Sanity check for robustness, should not be necessary if there are no bugs.
      if (left_surfel->meshing_state() == Surfel::MeshingState::kCompleted ||
          right_surfel->meshing_state() == Surfel::MeshingState::kCompleted) {
        ++ front_leads_to_completed_surfel_counter_;
        if (surfel->can_be_reset() && !no_surfel_resets) {
          if (debug) {
            LOG(INFO) << "Calling ResetSurfelToFree().";
          }
          ResetSurfelToFree(surfel_index);
        }
        return;
      }
      
      float distance_to_left_squared = (surfel->position() - left_surfel->position()).squaredNorm();
      if (distance_to_left_squared > max_front_neighbor_squared_dist) {
        max_front_neighbor_squared_dist = distance_to_left_squared;
      }
      
      float distance_to_right_squared = (surfel->position() - right_surfel->position()).squaredNorm();
      if (distance_to_right_squared > max_front_neighbor_squared_dist) {
        max_front_neighbor_squared_dist = distance_to_right_squared;
      }
    }
    
    float maximum_squared_search_radius =
        max_neighbor_search_range_increase_factor_squared_ * surfel->radius_squared();
    if (max_front_neighbor_squared_dist > maximum_squared_search_radius) {
      ++ front_neighbors_too_far_away_counter_;
      
      if (debug) {
        LOG(INFO) << "Front neighbors are too far away.";
      }
      
      // Special case: close one-triangle holes.
      // Require more than one triangle connected to the surfels, otherwise it
      // could also be that the condition evaluates to true for the single
      // triangle, thus creating an incorrect triangle on the opposite side.
      if (surfel->GetTriangleCount() > 1) {
        for (int front_index = static_cast<int>(surfel_front->size()) - 1; front_index >= 0; --front_index) {
          const Front& front = surfel_front->at(front_index);
          Surfel* left_surfel = &surfels_[front.left];
          Surfel* right_surfel = &surfels_[front.right];
          
          if (left_surfel->GetTriangleCount() > 1 &&
              left_surfel->fronts().size() == 1 &&
              left_surfel->fronts().front().left == front.right &&
              left_surfel->fronts().front().right == surfel_index &&
              right_surfel->GetTriangleCount() > 1 &&
              right_surfel->fronts().size() == 1 &&
              right_surfel->fronts().front().left == surfel_index &&
              right_surfel->fronts().front().right == front.left) {
            // TODO: In theory one must ensure here that no surfels are within the triangle
            AddTriangle(surfel_index, front.right, front.left, debug);
            left_surfel->fronts().clear();
            left_surfel->SetMeshingState(Surfel::MeshingState::kCompleted);
            // CheckSurfelState(front.left);
            right_surfel->fronts().clear();
            right_surfel->SetMeshingState(Surfel::MeshingState::kCompleted);
            // CheckSurfelState(front.right);
            surfel_front->erase(surfel_front->begin() + front_index);
          }
        }
      }
      if (surfel_front->empty()) {
        surfel->SetMeshingState(Surfel::MeshingState::kCompleted);
        // CheckSurfelState(surfel_index);
      } else {
        surfel->SetMeshingState(Surfel::MeshingState::kFront);
        surfel->SetCanBeRemeshed(false);
      }
      
      return;
    }
    
    // Enlarge the search radius just enough to find all the front neighbors.
    // Use a safety margin.
    max_front_neighbor_squared_dist = 1.05f * max_front_neighbor_squared_dist;
    if (max_front_neighbor_squared_dist > neighbor_search_radius_squared) {
      neighbor_search_radius_squared = std::min(maximum_squared_search_radius, max_front_neighbor_squared_dist);
    }
  }
  
  // Find the nearest neighbors of this surfel (at most max_neighbors). The
  // resulting indices will be in the neighbor_indices array,
  // the resulting squared neighbor distances will be in neighbor_distances_squared.
  // Completed surfels are already excluded at this stage to improve performance.
  int neighbor_count = octree_.FindNearestSurfelsWithinRadius<false, true>(
      surfel->position(),
      neighbor_search_radius_squared,
      max_neighbors,
      neighbor_distances_squared,
      neighbor_indices);
  if (neighbor_count < 2) {
    // Cannot triangulate with less than 2 neighbors.
    surfel->SetCanBeRemeshed(false);
    return;
  }
  
  // Ensure that neighbor 0 is the surfel itself (this is assumed by the
  // following functions). If not (this may happen if two or more surfels
  // are exactly at the same location), try to find it in the neighbor list
  // and swap it with the surfel in the first place. If not found, abort.
  if (neighbor_indices[0] != surfel_index) {
    bool found = false;
    for (int i = 1; i < neighbor_count; ++ i) {
      if (neighbor_indices[i] == surfel_index) {
        // Swap.
        std::swap(neighbor_indices[0], neighbor_indices[i]);
        found = true;
        break;
      }
    }
    if (!found) {
      LOG(ERROR) << "Did not find the reference surfel (" << surfel_index << ") in the list of its nearest neighbors. neighbor_count: " << neighbor_count;
//       LOG(ERROR) << "Reference surfel position: " << surfel->position().transpose();
//       LOG(ERROR) << "neighbor_search_radius_squared: " << neighbor_search_radius_squared;
//       LOG(ERROR) << "max_neighbors: " << max_neighbors;
//       LOG(ERROR) << "octree root: " << *octree_.root();
//       LOG(ERROR) << "octree root child_count: " << static_cast<int>(octree_.root()->child_count);
//       LOG(ERROR) << "surfel node: " << surfel->node();
//       LOG(ERROR) << "surfel index_in_node: " << surfel->index_in_node();
//       if (surfel->node()) {
//         LOG(ERROR) << "surfel node details: " << *surfel->node();
//       }
//       for (int i = 0; i < neighbor_count; ++ i) {
//         LOG(ERROR) << "Neighbor " << i << ": " << neighbor_indices[i] << " (distance_squared: " << neighbor_distances_squared[i] << ")";
//       }
      surfel->SetCanBeRemeshed(false);
      return;
    }
  }
  
  // If the surfel meshing state is "free", try to create an initial
  // triangle connected to this surfel.
  if (surfel->meshing_state() == Surfel::MeshingState::kFree) {
    bool initial_triangle_created =
        TryToCreateInitialTriangle(surfel_index, neighbor_count,
                                    neighbor_indices, neighbors,
                                    edges, debug);
    if (!initial_triangle_created) {
      // TODO: Did not find an initial triangle for this surfel, mark it somehow?
    } else {
      surfel_front = &surfel->fronts();
      if (surfel_front->size() != 1) {
        LOG(ERROR) << "A surfel should have 1 front after initial triangle creation, but encountered one with: " << surfel_front->size();
      }
    }
  }
  
  // DEBUG: Show state after creating the initial triangle.
  if (debug) {
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
    if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
      std::vector<Front>* surfel_front = &surfel->fronts();
      for (usize front_index = 0; front_index < surfel_front->size(); ++ front_index) {
        const Front& front = surfel_front->at(front_index);
        LOG(INFO) << "  Initial front neighbors: left: " << front.left << ", right: " << front.right;
        (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
        (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(205, 205, 60);
      }
    }
    
    render_window_->CenterViewOn(surfel->position());
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
  
  // Try to advance the front if this is a front / boundary surfel now.
  if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
    TryToAdvanceFront(
        surfel_index, surfel_front, neighbor_count, neighbor_indices, neighbors,
        edges, selected_neighbors, gaps, skinny, angle_diff,
        angle_indices, to_erase, skinny_surfels, max_neighbors,
        no_surfel_resets, debug);
  }
  
  // DEBUG: code to show the state after processing a specific surfel (marked
  //        red, front surfels marked yellow, right slightly darker than left).
  if (debug) {
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
    if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
      std::vector<Front>* surfel_front = &surfel->fronts();
      for (usize front_index = 0; front_index < surfel_front->size(); ++ front_index) {
        const Front& front = surfel_front->at(front_index);
        (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
        (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(205, 205, 60);
      }
    }
    
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
  
  surfel->SetCanBeRemeshed(false);
}

void SurfelMeshing::CheckRemeshing() {
  deleted_triangle_count_ = 0;
  
  // Delete old triangles where new surfels were created.
  ConditionalTimer remesh_old_triangles_loop_timer("- CheckRemeshing: Remesh new surfels");
  for (usize surfel_index = first_new_surfel_index_, size = surfels_.size();
        surfel_index < size;
        ++ surfel_index) {
    Surfel* surfel = &surfels_[surfel_index];
    if (surfel->node() == nullptr) {
      continue;
    }
    RemeshTrianglesAt(surfel, surfel->radius_squared());
    surfels_to_remesh_.push_back(surfel_index);
  }
  remesh_old_triangles_loop_timer.Stop();
  
  // Check existing surfels / triangles.
  vector<bool> triangle_was_checked(triangles_.size(), false);
  ConditionalTimer check_existing_surfels_timer("- CheckRemeshing: Check existing surfels");
  for (u32 surfel_index : surfels_to_check_) {
    Surfel* surfel = &surfels_[surfel_index];
    bool remeshed = false;
    
    float maximum_squared_edge_length =
        long_edge_total_factor_squared_ * surfel->radius_squared();
    
    if (maximum_squared_edge_length < 0) {
      // The surfel was merged. Remove it from the octree and mesh.
      if (surfel->node()) {
        DeleteAllTrianglesConnectedToSurfel(surfel_index);
        octree_.RemoveSurfel(surfel_index);
        surfel->SetOctreeNode(nullptr, 0);
        ++ merged_surfel_count_;
      }
      continue;
    }
    
    // Iterate over all triangles connected to this surfel.
    for (int triangle_index = 0, triangle_count = surfel->GetTriangleCount();
         triangle_index < triangle_count; ++ triangle_index) {
      if (triangle_was_checked[surfel->GetTriangle(triangle_index)]) {
        continue;
      } else {
        triangle_was_checked[surfel->GetTriangle(triangle_index)] = true;
      }
      SurfelTriangle* triangle = &triangles_[surfel->GetTriangle(triangle_index)];
      
      // Identify the two other surfels referenced in the triangle.
      u32 surfel_index_right;
      u32 surfel_index_left;
      if (surfel_index == triangle->index(0)) {
        surfel_index_right = triangle->index(1);
        surfel_index_left = triangle->index(2);
      } else if (surfel_index == triangle->index(1)) {
        surfel_index_right = triangle->index(2);
        surfel_index_left = triangle->index(0);
      } else if (surfel_index == triangle->index(2)) {
        surfel_index_right = triangle->index(0);
        surfel_index_left = triangle->index(1);
      } else {
        LOG(FATAL) << "CheckRemeshing(): found a triangle referenced in a surfel which does not have the surfel as one of its indices!";
        continue;
      }
      
      Surfel* surfel_right = &surfels_[surfel_index_right];
      Surfel* surfel_left = &surfels_[surfel_index_left];
      
      float maximum_squared_a_edge_length =
          long_edge_total_factor_squared_ * surfel_right->radius_squared();
      float maximum_squared_b_edge_length =
          long_edge_total_factor_squared_ * surfel_left->radius_squared();
      
      Vec3f s_to_right = surfel_right->position() - surfel->position();
      float edge_a_length_squared = s_to_right.squaredNorm();
      Vec3f s_to_left = surfel_left->position() - surfel->position();
      float edge_b_length_squared = s_to_left.squaredNorm();
      float edge_ab_length_squared = (surfel_right->position() - surfel_left->position()).squaredNorm();
      
      // Check edge lengths. If too long, remesh.
      if ((edge_a_length_squared > maximum_squared_edge_length &&        // edge to a
           edge_a_length_squared > maximum_squared_a_edge_length &&
            (edge_b_length_squared > maximum_squared_b_edge_length ||
             edge_ab_length_squared > maximum_squared_b_edge_length)) ||
          (edge_b_length_squared > maximum_squared_edge_length &&        // edge to b
           edge_b_length_squared > maximum_squared_b_edge_length &&
            (edge_a_length_squared > maximum_squared_a_edge_length ||
             edge_ab_length_squared > maximum_squared_a_edge_length)) ||
          (edge_ab_length_squared > maximum_squared_a_edge_length &&     // edge from a to b
           edge_ab_length_squared > maximum_squared_b_edge_length &&
            (edge_a_length_squared > maximum_squared_edge_length ||
             edge_b_length_squared > maximum_squared_edge_length))) {
        RemeshTrianglesAt(surfel, surfel->radius_squared());
        if (surfel_right->meshing_state() != Surfel::MeshingState::kFree) {
          RemeshTrianglesAt(surfel_right, surfel_right->radius_squared());
        }
        if (surfel_left->meshing_state() != Surfel::MeshingState::kFree) {
          RemeshTrianglesAt(surfel_left, surfel_left->radius_squared());
        }
        remeshed = true;
        break;
      }
      
      // Check whether the normal direction of the triangle is consistent with
      // any of the surfel normals. If not, remesh.
      Vec3f triangle_normal_direction = s_to_right.cross(s_to_left);
      if (triangle_normal_direction.dot(surfel->normal()) <= 0 &&
          triangle_normal_direction.dot(surfel_right->normal()) <= 0 &&
          triangle_normal_direction.dot(surfel_left->normal()) <= 0) {
        RemeshTrianglesAt(surfel, surfel->radius_squared());
        if (surfel_right->meshing_state() != Surfel::MeshingState::kFree) {
          RemeshTrianglesAt(surfel_right, surfel_right->radius_squared());
        }
        if (surfel_left->meshing_state() != Surfel::MeshingState::kFree) {
          RemeshTrianglesAt(surfel_left, surfel_left->radius_squared());
        }
        remeshed = true;
        break;
      }
    }
    
    if (remeshed) {
      continue;
    }
    
    // NOTE: Could check for triangle intersections here, but that would be slow.
  }
  surfels_to_check_.clear();
}

void SurfelMeshing::Triangulate(bool force_debug) {
  // Maximum number of surfel neighbors to consider.
  constexpr int kMaxNeighbors = 64;
  
  // Global indices of the neighbor points, indexed by neighbor_index.
  u32 neighbor_indices[kMaxNeighbors];
  
  // Squared distances to neighbors, indexed by neighbor_index.
  float neighbor_distances_squared[kMaxNeighbors];
  
  // Temporary data of neighbor surfels.
  Neighbor neighbors[kMaxNeighbors];
  
  // Temporary data of selected neighbor surfels.
  Neighbor selected_neighbors[kMaxNeighbors];
  
  // Double edges in the neighborhood. Uses a separate indexing (edge_index)
  // from the neighbor points. Use edges[...].index to obtain the
  // neighbor_index to which the double edge is associated.
  std::vector<EdgeData> edges(4 * kMaxNeighbors);
  
  // Additional storage for temporary variables used by TryToAdvanceFront().
  bool gaps[kMaxNeighbors];
  bool skinny[kMaxNeighbors];
  float angle_diff[kMaxNeighbors];
  int angle_indices[kMaxNeighbors];
  bool to_erase[kMaxNeighbors];
  SkinnySurfel skinny_surfels[kMaxNeighbors];
  
  // Triangulate all surfels in surfels_to_remesh_.
  while (!surfels_to_remesh_.empty()) {
    u32 surfel_index = surfels_to_remesh_.back();
    surfels_to_remesh_.pop_back();
    
    if (!surfels_[surfel_index].can_be_remeshed() ||
        surfels_[surfel_index].meshing_state() == Surfel::MeshingState::kCompleted) {
      continue;
    }
    
    TriangulateSurfel(
        surfel_index,
        kMaxNeighbors,
        neighbor_indices,
        neighbor_distances_squared,
        neighbors,
        selected_neighbors,
        &edges,
        gaps,
        skinny,
        angle_diff,
        angle_indices,
        to_erase,
        skinny_surfels,
        force_debug,
        false);
    // CheckSurfelState(surfel_index);
  }
  
//   // DEBUG: Show color-coded surfel states.
//   LOG(INFO) << "DEBUG: Showing surfel states visualization. Free is green, front is yellow, completed is black.";
//   shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
//   ConvertToMesh3fCu8(visualization_mesh.get());
//   for (usize surfel_index = 0; surfel_index < surfels_.size(); ++ surfel_index) {
//     if (surfels_[surfel_index].meshing_state() == Surfel::MeshingState::kFree) {
//       (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(60, 255, 60);
//     } else if (surfels_[surfel_index].meshing_state() == Surfel::MeshingState::kFront) {
//       (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 255, 20);
//     } else if (surfels_[surfel_index].meshing_state() == Surfel::MeshingState::kCompleted) {
//      (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(0, 0, 0);
//     }
//   }
//   render_window_->UpdateVisualizationMesh(visualization_mesh);
//   std::getchar();
  
//   // Write debug info.
//   LOG(INFO) << "[debug stats:" << std::endl
//             << "    holes_closed_counter_ = " << holes_closed_counter_ << std::endl
//             << "    octree_.numerical_issue_counter() = " << octree_.numerical_issue_counter() << std::endl
//             << "    front_neighbors_too_far_away_counter_ = " << front_neighbors_too_far_away_counter_ << std::endl
//             << "    front_leads_to_completed_surfel_counter_ = " << front_leads_to_completed_surfel_counter_ << std::endl
//             << "    max_neighbor_count_exceeded_counter_ = " << max_neighbor_count_exceeded_counter_ << std::endl
//             << "    front_neighbors_not_visible_counter_ = " << front_neighbors_not_visible_counter_ << std::endl
//             << "    fronts_triangles_inconsistency_counter_ = " << fronts_triangles_inconsistency_counter_ << std::endl
//             << "    fronts_sharing_edge_counter_ = " << fronts_sharing_edge_counter_ << std::endl
//             << "    connected_to_surfel_without_suitable_front_counter_ = " << connected_to_surfel_without_suitable_front_counter_ << " ]";
}

double SurfelMeshing::FullRetriangulation() {
  for (usize i = 0, size = surfels_.size(); i < size; ++ i) {
    if (!surfels_[i].node()) {
      continue;
    }
    ResetSurfelToFree(i);
    surfels_[i].SetCanBeRemeshed(true);
    surfels_to_remesh_.clear();
  }
  
  // Make sure that no triangles are left.
  usize triangle_count = 0;
  for (usize t = 0; t < triangles_.size(); ++ t) {
    if (triangles_[t].IsValid()) {
      ++ triangle_count;
    }
  }
  if (triangle_count == 0) {
    LOG(INFO) << "FullRetriangulation() deleted all triangles";
  } else {
    LOG(FATAL) << "FullRetriangulation() failed to delete " << triangle_count << " triangles!";
  }
  
  first_new_surfel_index_ = 0;
  for (usize i = 0; i < surfels_.size(); ++ i) {
    if (!surfels_[i].node()) {
      continue;
    }
    surfels_to_remesh_.push_back(i);
  }
  
  ConditionalTimer full_retriangulation_timer("Full retriangulation");
  Triangulate();
  double seconds = full_retriangulation_timer.Stop(false);
  LOG(INFO) << "Full retriangulation took: " << seconds << " seconds";
  return seconds;
}

void SurfelMeshing::SetSurfelToRemesh(u32 surfel_index) {
  surfels_to_remesh_.push_back(surfel_index);
  surfels_[surfel_index].SetCanBeRemeshed(true);
  first_new_surfel_index_ = surfels_.size();
}

void SurfelMeshing::RemeshTrianglesAt(
    Surfel* surfel,
    float neighbor_search_radius_squared) {
  #ifdef KEEP_TRIANGLES_IN_OCTREE
    static vector<u32> old_triangles;  // TODO: Avoid static: make member
    // old_triangles.clear(); // done in the search function
    
    float box_half_extent = sqrtf(neighbor_search_radius_squared);
    Vec3f box_min = surfel->position() - Vec3f::Constant(box_half_extent);
    Vec3f box_max = surfel->position() + Vec3f::Constant(box_half_extent);
    octree_.FindNearestTrianglesIntersectingBox(box_min, box_max, &old_triangles);
    
    for (u32 triangle_index : old_triangles) {
      DeleteTriangleForRemeshing(triangle_index);
    }
  #else
    constexpr int kMaxSurfelCount = 64;
    
    // Slow since it stores the results in a vector first and de-duplicates them:
//     octree_.FindNearestTrianglesViaSurfels(surfel->position(), neighbor_search_radius_squared, kMaxSurfelCount, &old_triangles);
    
    surfel_distances_squared_.resize(kMaxSurfelCount);
    surfel_indices_.resize(kMaxSurfelCount);
    int surfel_count = octree_.FindNearestSurfelsWithinRadius<true, false>(
        surfel->position(), neighbor_search_radius_squared, kMaxSurfelCount,
        surfel_distances_squared_.data(), surfel_indices_.data());
    for (int i = 0; i < surfel_count; ++ i) {
      Surfel* surfel = &surfels_.at(surfel_indices_[i]);
      for (int t = 0, triangle_count = surfel->GetTriangleCount(); t < triangle_count; ++ t) {
        u32 triangle_index = surfel->GetTriangle(t);
        DeleteTriangleForRemeshing(triangle_index, surfel_indices_[i]);
      }
      
      // Reset the surfel.
      surfel->RemoveAllTriangles();
      surfel->fronts().clear();
      surfel->SetMeshingState(Surfel::MeshingState::kFree);
      surfel->SetCanBeReset(false);  // Avoid remeshing this surfel again in the same iteration if it is free now.
      surfels_to_remesh_.push_back(surfel_indices_[i]);
      surfel->SetCanBeRemeshed(true);
    }
  #endif
}

void SurfelMeshing::DeleteTriangleForRemeshing(
    u32 triangle_index) {
  SurfelTriangle* triangle = &triangles_[triangle_index];
  
  if (!triangle->IsValid()) {
    LOG(FATAL) << "DeleteTriangleForRemeshing: attempting to delete an invalid triangle";
    return;
  }
  
  ++ deleted_triangle_count_;
  
  // Remove it from the octree.
  #ifdef KEEP_TRIANGLES_IN_OCTREE
    octree_.RemoveTriangle(triangle_index, *triangle);
  #endif
  
  // Remove it from its corner surfels.
  triangle->DeleteFromSurfels(triangle_index, &surfels_);
  
  // Update the meshing fronts.
  bool have_debug = false;
  u32 debug_surfel_index;
  for (int index = 0; index < 3; ++ index) {
    // "Left" and "right" here are from the point of view of the vertex index,
    // looking into the triangle, with the triangle viewed from the top.
    int left_index = index - 1;
    int right_index = index + 1;
    if (left_index == -1) {
      left_index = 2;
    } else if (right_index == 3) {
      right_index = 0;
    }
    
    u32 surfel_index = triangle->index(index);
    bool debug = UpdateFrontsOnTriangleRemoval(surfel_index,
                                               triangle->index(left_index),
                                               triangle->index(right_index));
    if (debug) {
      debug_surfel_index = surfel_index;
      have_debug = true;
    }
    
    surfels_to_remesh_.push_back(surfel_index);
    surfels_[surfel_index].SetCanBeRemeshed(true);
  }
  
  // Finally, remove the triangle from the triangles list.
  triangle->MakeFreeListEntry(next_free_triangle_index_);
  next_free_triangle_index_ = triangle_index;
  
  if (have_debug) {
    LOG(INFO) << "DEBUG: Showing state after triangle removal. Debug vertex in red and all its front neighbors in yellow.";
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(debug_surfel_index).color() = Vec3u8(255, 60, 60);
    for (const Front& front : surfels_[debug_surfel_index].fronts()) {
      (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
      (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(255, 255, 60);
    }
    render_window_->CenterViewOn(surfels_[debug_surfel_index].position());
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
}

void SurfelMeshing::DeleteTriangleForRemeshing(
    u32 triangle_index,
    u32 reset_surfel_index) {
  SurfelTriangle* triangle = &triangles_[triangle_index];
  
  if (!triangle->IsValid()) {
    LOG(FATAL) << "DeleteTriangleForRemeshing: attempting to delete an invalid triangle";
    return;
  }
  
  ++ deleted_triangle_count_;
  
  // Remove it from the octree.
  #ifdef KEEP_TRIANGLES_IN_OCTREE
    octree_.RemoveTriangle(triangle_index, *triangle);
  #endif
  
  bool have_debug = false;
  u32 debug_surfel_index;
  for (int index = 0; index < 3; ++ index) {
    u32 surfel_index = triangle->index(index);
    if (surfel_index == reset_surfel_index) {
      continue;
    }
    
    // Remove the triangle from the surfel.
    surfels_[surfel_index].RemoveTriangle(triangle_index);
    
    // Update the meshing fronts.
    // "Left" and "right" here are from the point of view of the vertex index,
    // looking into the triangle, with the triangle viewed from the top.
    int left_index = index - 1;
    int right_index = index + 1;
    if (left_index == -1) {
      left_index = 2;
    } else if (right_index == 3) {
      right_index = 0;
    }
    bool debug = UpdateFrontsOnTriangleRemoval(surfel_index,
                                               triangle->index(left_index),
                                               triangle->index(right_index));
    if (debug) {
      debug_surfel_index = surfel_index;
      have_debug = true;
    }
    
    surfels_to_remesh_.push_back(surfel_index);
    surfels_[surfel_index].SetCanBeRemeshed(true);
    
    // CheckSurfelState(surfel_index);
  }
  
  // Finally, remove the triangle from the triangles list.
  triangle->MakeFreeListEntry(next_free_triangle_index_);
  next_free_triangle_index_ = triangle_index;
  
  if (have_debug) {
    LOG(INFO) << "DEBUG: Showing state after triangle removal. Debug vertex in red and all its front neighbors in yellow.";
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(debug_surfel_index).color() = Vec3u8(255, 60, 60);
    for (const Front& front : surfels_[debug_surfel_index].fronts()) {
      (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
      (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(255, 255, 60);
    }
    render_window_->CenterViewOn(surfels_[debug_surfel_index].position());
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
}

bool SurfelMeshing::UpdateFrontsOnTriangleRemoval(
    u32 surfel_index, u32 left_surfel_index, u32 right_surfel_index) {
  Surfel* surfel = &surfels_[surfel_index];
  // Except in error cases, the surfel should either have fronts or get one,
  // so retrieve a pointer to them / create them.
  vector<Front>* surfel_fronts = &surfel->fronts();
  
  bool debug = false;
  // Uncomment this to debug this function for a specific surfel:
  // debug |= surfel_index == 118504;
  if (debug) {
    LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval called.";
  }
  
  // If the surfel was completed, make it a front surfel and add the front.
  if (surfel->meshing_state() == Surfel::MeshingState::kCompleted) {
    if (!surfel_fronts->empty()) {
      LOG(ERROR) << "Error: Found completed surfel with non-empty front list";
      surfel_fronts->clear();
    }
    surfel_fronts->emplace_back(left_surfel_index, right_surfel_index);
    surfel->SetMeshingState(Surfel::MeshingState::kFront);
    if (debug) {
      LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval changed completed surfel to front.";
    }
    // CheckSurfelState(surfel_index);
    return debug;
  }
  
  // If the surfel was free, something is wrong since a triangle is supposed to
  // be removed from it just now.
  if (surfel->meshing_state() == Surfel::MeshingState::kFree) {
    ++ fronts_triangles_inconsistency_counter_;
//     LOG(ERROR) << "Attempting to update the fronts of a free surfel from which a triangle is being removed. Contradiction.";
    if (debug) {
      LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval was called on a free surfel.";
    }
    // CheckSurfelState(surfel_index);
    return debug;
  }
  
  // At this point, the surfel is a front or boundary surfel.
  if (debug) {
    LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval called on a front or boundary surfel.";
    
    LOG(INFO) << "DEBUG: Showing front neighbors. surfel_index is red, left_surfel_index is green, right_surfel_index is blue.";
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
    (*visualization_mesh->vertices_mutable())->at(left_surfel_index).color() = Vec3u8(60, 255, 60);
    (*visualization_mesh->vertices_mutable())->at(right_surfel_index).color() = Vec3u8(60, 60, 255);
    
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
  
  // Check the existing fronts for having at least one edge in common with the
  // two edges of the removed triangle next to the surfel.
  bool matching_front_found = false;
  int right_matching_front_index = -1;  // Index of a front matching the right surfel.
  int left_matching_front_index = -1;  // Index of a front matching the left surfel.
  for (int i = 0; i < static_cast<int>(surfel_fronts->size()); ++ i) {
    Front* front = &surfel_fronts->at(i);
    
    // If a front directly matches the left and right surfels, remove it and set
    // the surfel to free if no other front is left.
    // NOTE: Not sure whether it is a good idea to also check the wrong sided variant.
    // NOTE: Theoretically this should only happen if there is exactly one front.
    if ((front->left == right_surfel_index && front->right == left_surfel_index) ||  // correctly sided variant.
        (/*front->left == left_surfel_index && front->right == right_surfel_index*/ false)) {  // wrongly sided variant.
      surfel_fronts->erase(surfel_fronts->begin() + i);
      matching_front_found = true;
    
      if (debug) {
        LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval removed a matching front.";
      }
      
      -- i;
      continue;
    }
    
    // Check if the front matches only one of the edges.
    if (front->left == right_surfel_index ||   // correctly sided variant.
        /*front->right == right_surfel_index*/ false) {  // wrongly sided variant.
      if (right_matching_front_index >= 0) {
        // Already found a front matching the right surfel. This should not
        // happen in theory.
        ++ fronts_sharing_edge_counter_;
//         LOG(WARNING) << "Found two fronts sharing an edge while removing a triangle. Deleting one of them.";
        surfel_fronts->erase(surfel_fronts->begin() + right_matching_front_index);
        -- i;
        if (left_matching_front_index > right_matching_front_index) {
          -- left_matching_front_index;
        }
      }
      right_matching_front_index = i;
      matching_front_found = true;
    } else if (front->right == left_surfel_index ||  // correctly sided variant.
               /*front->left == left_surfel_index*/ false) {   // wrongly sided variant.
      if (left_matching_front_index >= 0) {
        // Already found a front matching the left surfel. This should not
        // happen in theory.
        ++ fronts_sharing_edge_counter_;
//         LOG(WARNING) << "Found two fronts sharing an edge while removing a triangle. Deleting one of them.";
        surfel_fronts->erase(surfel_fronts->begin() + left_matching_front_index);
        -- i;
        if (right_matching_front_index > left_matching_front_index) {
          -- right_matching_front_index;
        }
      }
      left_matching_front_index = i;
      matching_front_found = true;
    }
  }
  
  if (debug) {
    LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval has left_matching_front_index: " <<
                 left_matching_front_index << ", right_matching_front_index: " <<
                 right_matching_front_index;
  }
  
  // Handle the case of fronts sharing a single edge with the two relevant ones
  // of the removed triangle.
  if (left_matching_front_index >= 0) {
    // CHECK_NE(left_matching_front_index, right_matching_front_index);
    
    Front* left_front = &surfel_fronts->at(left_matching_front_index);
    if (right_matching_front_index == -1) {
      // There is no hole after the right surfel, so extend the hole at the left
      // one until the right one.
      if (left_front->right == left_surfel_index) {
        // Correct sidedness.
        left_front->right = right_surfel_index;
      } else {
        // Wrong sidedness.
        left_front->left = right_surfel_index;
      }
    } else {
      // There is a hole to the left and one to the right. Merge them into a
      // single hole.
      Front* right_front = &surfel_fronts->at(right_matching_front_index);
      if (left_front->right == left_surfel_index) {
        // Correct sidedness of left_front.
        if (right_front->left == right_surfel_index) {
          // Correct sidedness of both fronts.
          left_front->right = right_front->right;
          surfel_fronts->erase(surfel_fronts->begin() + right_matching_front_index);
        } else {
          // Correct sidedness of left_front, but wrong sidedness of
          // right_front. Screwed up case.
          left_front->right = right_front->left;
          surfel_fronts->erase(surfel_fronts->begin() + right_matching_front_index);
        }
      } else {
        // Wrong sidedness of left_front.
        if (right_front->left == right_surfel_index) {
          // Correct sidedness of right_front, but wrong sidedness of
          // left_front. Screwed up case.
          right_front->left = left_front->right;
          surfel_fronts->erase(surfel_fronts->begin() + left_matching_front_index);
        } else {
          // Wrong sidedness of both fronts.
          right_front->right = left_front->right;
          surfel_fronts->erase(surfel_fronts->begin() + left_matching_front_index);
        }
      }
    }
  } else if (right_matching_front_index >= 0) {
    Front* right_front = &surfel_fronts->at(right_matching_front_index);
    // There is no hole after the left surfel, so extend the hole at the right
    // one until the left one.
    if (right_front->left == right_surfel_index) {
      // Correct sidedness.
      right_front->left = left_surfel_index;
    } else {
      // Wrong sidedness.
      right_front->right = left_surfel_index;
    }
  }
  
  if (debug) {
    LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval has matching_front_found: " << matching_front_found;
  }
  
  if (matching_front_found) {
    if (surfel_fronts->empty()) {
      if (surfel->GetTriangleCount() > 0) {
        //LOG(ERROR) << "Error: deleted all fronts for a surfel, but it has triangle references left. Not resolved yet.";
        // TODO: Re-build the fronts based on the triangle references (as those are more trustworthy), or reset the surfel?
      }
      surfel->SetMeshingState(Surfel::MeshingState::kFree);
      surfel->SetCanBeReset(false);  // Avoid remeshing this surfel again in the same iteration if it is free now.
      if (debug) {
        LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval set the state to free.";
      }
    } else {
      if (surfel->GetTriangleCount() == 0) {
        // Error case: fronts and triangles mismatch. Let the triangles take precedence since they are more trustworthy.
//         LOG(ERROR) << "Error: have front(s) for a surfel, but it has no triangle references.";
        ++ fronts_triangles_inconsistency_counter_;
        surfel->fronts().clear();
        surfel->SetMeshingState(Surfel::MeshingState::kFree);
        surfel->SetCanBeReset(false);  // Avoid remeshing this surfel again in the same iteration if it is free now.
        if (debug) {
          LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval set the state to free in an error case (!).";
        }
      } else {
        surfel->SetMeshingState(Surfel::MeshingState::kFront);
        if (debug) {
          LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval set the state to front. Number of fronts: " << surfel_fronts->size();
        }
      }
    }
    // CheckSurfelState(surfel_index);
    return debug;
  }
  
  // The triangle removal creates a new hole since it is not adjacent to an
  // existing front. Create a new front.
  surfel_fronts->emplace_back(left_surfel_index, right_surfel_index);
  surfel->SetMeshingState(Surfel::MeshingState::kFront);
  
  if (debug) {
    LOG(INFO) << "DEBUG: UpdateFrontsOnTriangleRemoval created a new front since no matching front was found, set the state to front. Number of fronts: " << surfel_fronts->size();
  }
  // CheckSurfelState(surfel_index);
  return debug;
}

void SurfelMeshing::ProjectNeighborsAndTestVisibility(
    u32 surfel_index,
    Surfel* surfel,
    const Vec3f& surfel_proj,
    int neighbor_count,
    u32* neighbor_indices,
    Neighbor* neighbors,
    const Vec3f& u,
    const Vec3f& v,
    std::vector<EdgeData>* edges,
    bool debug) {
  Vec3f offset;
  u32 edge_count = 0;
  for (int neighbor_index = 1; neighbor_index < neighbor_count;
        ++ neighbor_index) {  // Neighbor with index 0 is the surfel itself
    usize neighbor_surfel_index = neighbor_indices[neighbor_index];
    const Surfel& neighbor_surfel = surfels_[neighbor_surfel_index];
    Neighbor* neighbor = &neighbors[neighbor_index];
    
    neighbor->surfel_index = neighbor_surfel_index;
    neighbor->nearest_neighbor_index = neighbor_index;
    neighbor->visible = neighbor_surfel.meshing_state() != Surfel::MeshingState::kCompleted;
    if (neighbor->visible) {
      offset = neighbor_surfel.position() - surfel_proj;
      neighbor->uv = Vec2f(offset.dot(u), offset.dot(v));
      neighbor->angle =
          ApproxAtan2(neighbor->uv.coeff(1), neighbor->uv.coeff(0));
    }
    
    if (debug && !neighbor->visible) {
      LOG(INFO) << "Neighbor " << neighbor_index << " (surfel " << neighbors[neighbor_index].surfel_index << ") not visible because it is completed.";
    }
    
    // Estimate whether the reference surfel and the neighbor are on the
    // same side of the surface, or on opposite sides. Also apply the
    // max-angle criterion.
    bool same_side = true;
    if (neighbor->visible) {  // If the surfel is completed (--> visible set to false at this point), do not determine same_side.
      float cosine_angle = surfel->normal().dot(neighbor_surfel.normal());
      // If this is set, only surfels which are estimated to be on the same side of the surface will be connected:
      constexpr bool kEnforceNormalConsistency = true;
      if (!kEnforceNormalConsistency && cosine_angle < 0) {
        cosine_angle = 1 - cosine_angle;
      }
      if (cosine_angle < cos_max_angle_between_normals_) {
        neighbor->visible = false;
        same_side = false;
        if (debug) {
          LOG(INFO) << "Neighbor " << neighbor_index << " (surfel " << neighbors[neighbor_index].surfel_index << ") not visible because it is on the back side (angle: "
                    << (180.f / M_PI * acos(cosine_angle)) << ", max angle threshold: " << (180.f / M_PI * acos(cos_max_angle_between_normals_)) << ").";
        }
      }
    }
    
    if (same_side &&
        (neighbor_surfel.meshing_state() == Surfel::MeshingState::kFront)) {
      // Save the edges for visibility checking.
      const std::vector<Front>& neighbor_fronts = neighbor_surfel.fronts();
      bool reference_is_behind_all_fronts = true;  // Only if the reference surfel is behind all of the neighbor's fronts, we can be sure that the neighbor is not visible.
      for (const Front& front : neighbor_fronts) {
        if (edges->size() <= edge_count + 1) {
          edges->resize(2 * edges->size());
        }
        
        // NOTE: Could also use a multimap to determine whether edges are
        //       already present.
        bool have_left_edge = front.left == surfel_index;
        bool have_right_edge = front.right == surfel_index;
        for (u32 edge_index = 0; edge_index < edge_count; ++ edge_index) {
          if (edges->at(edge_index).end_index == neighbor_surfel_index) {
            u32 start_index = neighbors[edges->at(edge_index).neighbor_index].surfel_index;
            if (start_index == front.left) {
              have_left_edge = true;
              if (have_right_edge) {
                break;
              }
            } else if (start_index == front.right) {
              have_right_edge = true;
              if (have_left_edge) {
                break;
              }
            }
          }
        }
        
        offset = surfels_[front.left].position() - surfel_proj;
        Vec2f left_end_pos = Vec2f(offset.dot(u), offset.dot(v));
        if (!have_left_edge) {
          EdgeData* left_edge = &edges->at(edge_count);
          ++ edge_count;
          
          left_edge->neighbor_index = neighbor_index;
          left_edge->end_index = front.left;
          left_edge->end_pos = left_end_pos;
        }
        
        offset = surfels_[front.right].position() - surfel_proj;
        Vec2f right_end_pos = Vec2f(offset.dot(u), offset.dot(v));
        if (!have_right_edge) {
          EdgeData* right_edge = &edges->at(edge_count);
          ++ edge_count;
          
          right_edge->neighbor_index = neighbor_index;
          right_edge->end_index = front.right;
          right_edge->end_pos = right_end_pos;
        }
        
        
        // Pruning by first visibility criterion (for all neighbors that are
        // front / boundary surfels and which are not directly connected to the
        // reference surfel via a front edge): If the reference surfel is behind
        // the neighbor's front edges, the neighbor is not visible.
        if (neighbor->visible &&
            reference_is_behind_all_fronts) {
          if (front.left == surfel_index ||
              front.right == surfel_index) {
            reference_is_behind_all_fronts = false;
          } else {
            // Compute angle from neighbor to reference surfel.
            float angle_r = neighbor->angle + M_PI;
            if (angle_r >= M_PI) {
              angle_r -= 2 * M_PI;
            }
            
            float angle_left = ApproxAtan2(left_end_pos.coeff(1) - neighbor->uv.coeff(1),
                                     left_end_pos.coeff(0) - neighbor->uv.coeff(0));
            float angle_right = ApproxAtan2(right_end_pos.coeff(1) - neighbor->uv.coeff(1),
                                      right_end_pos.coeff(0) - neighbor->uv.coeff(0));
            
            if (angle_left <= angle_right) {
              // The space from angle_right to M_PI, and -M_PI to angle_left is occupied.
              if (!(angle_r < angle_left || angle_right < angle_r)) {
                reference_is_behind_all_fronts = false;
              }
            } else {
              // The space from angle_right to angle_left is occupied.
              if (!(angle_right < angle_r && angle_r < angle_left)) {
                reference_is_behind_all_fronts = false;
              }
            }
          }
        }  // Application of pruning by first visibility criterion.
      }  // Loop over the neighbor's fronts.
      
      if (reference_is_behind_all_fronts) {
        neighbor->visible = false;
        if (debug) {
          LOG(INFO) << "Neighbor " << neighbor_index << " (surfel " << neighbors[neighbor_index].surfel_index << ") not visible because the reference surfel is behind all of its fronts.";
          for (const Front& front : neighbor_fronts) {
            LOG(INFO) << "  Neighbor front left: " << front.left;
            LOG(INFO) << "  Neighbor front right: " << front.right;
          }
        }
      }
    }  // Special case handling for front and boundary points on the same side.
  }  // Loop over neighbors.
  neighbors[0].visible = false;  // Set the "neighbor" corresponding to the reference surfel to not visible.
  
  // Second part of visibility pruning: test intersections of the viewing ray
  // with front edges.
  // TODO: De-duplicate the edges to improve performance?
  for (u32 neighbor_index = 1; neighbor_index < static_cast<u32>(neighbor_count);
        ++ neighbor_index) {  // Neighbor with index 0 is the surfel itself
    usize neighbor_surfel_index = neighbor_indices[neighbor_index];
    Neighbor* neighbor = &neighbors[neighbor_index];
    if (!neighbor->visible /*||
        surfel_front_neighbors.count(neighbor_surfel_index) > 0*/) {
      continue;
    }
    
    for (u32 edge_index = 0; edge_index < edge_count; ++ edge_index) {
      const EdgeData& double_edge = edges->at(edge_index);
      if (double_edge.neighbor_index == neighbor_index ||
          double_edge.end_index == neighbor_surfel_index) {
        continue;
      }
      
      if (!IsVisible(neighbor->uv,  // neighbor point
                     neighbors[double_edge.neighbor_index].uv,  // edge start
                     double_edge.end_pos)) {  // edge end
        neighbor->visible = false;
        if (debug) {
          LOG(INFO) << "Neighbor " << neighbor_index << " (surfel " << neighbor_surfel_index << ") not visible because of front intersection.";
        }
        break;
      }
    }
  }
}

void SurfelMeshing::TryToAdvanceFront(
    u32 surfel_index, std::vector<Front>* surfel_front, int neighbor_count, u32* neighbor_indices,
    Neighbor* neighbors, std::vector<EdgeData>* edges, Neighbor* selected_neighbors,
    bool* gaps, bool* skinny, float* angle_diff, int* /*angle_indices*/, bool* to_erase, SkinnySurfel* skinny_surfels,
    int max_neighbor_count, bool no_surfel_resets, bool debug) {
  Surfel* surfel = &surfels_[surfel_index];
  
  // Define a coordinate system on the plane defined by the surfel normal,
  // going through the origin.
  const Vec3f& normal = surfel->normal();
  Vec3f v = normal.unitOrthogonal();
  Vec3f u = normal.cross(v);
  
  // Project the surfel onto this plane.
  Vec3f surfel_proj = surfel->position() - normal.dot(surfel->position()) * normal;  // NOTE: "proj_qp_" in PCL code
  
  new_fronts_.clear();
  for (usize front_index = 0; front_index < surfel_front->size(); ++ front_index) {
    const Front& front = surfel_front->at(front_index);
    
    // Project the neighbors onto the plane and save their coordinates,
    // angles, and adjacent boundary edges.
    ProjectNeighborsAndTestVisibility(
        surfel_index, surfel, surfel_proj, neighbor_count, neighbor_indices,
        neighbors, u, v, edges, debug);
    
    // DEBUG: Show all neighbors.
    if (debug) {
      LOG(INFO) << "DEBUG: Showing all neighbors. Reference surfel is red, front neighbors are yellow, other neighbors are blue.";
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      ConvertToMesh3fCu8(visualization_mesh.get());
      
      (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
      for (int i = 1; i < neighbor_count; ++ i) {  // surfel_index is not set for neighbor 0, which is the reference surfel.
        (*visualization_mesh->vertices_mutable())->at(neighbors[i].surfel_index).color() = Vec3u8(60, 60, 255);
      }
      if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
        for (const Front& debug_front : surfel->fronts()) {
          (*visualization_mesh->vertices_mutable())->at(debug_front.left).color() = Vec3u8(255, 255, 60);
          (*visualization_mesh->vertices_mutable())->at(debug_front.right).color() = Vec3u8(255, 255, 60);
        }
      }
      
      render_window_->UpdateVisualizationMesh(visualization_mesh);
      std::getchar();
    }
    
    // DEBUG: Show all neighbors classified as visible.
    if (debug) {
      LOG(INFO) << "DEBUG: Showing neighbors classified as visible. Reference surfel is red, front neighbors are yellow, other neighbors are blue.";
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      ConvertToMesh3fCu8(visualization_mesh.get());
      
      (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
      for (int i = 1; i < neighbor_count; ++ i) {
        if (neighbors[i].visible) {
          (*visualization_mesh->vertices_mutable())->at(neighbors[i].surfel_index).color() = Vec3u8(60, 60, 255);
        }
      }
      if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
        for (const Front& debug_front : surfel->fronts()) {
          (*visualization_mesh->vertices_mutable())->at(debug_front.left).color() = Vec3u8(255, 255, 60);
          (*visualization_mesh->vertices_mutable())->at(debug_front.right).color() = Vec3u8(255, 255, 60);
        }
      }
      
      render_window_->UpdateVisualizationMesh(visualization_mesh);
      std::getchar();
    }
  
    // Find the neighbor indices of the front neighbors.
    int left = -1, right = -1;
    for (int i = 1; i < neighbor_count; ++ i) {  // Neighbor 0 is the surfel itself
      if (front.left == neighbors[i].surfel_index) {
        left = i;
        for (i = i + 1; i < neighbor_count; ++ i) {
          if (front.right == neighbors[i].surfel_index) {
            right = i;
            break;
          }
        }
        break;
      } else if (front.right == neighbors[i].surfel_index) {
        right = i;
        for (i = i + 1; i < neighbor_count; ++ i) {
          if (front.left == neighbors[i].surfel_index) {
            left = i;
            break;
          }
        }
        break;
      }
    }
    
    // This case should never happen for the first checked reasons as the
    // front-connected neighbors were checked to be included before,
    // unless the maximum neighbor count is exceeded.
    // Due to position changes of the surfels, the neighbors can be invisible.
    // They can also be estimated to be invisible because their surfel is
    // mistakenly set to completed, or because their surfel normals differ more
    // than the threshold from the current surfel's normal, or they can be
    // hidden by nearby triangles given the current surfel's normal.
    if (left < 0 || right < 0 ||
        !neighbors[left].visible || !neighbors[right].visible) {
      if (neighbor_count == max_neighbor_count) {
        ++ max_neighbor_count_exceeded_counter_;
        //LOG(WARNING) << "Did not find the front neighbors among the neighbor set, presumably because kMaxNeighbors was reached.";
      } else {
        if (left >= 0 && right >= 0) {
          if (debug) {
            LOG(INFO) << "Front neighbors are not visible ...";
          }
          ++ front_neighbors_not_visible_counter_;
          
          // Force visibility to true to hopefully get more complete meshes.
          neighbors[left].visible = true;
          neighbors[right].visible = true;
          goto continue_meshing;
        } else {
          // As far as I know, this case would be an actual bug.
          // TODO: debug this case
          //LOG(WARNING) << "Front neighbors not found among neighbors, but max neighbor count not exceeded. Bug?";
        }
//         LOG(WARNING) << "Front neighbors are not found in neighbor list or not visible."
//                       << " left: " << left << " right: " << right
//                       << " front.left: " << front.left << " front.right: " << front.right
//                       << " neighbors[left].visible: " << ((left >= 0) ? neighbors[left].visible : false)
//                       << " neighbors[left] state: " << ((left >= 0) ? static_cast<int>(surfels_[neighbors[left].surfel_index].meshing_state()) : -1)
//                       << " neighbors[right].visible: " << ((right >= 0) ? neighbors[right].visible : false)
//                       << " neighbors[right] state: " << ((right >= 0) ? static_cast<int>(surfels_[neighbors[right].surfel_index].meshing_state()) : -1)
//                       << " neighbor count: " << neighbor_count
//                       << " max neighbor count: " << max_neighbor_count
//                       << " reference surfel index: " << surfel_index;
        
        // Try to reduce the damage: reset the surfel.
        
        // NOTE: Currently the strategy here is to delete all triangles
        //       adjacent to the surfel and re-triangulate. As an
        //       alternative one could only re-build the fronts instead.
        // Avoid endless loops by checking can_be_reset().
        if (surfel->can_be_reset() && !no_surfel_resets) {
          if (debug) {
            LOG(INFO) << "Calling ResetSurfelToFree().";
          }
          ResetSurfelToFree(surfel_index);
          return;
        } else if (debug) {
          LOG(INFO) << "Not resetting the surfel to free since can_be_reset() returned false.";
        }
        
//         LOG(INFO) << "DEBUG: Showing visualization. Reference surfel is red, left is green, right is blue.";
//         shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
//         ConvertToMesh3fCu8(visualization_mesh.get());
//         
//         (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
//         if (left >= 0) {
//           (*visualization_mesh->vertices_mutable())->at(neighbors[left].surfel_index).color() = Vec3u8(60, 255, 60);
//         }
//         if (right >= 0) {
//           (*visualization_mesh->vertices_mutable())->at(neighbors[right].surfel_index).color() = Vec3u8(60, 60, 255);
//         }
//         
//         render_window_->CenterViewOn(surfel->position());
//         render_window_->UpdateVisualizationMesh(visualization_mesh);
//         std::getchar();
      }
      surfel->SetMeshingState(Surfel::MeshingState::kFront);
      continue;
    }
continue_meshing:;
    
    bool have_wrap_around = neighbors[left].angle > neighbors[right].angle;
    
    // Copy all relevant neighbors (i.e., visible neighbors between left and
    // right) into a separate array. Wrap the angles if required.
    // TODO: Would it be faster to only create indices into the existing array?
    float wrap_angle = neighbors[left].angle;
    u32 selected_neighbor_count = 1;
    for (int neighbor_index = 1; neighbor_index < neighbor_count;
          ++ neighbor_index) {  // Neighbor with index 0 is the surfel itself
      // Add only relevant surfels to the selected surfels.
      if (neighbor_index != left && neighbor_index != right &&
          neighbors[neighbor_index].visible &&
          ((have_wrap_around && (neighbors[neighbor_index].angle >= neighbors[left].angle ||
                                 neighbors[neighbor_index].angle <= neighbors[right].angle)) ||
            (!have_wrap_around && (neighbors[neighbor_index].angle >= neighbors[left].angle &&
                                   neighbors[neighbor_index].angle <= neighbors[right].angle)))) {
        selected_neighbors[selected_neighbor_count] = neighbors[neighbor_index];
        selected_neighbors[selected_neighbor_count].angle += (selected_neighbors[selected_neighbor_count].angle < wrap_angle) ? (2 * M_PI) : 0;
        if (debug) {
          LOG(INFO) << "  selected neighbor: angle: " << neighbors[neighbor_index].angle << " surfel index: " << neighbors[neighbor_index].surfel_index;
        }
        ++ selected_neighbor_count;
      }
    }
    selected_neighbors[0] = neighbors[left];
    // No angle wrapping necessary for the left neighbor as the condition will never be true.
    selected_neighbors[selected_neighbor_count] = neighbors[right];
    selected_neighbors[selected_neighbor_count].angle += (selected_neighbors[selected_neighbor_count].angle < wrap_angle) ? (2 * M_PI) : 0;
    ++ selected_neighbor_count;
    
    // Sort the array of relevant neighbors according to the angle, such that
    // left is the first element and right is the last element.
    std::sort(selected_neighbors + 1,
              selected_neighbors + selected_neighbor_count - 1,
              NeighborSortByAngleAsc());
    
    // Collect information about the angles (indexed by their original,
    // angle/visibility-sorted index):
    // angle_diff[i] is the angle difference to the next neighbor.
    // skinny[i] is set if the angle to the next neighbor is too small.
    // gaps[i] is set if the angle to the next neighbor is too large.
    bool is_boundary = false;
    int skinny_surfel_count = 0;
    for (int i = 0; i < static_cast<int>(selected_neighbor_count) - 1; ++ i) {
      float this_angle = selected_neighbors[i].angle;
      float next_angle = selected_neighbors[i + 1].angle;
      angle_diff[i] = next_angle - this_angle;
      
      if (angle_diff[i] < min_triangle_angle_) {
        skinny[i] = true;
        gaps[i] = false;
        
        if (i > 0 && !skinny[i - 1]) {
          // Add i as skinny surfel
          skinny_surfels[skinny_surfel_count].selected_neighbor_index = i;
          skinny_surfels[skinny_surfel_count].nearest_neighbor_index = selected_neighbors[i].nearest_neighbor_index;
          ++ skinny_surfel_count;
        }
        if (i < static_cast<int>(selected_neighbor_count) - 2) {
          // Add i + 1 as skinny surfel
          skinny_surfels[skinny_surfel_count].selected_neighbor_index = i + 1;
          skinny_surfels[skinny_surfel_count].nearest_neighbor_index = selected_neighbors[i + 1].nearest_neighbor_index;
          ++ skinny_surfel_count;
        }
      } else if (angle_diff[i] > max_triangle_angle_) {
        skinny[i] = false;
        gaps[i] = is_boundary = true;
      } else {
        skinny[i] = false;
        gaps[i] = false;
      }
      
      if (debug) {
        LOG(INFO) << "DEBUG: for surfel index " << selected_neighbors[i].surfel_index << ": "
                  << " angle: " << selected_neighbors[i].angle
                  << ", diff: " << angle_diff[i]
                  << ", skinny: " << (skinny[i] ? "true" : "false")
                  << ", gap: " << (gaps[i] ? "true" : "false");
      }
    }
    skinny[selected_neighbor_count - 1] = false;
    gaps[selected_neighbor_count - 1] = false;
    
    // DEBUG: Show selected_neighbors classified as visible.
    if (debug) {
      constexpr int kDebugColorCount = 10;
      constexpr const char* kDebugColorNames[kDebugColorCount] = {
        "green",
        "yellow",
        "blue",
        "purple",
        "brown",
        "cyan",
        "light_blue",
        "orange",
        "beige",
        "pink"
      };
      const Vec3u8 kDebugColors[kDebugColorCount] = {
        Vec3u8(60, 255, 60),
        Vec3u8(255, 255, 0),
        Vec3u8(0, 0, 255),
        Vec3u8(255, 0, 255),
        Vec3u8(208, 113, 0),
        Vec3u8(0, 255, 255),
        Vec3u8(114, 127, 255),
        Vec3u8(255, 162, 0),
        Vec3u8(255, 217, 167),
        Vec3u8(255, 129, 189)
      };
      
      LOG(INFO) << "DEBUG: Showing selected_neighbors classified as visible for the current front. Reference surfel is red, front neighbors are yellow, visible neighbors are green.";
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      ConvertToMesh3fCu8(visualization_mesh.get());
      
      shared_ptr<Point3fC3u8Cloud> visualization_cloud(new Point3fC3u8Cloud());
      ConvertToPoint3fC3u8Cloud(visualization_cloud.get());
      
      (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
      for (int i = 0; i < static_cast<int>(selected_neighbor_count); ++ i) {
        LOG(INFO) << "  DEBUG: Visible neighbor " << i << ": " << selected_neighbors[i].surfel_index
                  << ((front.left == selected_neighbors[i].surfel_index) ? " (left)" : "")
                  << ((front.right == selected_neighbors[i].surfel_index) ? " (right)" : "")
                  << (skinny[i] ? " [skinny]" : "")
                  << (gaps[i] ? " [gap]" : "")
                  << " - " << kDebugColorNames[i % kDebugColorCount]
                  << ((i >= kDebugColorCount) ? " (warning: duplicate color, consider adding more)" : "");
        (*visualization_mesh->vertices_mutable())->at(selected_neighbors[i].surfel_index).color() = kDebugColors[i % kDebugColorCount];
        visualization_cloud->at(selected_neighbors[i].surfel_index).color() = kDebugColors[i % kDebugColorCount];
      }
      
      render_window_->UpdateVisualizationMesh(visualization_mesh);
      render_window_->UpdateVisualizationCloud(visualization_cloud);
      std::getchar();
    }
    
    // If possible, discard points that would form skinny triangles.
    if (skinny_surfel_count > 0) {
      u32 erase_count = 0;
      for (u32 i = 0; i < selected_neighbor_count; ++ i) {
        to_erase[i] = false;
      }
      
      // Sort the neighbors by distance to the center surfel and try to delete the
      // far-away neighbors first in order to avoid "spikes" in the resulting
      // triangulation.
      // NOTE: In principle, we already have this sorting from the nearest
      //       neighbor list, however it is discarded when creating
      //       selected_neighbors. Could try to use this to gain speed: store
      //       selected_neighbor_index in the selected_neighbors and after the
      //       sorting build an index list from these indices. This will make
      //       the selected_neighbors larger however, so sorting them will be
      //       slower.
      std::sort(skinny_surfels, skinny_surfels + skinny_surfel_count, SkinnySurfelSortByDistance());
      
      for (int skinny_surfel_index = 0; skinny_surfel_index < skinny_surfel_count; ++ skinny_surfel_index) {
        int considered_neighbor = skinny_surfels[skinny_surfel_index].selected_neighbor_index;
        if (debug) {
          LOG(INFO) << "Skinny surfel handling considers neighbor " << considered_neighbor << " ...";
        }
        
        // Find left neighbor surfel index.
        int left_neighbor = considered_neighbor - 1;
        while (to_erase[left_neighbor]) {
          -- left_neighbor;
        }
        
        const int left_triangle_index = left_neighbor;
        const int right_triangle_index = considered_neighbor;
        
        // If the neighboring triangles are not skinny anymore, there is nothing left to do.
        if (!skinny[left_triangle_index] && !skinny[right_triangle_index]) {
          continue;
        }
        
        // One of the neighboring triangles is still skinny.
        // If there is a gap nearby, simply extend the gap to remove the problem.
        if (gaps[left_triangle_index]) {
          gaps[right_triangle_index] = true;
          skinny[right_triangle_index] = false;
          // No need to erase - try to avoid the copy done by the erasing.
//           to_erase[considered_neighbor] = true;
//           ++ erase_count;
          if (debug) {
            LOG(INFO) << "  Extending a gap before neighbor " << considered_neighbor << ". Setting gaps[" << right_triangle_index << "] = true.";
          }
          continue;
        } else if (gaps[right_triangle_index]) {
          gaps[left_triangle_index] = true;
          skinny[left_triangle_index] = false;
          // No need to erase - try to avoid the copy done by the erasing.
//           to_erase[considered_neighbor] = true;
//           ++ erase_count;
          if (debug) {
            LOG(INFO) << "  Extending a gap after neighbor " << considered_neighbor << ". Setting gaps[" << left_triangle_index << "] = true.";
          }
          continue;
        }
        
        // Find right neighbor surfel index.
        int right_neighbor = considered_neighbor + 1;
        while (to_erase[right_neighbor]) {
          ++ right_neighbor;
        }
        
        // Check whether the angle of the resulting triangle would become too
        // large.
        float merged_angle = angle_diff[left_triangle_index] + angle_diff[right_triangle_index];
        if (merged_angle > max_triangle_angle_) {
          if (debug) {
            LOG(INFO) << "Cannot delete neighbor " << considered_neighbor << " since the resulting angle would be too large.";
          }
          continue;
        }
        
        // Check whether the new triangle would contain any of the surfels that
        // would remain free.
        Vec2f X, S1, S2;
        
        Vec3f offset = surfels_[selected_neighbors[left_triangle_index].surfel_index].position() - surfel_proj;
        S1 = Vec2f(offset.dot(u), offset.dot(v));
        u32 left_corner_nn_index = selected_neighbors[left_triangle_index].nearest_neighbor_index;
        offset = surfels_[selected_neighbors[right_neighbor].surfel_index].position() - surfel_proj;
        S2 = Vec2f(offset.dot(u), offset.dot(v));
        u32 right_corner_nn_index = selected_neighbors[right_neighbor].nearest_neighbor_index;
        
        // Loop over the other surfels in the direction of the potential new
        // triangle and check that they are not within that triangle.
        bool can_delete = true;
        for (int k = left_triangle_index + 1; k < right_neighbor; ++ k) {
          // If this surfel is farther away than both corners, it is impossible
          // that it can be within the remaining triangle.
          if (selected_neighbors[k].nearest_neighbor_index > left_corner_nn_index &&
              selected_neighbors[k].nearest_neighbor_index > right_corner_nn_index) {
            continue;
          }
          
          // Check whether k would be included in the triangle spanned by
          // the reference surfel, the last surfel to keep, and the next
          // surfel after the one considered for discarding. If k is inside,
          // we do not discard the considered surfel as the resulting
          // triangulation might include the triangle tested above, which
          // neglects k.
          offset = surfels_[selected_neighbors[k].surfel_index].position() - surfel_proj;
          X = Vec2f(offset.dot(u), offset.dot(v));
          
          if (IsInFrontOfLine(X, S1, S2)) {
            if (debug) {
              LOG(INFO) << "Cannot delete skinny triangle (in this attempt). left_triangle_index: " << left_triangle_index
                        << ", right_neighbor: " << right_neighbor << ", k: " << k;
              LOG(INFO) << "  S1: " << S1.transpose() << ", S2: " << S2.transpose() << ", X: " << X.transpose();
            }
            can_delete = false;
            break;
          }
        }
        
        if (!can_delete) {
          continue;
        }
        
        // Delete the considered_neighbor.
        to_erase[considered_neighbor] = true;
        ++ erase_count;
        angle_diff[left_triangle_index] = merged_angle;
        skinny[left_triangle_index] = merged_angle < min_triangle_angle_;
        
        if (debug) {
          LOG(INFO) << "Deleted neighbor " << considered_neighbor;
        }
      }
    
      if (debug) {
        for (u32 i = 1; i < selected_neighbor_count; ++ i) {  // Do not include the start angle.
          LOG(INFO) << "  to_erase[" << i << "] = " << to_erase[i];
        }
      }
      
      // Perform the deletion of the marked points from angle_indices.
      if (erase_count > 0) {
        int output_i = 1;
        for (u32 i = 1; i < selected_neighbor_count; ++ i) {  // Do not include the start angle.
          if (!to_erase[i]) {
            selected_neighbors[output_i] = selected_neighbors[i];
            gaps[output_i] = gaps[i];
            angle_diff[output_i] = angle_diff[i];
            ++ output_i;
          }
        }
        selected_neighbor_count -= erase_count;
      }  // erase_count > 0
    }  // skinny_surfel_count > 0
    
    // Close small holes.
    u32 hole_start = Surfel::kInvalidIndex;
    for (u32 i = 0; i < selected_neighbor_count; ++ i) {
      if (i < selected_neighbor_count - 1 && gaps[i]) {
        bool closable_hole = angle_diff[i] < M_PI;  // NOTE: Using M_PI instead of max_triangle_angle_ here on purpose to also close malformed gaps.
        
        if (debug) {
          LOG(INFO) << "Examining hole at selected neighbor " << i << " ...";
          LOG(INFO) << "Closable according to angle_diff: " << (closable_hole ? "yes" : "no") << " (angle_diff[i]: " << angle_diff[i] << ", M_PI: " << M_PI;
        }
        
        if (closable_hole) {
          // This is a potentially closable gap. Check whether it is closed on the opposite side.
          closable_hole = false;
          u32 left_opposite_surfel_index = selected_neighbors[i].surfel_index;
          Surfel* left_opposite_surfel = &surfels_[left_opposite_surfel_index];
          if (left_opposite_surfel->meshing_state() == Surfel::MeshingState::kFront) {
            int next_neighbor = i + 1;
            u32 right_opposite_surfel_index = selected_neighbors[next_neighbor].surfel_index;
            Surfel* right_opposite_surfel = &surfels_[right_opposite_surfel_index];
            if (right_opposite_surfel->meshing_state() == Surfel::MeshingState::kFront) {
              for (const Front& front : left_opposite_surfel->fronts()) {
                if (front.left == right_opposite_surfel_index) {
                  closable_hole = true;
                  break;
                }
              }
            }
          }
        }
        
        if (debug) {
          LOG(INFO) << "Closable: " << (closable_hole ? "yes" : "no");
        }
        
        if (closable_hole) {
          if (hole_start == Surfel::kInvalidIndex) {
            hole_start = i;
          }
        } else {
          // Unclosable hole. Abort if started on a connected gap.
          hole_start = Surfel::kInvalidIndex;
          ++ i;
          while (i < selected_neighbor_count && gaps[i]) {
            ++ i;
          }
          -- i;
        }
      } else if (hole_start != Surfel::kInvalidIndex) {
        // Code for debugging where hole-closing happens.
        // Reference surfel is red, the surfels of the
        // closed triangle are blue.
        // shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
        // ConvertToMesh3fCu8(visualization_mesh.get());
        // (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
        // for (u32 d = hole_start; d <= i; ++ d) {
        //   (*visualization_mesh->vertices_mutable())->at(selected_neighbors[d].surfel_index).color() = Vec3u8(60, 60, 255);
        // }
        // render_window_->CenterViewOn(surfel->position());
        // render_window_->UpdateVisualizationMesh(visualization_mesh);
        // std::getchar();
        // debug = true;
        
        // All gaps from hole_start to i - 1 can be filled to close a hole.
        while (hole_start < i) {
          if (debug) {
            LOG(INFO) << "Setting gaps[hole_start = " << hole_start << "] = false";
          }
          
          gaps[hole_start] = false;
          ++ hole_start;
        }
        hole_start = Surfel::kInvalidIndex;
        
        ++ holes_closed_counter_;
      }
    }
    
    // Write triangles and update the fronts.
    for (int i = 0; i < static_cast<int>(selected_neighbor_count) - 1; ++ i) {
      if (gaps[i]) {
        continue;
      }
      
      // Write the triangle between neighbor i and (i + 1).
      AddTriangle(surfel_index, selected_neighbors[i + 1].surfel_index, selected_neighbors[i].surfel_index, debug);
      
      // Update the fronts of all vertices of the triangle. Here we do not
      // check whether left/right is consistent because it is supposed to be
      // more important to keep the boundary sane.
      
      // Update the reference surfel's front(s). If the current front touches
      // the sides of the new triangle then it is updated, otherwise it must
      // be split up into two fronts.
      Front* front_mutable = &surfel_front->at(front_index);
      if (front_mutable->left == selected_neighbors[i].surfel_index) {
        // Move the left front towards the right (a check whether this closes
        // the front will be done later).
        front_mutable->left = selected_neighbors[i + 1].surfel_index;
      } else if (front_mutable->right == selected_neighbors[i + 1].surfel_index) {
        // Move the right front towards the left (a check whether this closes
        // the front will be done later).
        front_mutable->right = selected_neighbors[i].surfel_index;
      } else if (front_mutable->right == selected_neighbors[i].surfel_index) {
        // Flipped case 1.
        front_mutable->right = selected_neighbors[i + 1].surfel_index;
      } else if (front_mutable->left == selected_neighbors[i + 1].surfel_index) {
        // Flipped case 2.
        front_mutable->left = selected_neighbors[i].surfel_index;
      } else {
        // Split the front up into two new ones. Split off the left one
        // because it won't be updated again.
        Front left_front(front_mutable->left, selected_neighbors[i].surfel_index);
        new_fronts_.push_back(left_front);
        front_mutable->left = selected_neighbors[i + 1].surfel_index;
        if (debug) {
          LOG(INFO) << "DEBUG: Split off front. front_mutable->left = " << front_mutable->left;
        }
      }
      
      // Update the front(s) of neighbor i.
      UpdateSurfelFronts(selected_neighbors[i].surfel_index,
                          surfel_index,
                          selected_neighbors[i + 1].surfel_index,
                          normal,
                          selected_neighbors[i].angle,
                          surfel_proj,
                          selected_neighbors[i].uv,
                          u,
                          v,
                          surfel_index);
      
      // Update the front(s) of neighbor (i + 1).
      UpdateSurfelFronts(selected_neighbors[i + 1].surfel_index,
                          selected_neighbors[i].surfel_index,
                          surfel_index,
                          normal,
                          selected_neighbors[i + 1].angle,
                          surfel_proj,
                          selected_neighbors[i + 1].uv,
                          u,
                          v,
                          surfel_index);
    }  // End of loop over the selected neighbors for triangulation.
  }  // End of loop over the reference surfel's fronts.
  
  // Delete completed fronts (check if left == right).
  int output_index = 0;
  for (usize front_index = 0; front_index < surfel_front->size(); ++ front_index) {
    const Front& front = surfel_front->at(front_index);
    if (front.left != front.right) {
      // Keep this front.
      surfel_front->at(output_index) = front;
      ++ output_index;
    }
  }
  surfel_front->resize(output_index);
  
  // Add new_fronts_.
  if (!new_fronts_.empty()) {
    surfel_front->insert(surfel_front->end(), new_fronts_.begin(), new_fronts_.end());
  }
  
  // Delete vector from front map if empty. Set the surfel's state to boundary
  // if there is still a front remaining for it, otherwise set it to complete.
  if (surfel_front->empty()) {
    surfel->SetMeshingState(Surfel::MeshingState::kCompleted);
    // CheckSurfelState(surfel_index);
  } else {
    surfel->SetMeshingState(Surfel::MeshingState::kFront);
  }
  
  if (debug) {
    LOG(INFO) << "Meshing state of reference surfel on finishing TryToAdvanceFront(): " << static_cast<int>(surfel->meshing_state());
    
    LOG(INFO) << "DEBUG: Showing final state after TryToAdvanceFront(). Reference surfel is red, front neighbors are yellow if the state is boundary.";
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
    if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
      for (usize front_index = 0; front_index < surfel_front->size(); ++ front_index) {
        const Front& front = surfel_front->at(front_index);
        LOG(INFO) << "  Final front neighbors: left: " << front.left << ", right: " << front.right;
        (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
        (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(205, 205, 60);
      }
    }
    
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
}

void SurfelMeshing::UpdateSurfelFronts(
    u32 corner_surfel_index,
    u32 left_surfel_index,
    u32 right_surfel_index,
    const Vec3f& /*projection_normal*/,
    float corner_angle,
    const Vec3f& surfel_proj,
    const Vec2f& corner_uv,
    const Vec3f& u,
    const Vec3f& v,
    u32 reference_surfel_index) {
  Surfel* corner_surfel = &surfels_[corner_surfel_index];
  
  constexpr bool debug = false;
  // Uncomment this to debug a specific surfel:
  // debug |= corner_surfel_index == 118504;
  if (debug) {
    LOG(INFO) << "DEBUG: UpdateSurfelFronts() start state. Debug vertex in red and all its front neighbors in yellow.";
    LOG(INFO) << "corner_surfel_index: " << corner_surfel_index << ", left_surfel_index: " << left_surfel_index << ", right_surfel_index: " << right_surfel_index;
    shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
    ConvertToMesh3fCu8(visualization_mesh.get());
    
    (*visualization_mesh->vertices_mutable())->at(corner_surfel_index).color() = Vec3u8(255, 60, 60);
    for (const Front& front : surfels_[corner_surfel_index].fronts()) {
      (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
      (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(255, 255, 60);
    }
    render_window_->CenterViewOn(surfels_[corner_surfel_index].position());
    render_window_->UpdateVisualizationMesh(visualization_mesh);
    std::getchar();
  }
  
  // Sanity check.
  if (corner_surfel->meshing_state() == Surfel::MeshingState::kCompleted) {
    ++ fronts_triangles_inconsistency_counter_;
    // LOG(WARNING) << "Attached a triangle to a complete surfel. This should not happen.";
    return;
  }
  
  // If the corner surfel was free before, add a new front for it and set it
  // to the front meshing state.
  if (corner_surfel->meshing_state() == Surfel::MeshingState::kFree) {
    corner_surfel->SetMeshingState(Surfel::MeshingState::kFront);
    corner_surfel->fronts().push_back(Front(left_surfel_index, right_surfel_index));
    
    // CheckSurfelState(corner_surfel_index);
    if (debug) {
      LOG(INFO) << "DEBUG: UpdateSurfelFronts() end state after adding a new front to the free surfel.";
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      ConvertToMesh3fCu8(visualization_mesh.get());
      
      (*visualization_mesh->vertices_mutable())->at(corner_surfel_index).color() = Vec3u8(255, 60, 60);
      for (const Front& front : surfels_[corner_surfel_index].fronts()) {
        (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
        (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(255, 255, 60);
      }
      render_window_->CenterViewOn(surfels_[corner_surfel_index].position());
      render_window_->UpdateVisualizationMesh(visualization_mesh);
      std::getchar();
    }
    return;
  }
  
  // The corner surfel was not free before, get its fronts.
  std::vector<Front>* corner_fronts = &corner_surfel->fronts();
  
  // If any front contains the new triangle's left or middle edge, then
  // flip it over to the other of those two.
  for (usize neighbor_front_index = 0; neighbor_front_index < corner_fronts->size(); ++ neighbor_front_index) {
    Front* neighbor_front = &corner_fronts->at(neighbor_front_index);
    if (neighbor_front->right == left_surfel_index) {
      // Move right front towards the left.
      neighbor_front->right = right_surfel_index;
      if (neighbor_front->left == neighbor_front->right) {
        CloseFront(corner_surfel_index, corner_fronts, neighbor_front_index);
      }
      // CheckSurfelState(corner_surfel_index);
      if (debug) {
        LOG(INFO) << "UpdateSurfelFronts(): Moved right front towards the left";
      }
      return;
    } else if (neighbor_front->left == right_surfel_index) {
      // Move left front towards the right.
      neighbor_front->left = left_surfel_index;
      if (neighbor_front->left == neighbor_front->right) {
        CloseFront(corner_surfel_index, corner_fronts, neighbor_front_index);
      }
      // CheckSurfelState(corner_surfel_index);
      if (debug) {
        LOG(INFO) << "UpdateSurfelFronts(): Moved left front towards the right";
      }
      return;
    } else if (neighbor_front->left == left_surfel_index) {
      // Flipped case 1.
      neighbor_front->left = right_surfel_index;
      if (neighbor_front->left == neighbor_front->right) {
        CloseFront(corner_surfel_index, corner_fronts, neighbor_front_index);
      }
      // CheckSurfelState(corner_surfel_index);
      if (debug) {
        LOG(INFO) << "UpdateSurfelFronts(): Flipped case 1";
      }
      return;
    } else if (neighbor_front->right == right_surfel_index) {
      // Flipped case 2.
      neighbor_front->right = left_surfel_index;
      if (neighbor_front->left == neighbor_front->right) {
        CloseFront(corner_surfel_index, corner_fronts, neighbor_front_index);
      }
      // CheckSurfelState(corner_surfel_index);
      if (debug) {
        LOG(INFO) << "UpdateSurfelFronts(): Flipped case 2";
      }
      return;
    }
  }
  
  // Compute angle from neighbor to reference surfel (this is arbitrary,
  // could use any direction within the new triangle).
  float angle_r = corner_angle + M_PI;
  while (angle_r >= M_PI) {
    angle_r -= 2 * M_PI;
  }
  
  // If no adjacent front has been found, the existing front which contains
  // the directions to the new triangle must be split up into two.
  for (usize neighbor_front_index = 0; neighbor_front_index < corner_fronts->size(); ++ neighbor_front_index) {
    Front* front = &corner_fronts->at(neighbor_front_index);
    
    // Compute the angles to the front edges.
    Vec3f offset = surfels_[front->left].position() - surfel_proj;
    Vec2f left_pos(offset.dot(u), offset.dot(v));
    float angle_left = ApproxAtan2(left_pos.coeff(1) - corner_uv.coeff(1),
                                   left_pos.coeff(0) - corner_uv.coeff(0));
    
    offset = surfels_[front->right].position() - surfel_proj;
    Vec2f right_pos(offset.dot(u), offset.dot(v));
    float angle_right = ApproxAtan2(right_pos.coeff(1) - corner_uv.coeff(1),
                                    right_pos.coeff(0) - corner_uv.coeff(0));
    
    bool found = false;
    if (angle_left <= angle_right) {
      // The space from angle_left to angle_right is free.
      if (angle_left <= angle_r && angle_r <= angle_right) {
        found = true;
      }
    } else {
      // The space from angle_left to M_PI, and -M_PI to angle_right is free.
      if (angle_r >= angle_left || angle_r <= angle_right) {
        found = true;
      }
    }
    
    if (found) {
      // Split up this front.
      u32 old_front_right = front->right;
      front->right = right_surfel_index;
      corner_fronts->push_back(Front(left_surfel_index, old_front_right));
      
      // CheckSurfelState(corner_surfel_index);
      if (debug) {
        LOG(INFO) << "DEBUG: UpdateSurfelFronts() end state after split.";
        shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
        ConvertToMesh3fCu8(visualization_mesh.get());
        
        (*visualization_mesh->vertices_mutable())->at(corner_surfel_index).color() = Vec3u8(255, 60, 60);
        for (const Front& front : surfels_[corner_surfel_index].fronts()) {
          (*visualization_mesh->vertices_mutable())->at(front.left).color() = Vec3u8(255, 255, 60);
          (*visualization_mesh->vertices_mutable())->at(front.right).color() = Vec3u8(255, 255, 60);
        }
        render_window_->CenterViewOn(surfels_[corner_surfel_index].position());
        render_window_->UpdateVisualizationMesh(visualization_mesh);
        std::getchar();
      }
      return;
    }
  }
  
  // No front found which contains the triangle directions, not even on the
  // other side. Something went wrong, as in this case we should not have
  // connected to this surfel at all. Not sure what we can do to try to
  // minimize the damage.
  ++ connected_to_surfel_without_suitable_front_counter_;
  
  if (debug) {
    LOG(WARNING) << "Connected a triangle to a front/boundary surfel, but did not find a front at this surfel to update."
                  << " Reference surfel index: " << reference_surfel_index << ", problem surfel index: " << corner_surfel_index
                  << " Number of fronts at the problem surfel: " << corner_fronts->size();
    // DEBUG
    LOG(WARNING) << "  Angle to reference surfel (within the new triangle directions): " << angle_r << " (corner_angle: " << corner_angle << ")";
    for (usize neighbor_front_index = 0; neighbor_front_index < corner_fronts->size(); ++ neighbor_front_index) {
      Front* front = &corner_fronts->at(neighbor_front_index);
      
      // Compute the angles to the front edges.
      Vec3f offset = surfels_[front->left].position() - surfel_proj;
      Vec2f left_pos(offset.dot(u), offset.dot(v));
      float angle_left = ApproxAtan2(left_pos.coeff(1) - corner_uv.coeff(1),
                                left_pos.coeff(0) - corner_uv.coeff(0));
      LOG(WARNING) << "  Neighbor front " << neighbor_front_index << " left angle: " << angle_left << " surfel index: " << front->left;
      
      offset = surfels_[front->right].position() - surfel_proj;
      Vec2f right_pos(offset.dot(u), offset.dot(v));
      float angle_right = ApproxAtan2(right_pos.coeff(1) - corner_uv.coeff(1),
                                right_pos.coeff(0) - corner_uv.coeff(0));
      LOG(WARNING) << "  Neighbor front " << neighbor_front_index << " right angle: " << angle_right << " surfel index: " << front->right;
    }
  }
}

void SurfelMeshing::CloseFront(
    u32 surfel_index,
    std::vector<Front>* surfel_fronts,
    usize front_index) {
  if (surfel_fronts->size() == 1) {
    // Mark the surfel complete.
    surfels_[surfel_index].SetMeshingState(Surfel::MeshingState::kCompleted);
    // CheckSurfelState(surfel_index);
    surfel_fronts->clear();
  } else {
    surfel_fronts->erase(surfel_fronts->begin() + front_index);
  }
}

bool SurfelMeshing::TryToCreateInitialTriangle(
    u32 surfel_index, int neighbor_count, u32* neighbor_indices,
    Neighbor* neighbors, std::vector<EdgeData>* edges, bool debug) {
  Surfel* surfel = &surfels_[surfel_index];
  
  // Define a coordinate system on the plane defined by the surfel normal,
  // going through the origin.
  const Vec3f& normal = surfel->normal();
  Vec3f v = normal.unitOrthogonal();
  Vec3f u = normal.cross(v);
  
  // Project the surfel onto this plane.
  Vec3f surfel_proj = surfel->position() - normal.dot(surfel->position()) * normal;  // NOTE: "proj_qp_" in PCL code
  
  // Project the neighbors onto the plane and save their coordinates,
  // angles, and adjacent boundary edges.
  ProjectNeighborsAndTestVisibility(
      surfel_index, surfel, surfel_proj, neighbor_count, neighbor_indices,
      neighbors, u, v, edges, debug);
  
  // Delete all invisible neighbors from the neighbor list (including the
  // reference surfel, having index 0). Note that neighbor_indices must not
  // be used afterwards anymore as the indexing changes.
  u32 selected_neighbor_count = 0;
  for (int neighbor_index = 1; neighbor_index < neighbor_count;
        ++ neighbor_index) {  // Neighbor with index 0 is the surfel itself
    if (neighbors[neighbor_index].visible) {
      neighbors[selected_neighbor_count] = neighbors[neighbor_index];
      ++ selected_neighbor_count;
    }
  }
  
  // Select the first (closest) two visible neighbors with suitable properties.
  bool triangle_created = false;
  for (u32 first = 0; first < selected_neighbor_count; ++ first) {
    Neighbor* first_neighbor = &neighbors[first];
    Surfel* first_surfel = &surfels_[neighbors[first].surfel_index];
    
    for (u32 second = first + 1; second < selected_neighbor_count; ++ second) {
      Neighbor* second_neighbor = &neighbors[second];
      Surfel* second_surfel = &surfels_[neighbors[second].surfel_index];
      
      // Check whether the angle at the reference surfel is within the desired
      // range (note: we do not check the angle at the other surfels).
      float angle_diff = fabs(second_neighbor->angle - first_neighbor->angle);
      bool triangle_between_min_max_angles = angle_diff < M_PI;
      if (!triangle_between_min_max_angles) {
        angle_diff = 2 * M_PI - angle_diff;
      }
      if (angle_diff < min_triangle_angle_) {
        continue;
      } else if (angle_diff > max_triangle_angle_) {
        continue;
      }
      
      // Check that no other visible neighbor is included in the triangle.
      // If first == 0 and second == 1, then there is no need to check since
      // these are the closest two neighbors, thus no other one can be
      // included.
      if (first != 0 || second != 1) {
        float angle_min = second_neighbor->angle;
        float angle_max = first_neighbor->angle;
        if (angle_max < angle_min) {
          angle_min = first_neighbor->angle;
          angle_max = second_neighbor->angle;
        }
        
        Vec3f offset = surfels_[neighbors[first].surfel_index].position() - surfel_proj;
        Vec2f S1(offset.dot(u), offset.dot(v));
        offset = surfels_[neighbors[second].surfel_index].position() - surfel_proj;
        Vec2f S2(offset.dot(u), offset.dot(v));
        
        bool have_problem = false;
        for (u32 neighbor_index = 0; neighbor_index < selected_neighbor_count; ++ neighbor_index) {
          if (neighbor_index == first || neighbor_index == second) {
            continue;
          }
          
          // Check whether the neighbor is in the direction of the potential
          // triangle, i.e., between first and second.
          if (triangle_between_min_max_angles) {
            if (neighbors[neighbor_index].angle < angle_min ||
                neighbors[neighbor_index].angle > angle_max) {
              continue;
            }
          } else {
            if (neighbors[neighbor_index].angle > angle_min &&
                neighbors[neighbor_index].angle < angle_max) {
              continue;
            }
          }
          
          // Check whether the neighbor would be within the triangle.
          offset = surfels_[neighbors[neighbor_index].surfel_index].position() - surfel_proj;
          Vec2f X(offset.dot(u), offset.dot(v));
          
          if (IsVisible(X, S1, S2)) {
            have_problem = true;
            break;
          }
        }
        
        if (have_problem) {
          continue;
        }
      }
      
      if (debug) {
        LOG(INFO) << "Adding an initial triangle";
        LOG(INFO) << "First: " << first << " neighbors[first].surfel_index: " << neighbors[first].surfel_index
                  << " second: " << second << " neighbors[second].surfel_index: " << neighbors[second].surfel_index
                  << " selected_neighbor_count: " << selected_neighbor_count;
      }
      
      // Determine the triangle orientation.
      // Vector from first_surfel to reference_index.
      const Vec3f first_to_reference = surfel->position() - first_surfel->position();
      // Vector from second_surfel to reference_index.
      const Vec3f second_to_reference = surfel->position() - second_surfel->position();
      
      // Create the initial triangle.
      // For counter-clockwise ordering, first_to_reference.cross(second_to_reference) should point in the same
      // direction as the normal.
      u32 left_neighbor, right_neighbor;  // Left and right from the point of view of the reference surfel's triangulation. For the reference surfel's front, they would be called in the opposite way.
      if (normal.dot(first_to_reference.cross(second_to_reference)) > 0) {
        left_neighbor = second;
        right_neighbor = first;
      } else {
        left_neighbor = first;
        right_neighbor = second;
      }
      
      AddTriangle(surfel_index, neighbors[right_neighbor].surfel_index, neighbors[left_neighbor].surfel_index, debug);
      
      // Adjust the reference surfel's state.
      surfel->fronts().push_back(Front(neighbors[right_neighbor].surfel_index, neighbors[left_neighbor].surfel_index));
      surfel->SetMeshingState(Surfel::MeshingState::kFront);
      
      // Adjust the left neighbor's state.
      UpdateSurfelFronts(neighbors[left_neighbor].surfel_index,
                          surfel_index,
                          neighbors[right_neighbor].surfel_index,
                          normal,
                          neighbors[left_neighbor].angle,
                          surfel_proj,
                          neighbors[left_neighbor].uv,
                          u,
                          v,
                          surfel_index);
      
      // Adjust the right neighbor's state.
      UpdateSurfelFronts(neighbors[right_neighbor].surfel_index,
                          neighbors[left_neighbor].surfel_index,
                          surfel_index,
                          normal,
                          neighbors[right_neighbor].angle,
                          surfel_proj,
                          neighbors[right_neighbor].uv,
                          u,
                          v,
                          surfel_index);
      
      triangle_created = true;
      break;
    }
    if (triangle_created) {
      break;
    }
  }
  return triangle_created;
}

void SurfelMeshing::AddTriangle(u32 a, u32 b, u32 c, bool debug) {
  Surfel* surfel_a = &surfels_[a];
  Surfel* surfel_b = &surfels_[b];
  Surfel* surfel_c = &surfels_[c];
  
  if (debug) {
    LOG(INFO) << "  triangle vertices: " << a << ", " << b << ", " << c;
  }
  
//   // DEBUG:
//   if (a == b || b == c || a == c) {
//     LOG(FATAL) << "Attempting to generate a degenerate triangle: " << a << ", " << b << ", " << c << std::endl << *surfel_a << std::endl << *surfel_b << std::endl << *surfel_c;
//   }
  
  u32 triangle_index;
  if (next_free_triangle_index_ == kNoFreeIndex) {
    triangles_.emplace_back(a, b, c);
    triangle_index = triangles_.size() - 1;
  } else {
    triangle_index = next_free_triangle_index_;
    next_free_triangle_index_ = triangles_[next_free_triangle_index_].free_list_value();
    // Use placement new to construct the new triangle over the next free list entry.
    new(&triangles_[triangle_index]) SurfelTriangle(a, b, c);
  }
  
  surfel_a->AddTriangle(triangle_index);
  surfel_b->AddTriangle(triangle_index);
  surfel_c->AddTriangle(triangle_index);
  #ifdef KEEP_TRIANGLES_IN_OCTREE
    octree_.AddTriangle(triangle_index, triangles_[triangle_index]);
  #endif
}

// Variant of IsVisible where the ray starts from the given origin point.
template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
bool SurfelMeshing::IsVisible(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& S1, const MatrixBase<DerivedC>& S2, const MatrixBase<DerivedD>& origin) {
  return IsVisible(X - origin, S1 - origin, S2 - origin);
}

template <typename DerivedA, typename DerivedB, typename DerivedC>
bool SurfelMeshing::IsVisible(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& S1, const MatrixBase<DerivedC>& S2) {
  // Compute dot product of a vector which is perpendicular to X with S1 and S2.
  // If the results have the same sign, there is no intersection.
  float X_perp_dot_S1 = X.y() * S1.x() + -X.x() * S1.y();
  float X_perp_dot_S2 = X.y() * S2.x() + -X.x() * S2.y();
  if (X_perp_dot_S1 * X_perp_dot_S2 > 0) {
    return true;
  }
  
  // Compute dot product of a vector which is perpendicular to (S2 - S1) with X and S1 (arbitrary, could also use S2).
  const float S1_S2_perp_x = S2.y() - S1.y();
  const float S1_S2_perp_y = -(S2.x() - S1.x());
  float S1_S2_perp_dot_x = S1_S2_perp_x * X.x() + S1_S2_perp_y * X.y();
  float S1_S2_perp_dot_S1 = S1_S2_perp_x * S1.x() + S1_S2_perp_y * S1.y();
  return (S1_S2_perp_dot_S1 > 0 && S1_S2_perp_dot_S1 > S1_S2_perp_dot_x) ||
         (S1_S2_perp_dot_S1 < 0 && S1_S2_perp_dot_S1 < S1_S2_perp_dot_x);
}

template <typename DerivedA, typename DerivedB, typename DerivedC>
bool SurfelMeshing::IsInFrontOfLine(const MatrixBase<DerivedA>& X, const MatrixBase<DerivedB>& S1, const MatrixBase<DerivedC>& S2) {
  Vec2f S1_to_S2 = S2 - S1;
  Vec2f perpendicular = Vec2f(-S1_to_S2.y(), S1_to_S2.x());
  return perpendicular.dot(S1 - X) * perpendicular.dot(S1) > 0;
}

void SurfelMeshing::CheckSurfelState(u32 surfel_index) {
  Surfel* surfel = &surfels_[surfel_index];
  
  vector<ConnectedTriangleComponent> components;
  
  bool have_closed_component = false;
  bool has_stray_component = false;
  for (int t = 0; t < surfel->GetTriangleCount(); ++ t) {
    int triangle_index = surfel->GetTriangle(t);
    SurfelTriangle* triangle = &triangles_[triangle_index];
    
    // Identify the two other surfels referenced in the triangle.
    bool found = false;
    u32 surfel_index_a;
    u32 surfel_index_b;
    for (int i = 0; i < 3; ++ i) {
      if (triangle->index(i) == surfel_index) {
        surfel_index_a = triangle->index((i + 1) % 3);
        surfel_index_b = triangle->index((i + 2) % 3);
        found = true;
        break;
      }
    }
    if (!found) {
      LOG(FATAL) << "CheckSurfelState(): found a triangle referenced in a surfel which does not have the surfel as one of its indices!";
      continue;
    }
    
    // Does the triangle attach to an existing component?
    bool attached_to_existing = false;
    for (usize c = 0; c < components.size(); ++ c) {
      ConnectedTriangleComponent* component = &components[c];
      if (component->surfel_index_a == surfel_index_a) {
        if (component->surfel_index_b == surfel_index_b) {
          // The component got closed.
          if (have_closed_component) {
            has_stray_component = true;
          }
          have_closed_component = true;
          components.erase(components.begin() + c);
          attached_to_existing = true;
          break;
        } else {
          component->surfel_index_a = surfel_index_b;
          attached_to_existing = true;
          break;
        }
      } else if (component->surfel_index_a == surfel_index_b) {
        if (component->surfel_index_b == surfel_index_a) {
          // The component got closed.
          if (have_closed_component) {
            has_stray_component = true;
          }
          have_closed_component = true;
          components.erase(components.begin() + c);
          attached_to_existing = true;
          break;
        } else {
          component->surfel_index_a = surfel_index_a;
          attached_to_existing = true;
          break;
        }
      } else if (component->surfel_index_b == surfel_index_a) {
        component->surfel_index_b = surfel_index_b;
        attached_to_existing = true;
        break;
      } else if (component->surfel_index_b == surfel_index_b) {
        component->surfel_index_b = surfel_index_a;
        attached_to_existing = true;
        break;
      }
    }
    
    if (!attached_to_existing) {
      // Create new component.
      ConnectedTriangleComponent new_component;
      new_component.surfel_index_a = surfel_index_a;
      new_component.surfel_index_b = surfel_index_b;
      components.push_back(new_component);
    }
  }
  
  // Merge components that touch.
  bool had_changes = true;
  while (had_changes) {
    had_changes = false;
    for (int c1 = 0; c1 < static_cast<int>(components.size()); ++ c1) {
      ConnectedTriangleComponent* component1 = &components[c1];
      for (int c2 = c1 + 1; c2 < static_cast<int>(components.size()); ++ c2) {
        ConnectedTriangleComponent* component2 = &components[c2];
        
        if (component1->surfel_index_a == component2->surfel_index_a) {
          if (component1->surfel_index_b == component2->surfel_index_b) {
            if (have_closed_component) {
              has_stray_component = true;
            }
            have_closed_component = true;
            components.erase(components.begin() + c2);
            components.erase(components.begin() + c1);
            -- c1;
            had_changes = true;
            break;
          } else {
            component1->surfel_index_a = component2->surfel_index_b;
            components.erase(components.begin() + c2);
            -- c2;
            had_changes = true;
          }
        } else if (component1->surfel_index_a == component2->surfel_index_b) {
          if (component1->surfel_index_b == component2->surfel_index_a) {
            if (have_closed_component) {
              has_stray_component = true;
            }
            have_closed_component = true;
            components.erase(components.begin() + c2);
            components.erase(components.begin() + c1);
            -- c1;
            had_changes = true;
            break;
          } else {
            component1->surfel_index_a = component2->surfel_index_a;
            components.erase(components.begin() + c2);
            -- c2;
            had_changes = true;
          }
        } else if (component1->surfel_index_b == component2->surfel_index_a) {
          component1->surfel_index_b = component2->surfel_index_b;
          components.erase(components.begin() + c2);
          -- c2;
          had_changes = true;
        } else if (component1->surfel_index_b == component2->surfel_index_b) {
          component1->surfel_index_b = component2->surfel_index_a;
          components.erase(components.begin() + c2);
          -- c2;
          had_changes = true;
        }
      }
    }
  }
  
//   if (!components.empty()) {
//     has_stray_component = true;
//     LOG(ERROR) << "Leftover components in CheckSurfelState:";
//     for (int c = 0; c < static_cast<int>(components.size()); ++ c) {
//       ConnectedTriangleComponent* component = &components[c];
//       LOG(ERROR) << "  a: " << component->surfel_index_a << ", b: " << component->surfel_index_b;
//     }
//   }
  
  if (has_stray_component) {
    LOG(ERROR) << "CheckSurfelState found a stray component!";
    
//     shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
//     ConvertToMesh3fCu8(visualization_mesh.get());
//     (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
//     for (int c = 0; c < static_cast<int>(components.size()); ++ c) {
//       ConnectedTriangleComponent* component = &components[c];
//       (*visualization_mesh->vertices_mutable())->at(component->surfel_index_a).color() = Vec3u8(255, 255, 60);
//       (*visualization_mesh->vertices_mutable())->at(component->surfel_index_b).color() = Vec3u8(255, 255, 60);
//     }
//     render_window_->CenterViewOn(surfel->position());
//     render_window_->UpdateVisualizationMesh(visualization_mesh);
//     std::getchar();
  }
  
  Surfel::MeshingState computed_state;
  if (surfel->GetTriangleCount() > 0) {
    computed_state = have_closed_component ? Surfel::MeshingState::kCompleted : Surfel::MeshingState::kFront;
  } else {
    computed_state = Surfel::MeshingState::kFree;
  }
  
  if (surfel->meshing_state() != computed_state) {
    LOG(ERROR) << "CheckSurfelState found a mismatch: computed state is " << static_cast<int>(computed_state) << ", saved surfel state is " << static_cast<int>(surfel->meshing_state()) << ".";
    LOG(ERROR) << "  Number of attached triangles: " << surfel->GetTriangleCount();
  
//     LOG(ERROR) << "  Visualization: all vertices of adjacent triangles are yellow, the surfel " << surfel_index << " with the state mismatch is red.";
//     
//     shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
//     ConvertToMesh3fCu8(visualization_mesh.get());
//     
//     for (int t = 0; t < surfel->GetTriangleCount(); ++ t) {
//       int triangle_index = surfel->GetTriangle(t);
//       SurfelTriangle* triangle = &triangles_[triangle_index];
//       for (int i = 0; i < 3; ++ i) {
//         (*visualization_mesh->vertices_mutable())->at(triangle->index(i)).color() = Vec3u8(255, 255, 60);
//       }
//     }
//     (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
//     
//     render_window_->CenterViewOn(surfel->position());
//     render_window_->UpdateVisualizationMesh(visualization_mesh);
//     std::getchar();
  }
  
  // Check that the computed fronts match the fronts in the front map.
  if (surfel->meshing_state() == Surfel::MeshingState::kFront) {
    bool have_front_mismatch = false;
    
    std::vector<Front>* surfel_front = &surfel->fronts();
    std::vector<bool> front_left_matched(surfel_front->size(), false);
    std::vector<bool> front_right_matched(surfel_front->size(), false);
    for (int c = 0; c < static_cast<int>(components.size()); ++ c) {
      bool component_surfel_a_matched = false;
      bool component_surfel_b_matched = false;
      for (int f = 0; f < static_cast<int>(surfel_front->size()); ++ f) {
        if (surfel_front->at(f).left == components[c].surfel_index_a ||
            surfel_front->at(f).left == components[c].surfel_index_b) {
          if (front_left_matched[f]) {
            LOG(WARNING) << "Double match of left side of a front. surfel_index: " << surfel_index << ", Test surfel index: " << surfel_front->at(f).left;
            have_front_mismatch = true;
          }
          front_left_matched[f] = true;
          if (surfel_front->at(f).left == components[c].surfel_index_a) {
            component_surfel_a_matched = true;
          } else {
            component_surfel_b_matched = true;
          }
        }
        if (surfel_front->at(f).right == components[c].surfel_index_a ||
            surfel_front->at(f).right == components[c].surfel_index_b) {
          if (front_right_matched[f]) {
            LOG(WARNING) << "Double match of right side of a front. surfel_index: " << surfel_index << ", Test surfel index: " << surfel_front->at(f).right;
            have_front_mismatch = true;
          }
          front_right_matched[f] = true;
          if (surfel_front->at(f).right == components[c].surfel_index_a) {
            component_surfel_a_matched = true;
          } else {
            component_surfel_b_matched = true;
          }
        }
      }
      
      if (!component_surfel_a_matched) {
        LOG(WARNING) << "Did not find a front next to a component edge. surfel_index: " << surfel_index << ", Test surfel index: " << components[c].surfel_index_a;
        have_front_mismatch = true;
      }
      if (!component_surfel_b_matched) {
        LOG(WARNING) << "Did not find a front next to a component edge. surfel_index: " << surfel_index << ", Test surfel index: " << components[c].surfel_index_b;
        have_front_mismatch = true;
      }
    }
    
    if (have_front_mismatch) {
      shared_ptr<Mesh3fCu8> visualization_mesh(new Mesh3fCu8());
      ConvertToMesh3fCu8(visualization_mesh.get());
      
      (*visualization_mesh->vertices_mutable())->at(surfel_index).color() = Vec3u8(255, 60, 60);
      
      render_window_->CenterViewOn(surfel->position());
      render_window_->UpdateVisualizationMesh(visualization_mesh);
      std::getchar();
    }
  }
}

void SurfelMeshing::DeleteAllTrianglesConnectedToSurfel(u32 surfel_index) {
  Surfel* surfel = &surfels_[surfel_index];
  
  for (int t = surfel->GetTriangleCount() - 1; t >= 0; -- t) {
    int triangle_index = surfel->GetTriangle(t);
    DeleteTriangleForRemeshing(triangle_index, surfel_index);
  }
  
  // Reset the surfel.
  surfel->RemoveAllTriangles();
  surfel->fronts().clear();
  surfel->SetMeshingState(Surfel::MeshingState::kFree);
  surfel->SetCanBeReset(false);  // Avoid remeshing this surfel again in the same iteration if it is free now.
  surfels_to_remesh_.push_back(surfel_index);
}

void SurfelMeshing::ResetSurfelToFree(u32 surfel_index) {
  Surfel* surfel = &surfels_[surfel_index];
  
  // Delete all triangles adjacent to the surfel (as normal).
  DeleteAllTrianglesConnectedToSurfel(surfel_index);
  
  // Make sure that the surfel is not reset again in this iteration.
  surfel->SetCanBeReset(false);
}

void SurfelMeshing::ConvertToPoint3fC3u8Cloud(Point3fC3u8Cloud* output) {
  output->Resize(surfels_.size() - merged_surfel_count_);
  usize index = 0;
  for (usize i = 0, size = surfels_.size(); i < size; ++ i) {
    const Surfel& surfel = surfels_[i];
    if (surfel.node() == nullptr) {
      continue;
    }
    (*output)[index] = Point3fC3u8(
        surfel.position(), Vec3u8(0, 0, 0));
    ++ index;
  }
  CHECK_EQ(index, output->size());
}

void SurfelMeshing::ConvertToMesh3fCu8(Mesh3fCu8* output, bool indices_only) {
  if (indices_only) {
    // Indices (with merged vertices included in the indexing).
    output->triangles_mutable()->resize(triangles_.size());
    usize out_index = 0;
    for (usize i = 0, size = triangles_.size(); i < size; ++ i) {
      const SurfelTriangle& st = triangles_[i];
      if (st.IsValid()) {
        output->triangles_mutable()->at(out_index) =
            Triangle<u32>(st.index(0), st.index(1), st.index(2));
        ++ out_index;
      }
    }
    output->triangles_mutable()->resize(out_index);
  } else {
    // Vertices.
    if (!indices_only) {
      output->vertices_mutable()->reset(new Point3fC3u8Cloud());
      ConvertToPoint3fC3u8Cloud(output->vertices().get());
    }
    
    std::vector<usize> index_remapping(surfels_.size());
    usize index = 0;
    for (usize i = 0, size = surfels_.size(); i < size; ++ i) {
      const Surfel& surfel = surfels_[i];
      if (surfel.node() == nullptr) {
        continue;
      }
      
      index_remapping[i] = index;
      ++ index;
    }
    CHECK_EQ(index, surfels_.size() - merged_surfel_count_);
    
    // Indices (adjusted to merged vertices being deleted).
    output->triangles_mutable()->resize(triangles_.size());
    usize out_index = 0;
    for (usize i = 0, size = triangles_.size(); i < size; ++ i) {
      const SurfelTriangle& st = triangles_[i];
      if (st.IsValid()) {
        output->triangles_mutable()->at(out_index) =
            Triangle<u32>(index_remapping[st.index(0)], index_remapping[st.index(1)], index_remapping[st.index(2)]);
        ++ out_index;
      }
    }
    output->triangles_mutable()->resize(out_index);
  }
}

}
