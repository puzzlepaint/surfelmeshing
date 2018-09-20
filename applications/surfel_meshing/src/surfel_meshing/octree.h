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

#include <unordered_set>

#include <glog/logging.h>
#include <libvis/eigen.h>
#include <libvis/libvis.h>

#include "surfel_meshing/surfel.h"

// Uncomment this to enable storing triangles in the octree. This is very slow.
// #define KEEP_TRIANGLES_IN_OCTREE

namespace vis {

// An octree node, containing up to 8 children and a list of surfels.
struct OctreeNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Constructor. Does not initialize the parent. This can be done by calling
  // AddChild() on the new node's parent.
  template <typename Derived>
  inline OctreeNode(const MatrixBase<Derived>& midpoint_, float half_extent_)
      : children{nullptr, nullptr, nullptr, nullptr,
                 nullptr, nullptr, nullptr, nullptr},
        midpoint(midpoint_),
        half_extent(half_extent_),
        min(midpoint_ - Vec3f::Constant(half_extent_)),
        max(midpoint_ + Vec3f::Constant(half_extent_)),
        child_count(0) {}
  
  // Adds a surfel to the list. Sets the surfel's node to this.
  inline void AddSurfel(u32 surfel_index, Surfel* surfel) {
    surfels.push_back(surfel_index);
    surfel->SetOctreeNode(this, surfels.size() - 1);
  }
  
  // Erases a surfel from the list.
  inline void EraseSurfel(u32 index, std::vector<Surfel>* surfel_vector) {
    if (surfels.size() == 1) {
      EraseAllSurfels();
    } else {
      if (index != surfels.size() - 1) {
        surfel_vector->at(surfels.back()).SetOctreeNodeIndex(index);
      }
      surfels[index] = surfels.back();
      surfels.pop_back();
    }
  }
  
  // Erases a range of surfels from this node. The index refers
  // to the index within the node's vector, not to the surfel index.
  // This function preserves the indices of the remaining surfels within the
  // node (if any).
  inline void EraseSurfels(u32 first_index, u32 last_index) {
    if (surfels.size() == 1) {
      EraseAllSurfels();
    } else if (last_index == surfels.size() - 1) {
      surfels.erase(surfels.begin() + first_index, surfels.end());
    } else {
      // Move entries from the end of the list into the gap until filled up or
      // no more entries left.
      for (int i = surfels.size() - 1; i > static_cast<int>(last_index); -- i) {
        surfels[first_index] = surfels[i];
        ++ first_index;
        if (first_index > last_index) {
          // The hole was filled.
          break;
        }
      }
      surfels.erase(surfels.begin() + first_index, surfels.end());
    }
  }
  
  // Erases all surfels from this node.
  inline void EraseAllSurfels() {
    surfels.clear();
    surfels.shrink_to_fit();
  }
  
  // Adds a child to this node. Assumes that no child was set before for the
  // same child_index.
  inline void AddChild(OctreeNode* child, int child_index) {
    child->parent = this;
    // CHECK(children[child_index] == nullptr);
    children[child_index] = child;
    ++ child_count;
    CheckChildCount();
  }
  
  // Creates and adds a child to this node. level must be at least 1: the
  // child's extent will be pow(2, -level) * this->extent.
  template <typename Derived>
  inline OctreeNode* CreateChild(int level, const MatrixBase<Derived>& point_in_child, int* new_node_child_index, bool add_as_child) {
    unsigned int half_fraction = 1 << (level - 1);
    float new_node_extent = half_extent / half_fraction;
    float new_node_extent_inv = 1.0f / new_node_extent;
    unsigned int new_node_x = (point_in_child.x() - min.x()) * new_node_extent_inv;
    unsigned int new_node_y = (point_in_child.y() - min.y()) * new_node_extent_inv;
    unsigned int new_node_z = (point_in_child.z() - min.z()) * new_node_extent_inv;
    
    // Robustify: ensure that the node is created within the parent node. If
    // this is not done, points that are slightly beyond the border of a node
    // and still added to it can cause the attempt to create children outside
    // the node. This check could be dropped if we ensured that only points
    // within a node are added to it.
    // We do not check for (slightly) negative coordinates since those would be
    // rounded to zero above.
    unsigned int size = 1 << level;
    new_node_x = std::min(new_node_x, size - 1);
    new_node_y = std::min(new_node_y, size - 1);
    new_node_z = std::min(new_node_z, size - 1);
    
    int plus_x = (new_node_x >= half_fraction) * 2 - 1;  // (new_node_x >= half_fraction) ? 1 : -1;
    int plus_y = (new_node_y >= half_fraction) * 2 - 1;
    int plus_z = (new_node_z >= half_fraction) * 2 - 1;
    
    float new_node_half_extent = 0.5f * new_node_extent;
    Vec3f new_node_midpoint =
        min +
        new_node_extent * Vec3f(new_node_x, new_node_y, new_node_z) +
        Vec3f::Constant(new_node_half_extent);
    OctreeNode* new_node = new OctreeNode(new_node_midpoint,
                                          new_node_half_extent);
    
    *new_node_child_index =
        ((plus_x == 1) * (1 << 0)) |  // (plus_x == 1) ? (1 << 0) : 0
        ((plus_y == 1) * (1 << 1)) |
        ((plus_z == 1) * (1 << 2));
    if (add_as_child) {
      AddChild(new_node, *new_node_child_index);
    }
    
    return new_node;
  }
  
#ifdef KEEP_TRIANGLES_IN_OCTREE
  inline void AddTriangle(u32 triangle_index) {
    triangles.insert(triangle_index);
  }
  
  inline void EraseTriangle(u32 triangle_index) {
    triangles.erase(triangle_index);
  }
#endif
  
  inline bool IsEmpty() const {
    return surfels.empty();
  }
  
  inline bool IsLeaf() const {
    return child_count == 0;
  }
  
  inline Vec3f GetQuarterMidpoint(int child_index) const {
    return midpoint + (0.5f * half_extent) *
           Vec3f((child_index & (1 << 0)) ? 1 : -1,
                 (child_index & (1 << 1)) ? 1 : -1,
                 (child_index & (1 << 2)) ? 1 : -1);
  }
  
  template <typename Derived>
  inline bool Contains(const MatrixBase<Derived>& point) const {
    return point.x() >= min.x() &&
           point.y() >= min.y() &&
           point.z() >= min.z() &&
           point.x() < max.x() &&
           point.y() < max.y() &&
           point.z() < max.z();
  }
  
  inline bool Contains(const Eigen::AlignedBox<float, 3>& box) const {
    return min.x() <= box.min().x() &&
           min.y() <= box.min().y() &&
           min.z() <= box.min().z() &&
           max.x() >= box.max().x() &&
           max.y() >= box.max().y() &&
           max.z() >= box.max().z();
  }
  
  template <typename Derived>
  inline int ComputeChildIndex(const MatrixBase<Derived>& point) const {
    return ((point.x() >= midpoint.x()) * (1 << 0)) |  // (point.x() >= midpoint.x()) ? (1 << 0) : 0
           ((point.y() >= midpoint.y()) * (1 << 1)) |
           ((point.z() >= midpoint.z()) * (1 << 2));
  }
  
  template <typename Derived>
  inline Vec3f ClosestPointTo(const MatrixBase<Derived>& point) const {
    return Vec3f((point.x() < min.x()) ? min.x() : ((point.x() > max.x()) ? max.x() : point.x()),
                 (point.y() < min.y()) ? min.y() : ((point.y() > max.y()) ? max.y() : point.y()),
                 (point.z() < min.z()) ? min.z() : ((point.z() > max.z()) ? max.z() : point.z()));
  }
  
  // For debugging only.
  friend std::ostream& operator<<(std::ostream& os, const OctreeNode& node);
  
  // For debugging only (slow).
  usize CountSurfelsRecursive() const;
  
  // For debugging only.
  void CheckContains(OctreeNode* other, float epsilon) const {
    // Check that the coordinates are not outside the cell.
    CHECK_GE(other->min.x() + epsilon, min.x());
    CHECK_GE(other->min.y() + epsilon, min.y());
    CHECK_GE(other->min.z() + epsilon, min.z());
    CHECK_LE(other->max.x() - epsilon, max.x());
    CHECK_LE(other->max.y() - epsilon, max.y());
    CHECK_LE(other->max.z() - epsilon, max.z());
    
    // Check that the other cell extents match that of a valid child node of
    // this one:
    
    // Check that the extent of the child is approximately equal to
    // pow(2, -x) * parent_extent for some integer x >= 1.
    float log_result = std::log2f(half_extent / other->half_extent);
    int int_log = static_cast<int>(log_result + 0.5f);  // Round.
    CHECK_LE(fabs(log_result - int_log), 0.01f);
    CHECK_GE(int_log, 1) << "half_extent: " << half_extent << ", other->half_extent: " << other->half_extent;
  }
  
  // For debugging only. Uncomment the implementation below to activate the
  // check.
  inline void CheckChildCount() const {}
//   inline void CheckChildCount() const {
//     u8 actual_child_count = 0;
//     for (int i = 0; i < 8; ++ i) {
//       if (children[i] != nullptr) {
//         ++ actual_child_count;
//       }
//     }
//     CHECK_EQ(actual_child_count, child_count);
//   }
  
  
  // Parent node. The root node's parent is set to nullptr.
  OctreeNode* parent;
  
  // Ordering:
  // 0 : -x, -y, -z
  // 1 : +x, -y, -z
  // 2 : -x, +y, -z
  // 3 : +x, +y, -z
  // 4 : -x, -y, +z
  // 5 : +x, -y, +z
  // 6 : -x, +y, +z
  // 7 : +x, +y, +z
  // Pointer owned by this struct.
  OctreeNode* children[8];
  
  // Center point of the node.
  Vec3f midpoint;
  
  // Half extent of the node, going from the midpoint in all axis-aligned
  // directions. A point is said to be within a node if its coordinates p are:
  // p >= node_min and
  // p < node_max.
  float half_extent;
  
  // Minimum and maximum coordinates of the octree node. While this is redundant
  // with (midpoint, half_extent) in principle, we use the policy to always use
  // (min, max) for inclusion testing to prevent different result coming from
  // numerical inaccuracies.
  Vec3f min;
  Vec3f max;
  
  // List of surfels anywhere within this octree node.
  vector<u32> surfels;
  
#ifdef KEEP_TRIANGLES_IN_OCTREE
  // List of triangles whose bounding box intersects this octree node.
  unordered_set<u32> triangles;
#endif
  
  // Number of non-null children.
  u8 child_count;
};

inline std::ostream& operator<<(std::ostream& os, const OctreeNode& node) {
  return os << "[OctreeNode min: " << node.min.transpose()
            << ", max: " << node.max.transpose()
            << ", midpoint: " << node.midpoint.transpose() << "]";
}


// Manages a compressed octree.
class CompressedOctree {
 public:
  CompressedOctree(usize max_surfels_per_node,
                   std::vector<Surfel>* surfels,
                   vector<SurfelTriangle>* triangles);
  
  ~CompressedOctree();
  
  
  // Addition / removal / moving.
  
  // Lazy default version of AddSurfel():
  // - If the surfel is within the root node, adds it to the root node.
  // - Otherwise, expands the root node suitably.
  // However, this function does not sort the surfel down into a leaf.
  void AddSurfel(u32 surfel_index, Surfel* surfel);
  
  // Active version of AddSurfel() which sorts the surfel downwards.
  void AddSurfelActive(u32 surfel_index, Surfel* surfel);
  
  // Also removes all triangles connected to the surfel. These can be quickly
  // located once the surfel's octree node is found.
  void RemoveSurfel(u32 surfel_index);
  
  // Also moves all triangles connected to the surfel. These can be quickly
  // located once the surfel's octree node is found. Note that the surfel
  // must still have its old position assigned when this is called.
  template <typename Derived>
  void MoveSurfel(u32 surfel_index, Surfel* surfel, const MatrixBase<Derived>& new_pos) {
    // CHECK_EQ(surfel, &surfels_->at(surfel_index));
    
    OctreeNode* old_node = surfel->node();
    usize index = surfel->index_in_node();  // Save index before the new one is assigned.
    
    // Move the surfel as far up as necessary.
    // TODO: This could maybe be faster. We don't need to repeatedly call the full
    //       Contains() function since the nodes only get larger. So, if a test
    //       for a given dimension succeeds for a node, then it will also succeed
    //       for the parent nodes.
    OctreeNode* node = old_node;
    while (node && !node->Contains(new_pos)) {
      node = node->parent;
    }
    
    OctreeNode* new_surfel_node = node;
    if (node == old_node) {
      // The surfel's node did not change. Do nothing.
    } else if (node) {
      // Insert the surfel into its new node.
      node->AddSurfel(surfel_index, surfel);
    } else {
      // The surfel's new position is outside of the root node. Extend the root
      // node.
      new_surfel_node = ExtendRootForSurfel(surfel_index, new_pos);
    }
    
    // Move the triangles connected to the surfel.
    #ifdef KEEP_TRIANGLES_IN_OCTREE
      for (int triangle_index_in_surfel = 0, end = surfel->GetTriangleCount();
          triangle_index_in_surfel < end; ++ triangle_index_in_surfel) {
        u32 triangle_index = surfel->GetTriangle(triangle_index_in_surfel);
        const SurfelTriangle& triangle = triangles_->at(triangle_index);
        // CHECK(triangle.index(0) == surfel_index ||
        //       triangle.index(1) == surfel_index ||
        //       triangle.index(2) == surfel_index);
        
        // Compute the old bounding box of the triangle.
        Eigen::AlignedBox<float, 3> old_triangle_bbox(surfels_->at(triangle.index(0)).position());
        old_triangle_bbox.extend(surfels_->at(triangle.index(1)).position());
        old_triangle_bbox.extend(surfels_->at(triangle.index(2)).position());
        
        // If the new surfel position is within the old bbox, no action is
        // necessary. In case a node referencing the triangle does not intersect
        // it anymore, the next triangle search touching the node will delete the
        // reference.
        if (!old_triangle_bbox.contains(new_pos)) {
          // The new bbox extends to a place which the old bbox didn't extend to.
          // (This does not mean that it becomes larger.)
          // Compute the new bbox.
          Eigen::AlignedBox<float, 3> new_triangle_bbox;
          for (int i = 0; i < 3; ++ i) {
            if (triangle.index(i) == surfel_index) {
              new_triangle_bbox.extend(new_pos);
            } else {
              new_triangle_bbox.extend(surfels_->at(triangle.index(i)).position());
            }
          }
          
          // This could also be used here but is not necessary:
          // RemoveTriangle(triangle_index, triangle);
          
          // We need to make sure that the nodes referencing the triangle still
          // cover all of its bbox's volume.
          // NOTE: I'm not sure at all what the best strategy is here:
          //       * One could first check whether the current vertices' nodes
          //         already cover the changed bbox (no action necessary).
          //       * One could simply insert the triangle at the root again.
          //       * One could go upwards from any of the vertices until a node
          //         is found that covers the triangle bbox completely, and insert
          //         the triangle there. (<-- current strategy)
          //       * One could go upwards from all 3 vertices and check for a
          //         combined coverage of the triangle bbox by the (up to) 3
          //         nodes. This avoids having to go up very far if the triangle
          //         lies on an early decision boundary, but the checks would be
          //         slower.
          //       In addition, one could also explicitly remove the triangle from
          //       the old_node if it does not intersect the new bbox anymore.
          OctreeNode* new_triangle_node = new_surfel_node;
          while (new_triangle_node != root_ &&
                  !new_triangle_node->Contains(new_triangle_bbox)) {
            new_triangle_node = new_triangle_node->parent;
          }
          
          new_triangle_node->AddTriangle(triangle_index);
        }
      }
    #else
      (void) new_surfel_node;
    #endif
    
    // Remove the surfel from its old node and potentially delete it (and its
    // parent if it becomes a one-child node).
    // CHECK_EQ(*(old_node->surfels.begin() + index), surfel_index);
    if (node != old_node) {
      old_node->EraseSurfel(index, surfels_);
      if (old_node->IsEmpty()) {
        if (old_node->IsLeaf()) {
          RemoveEmptyLeaf(old_node);
        } else if (old_node->child_count == 1) {
          RemoveSingleChildNode(old_node);
        }
      }
    }
  }
  
#ifdef KEEP_TRIANGLES_IN_OCTREE
  void AddTriangle(u32 triangle_index, const SurfelTriangle& triangle);
  
  // At the point this function is called, the triangle vertices must still
  // exist.
  void RemoveTriangle(u32 triangle_index, const SurfelTriangle& triangle);
#endif
  
  
  // Queries.
  
  // Actively sorts down surfels into new nodes if the maximum surfel count in
  // a node that is searched is violated.
  template <bool include_completed_surfels, bool include_free_surfels>
  int FindNearestSurfelsWithinRadius(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices);
  
  // Version of FindNearestSurfelsWithinRadius() which leaves the octree
  // constant. Only use this if all surfels have been inserted with
  // AddSurfelActive(), otherwise it might be extremely slow.
  template <bool include_completed_surfels, bool include_free_surfels>
  int FindNearestSurfelsWithinRadiusPassive(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) const;
  
#ifdef KEEP_TRIANGLES_IN_OCTREE
  inline void FindNearestTrianglesIntersectingBox(const Vec3f& min, const Vec3f& max, vector<u32>* result_indices) {  // Unlimited result count
    FindNearestTrianglesIntersectingBoxImpl(min, max, result_indices);
  }
#endif
  
  inline void FindNearestTrianglesViaSurfels(const Vec3f& position, float radius_squared, int max_surfel_count, vector<u32>* result_indices) {  // Unlimited result count
    FindNearestTrianglesViaSurfelsImpl(position, radius_squared, max_surfel_count, result_indices);
  }
  
  // NOTE: Frustum culling queries for surfels (and triangle blocks?) might
  //       also be useful.
  
  
  // For debugging.
  
  usize numerical_issue_counter() const { return numerical_issue_counter_; }
  
  usize CountSurfelsSlow();
  
  bool FindSurfelAnywhereSlow(u32 surfel_index, const Surfel& surfel, OctreeNode* start_node, OctreeNode** node, usize* index) const;
  
  inline OctreeNode* root() const { return root_; }
  
  
 private:
  // Given a surfel and the node in which it currently is in, move it downwards
  // in the octree as far as possible. Potentially creates a new leaf node, but
  // does not create any other nodes. Does not delete the surfel from
  // the node's surfel list.
  void SortSurfelDownwards(OctreeNode* node, u32 surfel_index);
  
  // Sorts all surfels in the given node (at least) one step downwards in the
  // tree. To be called on nodes that violate the max surfel count. As a
  // result of the operation the node might get replaced by another one. The
  // function thus returns the resulting node.
  OctreeNode* SortSurfelsInNodeDownwardsOneStep(OctreeNode* node);
  
  // Creates a root node containing the given surfel (with arbitrary, non-zero
  // extent).
  void CreateRootForSurfel(u32 surfel_index, Surfel* surfel);
  
  // Creates a new root node to extend the octree area to the given surfel,
  // which is outside the current root. Inserts the surfel into a new child of
  // the new root and returns this child node.
  template <typename Derived>
  OctreeNode* ExtendRootForSurfel(u32 surfel_index, const MatrixBase<Derived>& position) {
    // Create a new root node containing the old root node and the new surfel.
    // First, find the direction of the new surfel relative to the old root.
    int plus_x = (position.x() < root_->min.x()) ? -1 : 1;
    int plus_y = (position.y() < root_->min.y()) ? -1 : 1;
    int plus_z = (position.z() < root_->min.z()) ? -1 : 1;
    
    // Measure the distance from the opposite corner of the old root to the
    // new surfel.
    Vec3f opposite_corner((plus_x == 1) ? root_->min.x() : root_->max.x(),
                          (plus_y == 1) ? root_->min.y() : root_->max.y(),
                          (plus_z == 1) ? root_->min.z() : root_->max.z());
    float dist_x = fabs(position.x() - opposite_corner.x());
    float dist_y = fabs(position.y() - opposite_corner.y());
    float dist_z = fabs(position.z() - opposite_corner.z());
    float dist_max = std::max(std::max(dist_x, dist_y), dist_z);
    
    // How large does the new root need to be to contain the new surfel,
    // while being compatible to the old root?
    float old_root_extent = 2 * root_->half_extent;
    // Compute the following in a fast way:
    // int level = floor(std::log2f(dist_max / old_root_extent));
    unsigned int rounded_factor = dist_max / old_root_extent;  // Round down in cast to int.
    if (rounded_factor == 0) {
      rounded_factor = 1;
    }
    // Count leading zeros in rounded_factor and invert the result to get
    // the number of "occupied" bits. Subtract 1 to get log2(rounded_factor).
    // Note that __builtin_clz() is undefined if its argument is 0.
    int level = (8 * sizeof(unsigned int)) - __builtin_clz(rounded_factor) - 1;
    
    // Compute the new midpoint.
    float midpoint_dist = old_root_extent * (1 << level);
    Vec3f new_root_midpoint = opposite_corner + Vec3f(
        plus_x * midpoint_dist,
        plus_y * midpoint_dist,
        plus_z * midpoint_dist);
    
    // Create the new root.
    OctreeNode* new_root = new OctreeNode(new_root_midpoint, midpoint_dist);
    new_root->parent = nullptr;
    
    // This is flipped because plus_xyz denotes the direction as viewed from
    // the old root, so the direction from the new to the old one is the
    // inverse.
    int old_root_child_index = ((plus_x == 1) ? 0 : (1 << 0)) |
                              ((plus_y == 1) ? 0 : (1 << 1)) |
                              ((plus_z == 1) ? 0 : (1 << 2));
    new_root->AddChild(root_, old_root_child_index);
    
    root_ = new_root;
    
    // Create a child of the new root, containing the added surfel.
    int child_plus_x = (position.x() >= new_root->midpoint.x()) ? 1 : -1;
    int child_plus_y = (position.y() >= new_root->midpoint.y()) ? 1 : -1;
    int child_plus_z = (position.z() >= new_root->midpoint.z()) ? 1 : -1;
    
    float child_half_extent = 0.5f * new_root->half_extent;
    Vec3f child_midpoint = new_root->midpoint + child_half_extent * Vec3f(child_plus_x, child_plus_y, child_plus_z);
    OctreeNode* child = new OctreeNode(child_midpoint, child_half_extent);
    child->AddSurfel(surfel_index, &surfels_->at(surfel_index));
    int child_node_child_index = ((child_plus_x == 1) ? (1 << 0) : 0) |
                                ((child_plus_y == 1) ? (1 << 1) : 0) |
                                ((child_plus_z == 1) ? (1 << 2) : 0);
    if (old_root_child_index == child_node_child_index) {
      // There is a numerical problem. Ideally, this case should not happen.
      // Delete the new root again and insert the surfel into the old root.
      //LOG(ERROR) << "Numerical problem in ExtendRootForSurfel()";
      // NOTE: This case happens for NaN surfel positions.
      //LOG(ERROR) << "Old root: " << *new_root->children[old_root_child_index];
      //LOG(ERROR) << "Old root contains surfel (should be false!): " << new_root->children[old_root_child_index]->Contains(position);
      //LOG(ERROR) << "Surfel position: " << position;
      //CHECK(false);
      root_ = new_root->children[old_root_child_index];
      root_->parent = nullptr;
      root_->AddSurfel(surfel_index, &surfels_->at(surfel_index));
      delete new_root;
      delete child;
    } else {
      new_root->AddChild(child, child_node_child_index);
    }
    return child;
  }
  
  template <typename Derived>
  bool InsertIntermediateLevel(const MatrixBase<Derived>& position, OctreeNode* node, int existing_child_index, OctreeNode* existing_child, OctreeNode** new_node) {
    // level == 1 means that the new level is directly above the child level,
    // level == 2 means that it is one step further up, etc.
    int level = GetMinLevelContaining(existing_child->midpoint,
                                      position,
                                      node->min,
                                      2 * existing_child->half_extent);
    if (level <= 0) {
      // Should not happen.
      return false;
    }
    
    // Check for robustness: The intermediate node must be smaller than the
    // containing node.
    if (existing_child->half_extent * (1 << level) >= 0.75f * node->half_extent) {
      return false;
    }
    
    float new_node_extent = existing_child->half_extent * (2 << level);
    float new_node_extent_inv = 1.0f / new_node_extent;
    // The choice of the point within the new node should be arbitrary here.
    unsigned int new_node_x = (existing_child->midpoint.x() - node->min.x()) * new_node_extent_inv;
    unsigned int new_node_y = (existing_child->midpoint.y() - node->min.y()) * new_node_extent_inv;
    unsigned int new_node_z = (existing_child->midpoint.z() - node->min.z()) * new_node_extent_inv;
    
    float new_node_half_extent = 0.5f * new_node_extent;
    Vec3f new_node_midpoint =
        node->min +
        new_node_extent * Vec3f(new_node_x, new_node_y, new_node_z) +
        Vec3f::Constant(new_node_half_extent);
    *new_node =
        new OctreeNode(new_node_midpoint, new_node_half_extent);
    
    // Connect the parent and the new node.
    node->children[existing_child_index] = *new_node;
    (*new_node)->parent = node;
    
    // Connect the new node and the child.
    int child_index_within_new_node = (*new_node)->ComputeChildIndex(existing_child->midpoint);
    (*new_node)->AddChild(existing_child, child_index_within_new_node);
    
    return true;
  }
  
  // Assuming an octree whose cells have the given extent on octree level 0 and
  // which has the minimum bounding box coordinates at min, computes the smallest
  // octree level for which a node contains both point_a and point_b.
  // For example, level 1 would have twice the cell extent, level 2 four times
  // the cell extent, and so on.
  template <typename DerivedA, typename DerivedB, typename DerivedC>
  static inline int GetMinLevelContaining(
      const MatrixBase<DerivedA>& point_a,
      const MatrixBase<DerivedB>& point_b,
      const MatrixBase<DerivedC>& min,
      float cell_extent) {
    float extent_inv = 1.0f / cell_extent;
    
    unsigned int max_differences = 0;
    for (int d = 0; d < 3; ++ d) {
      unsigned int grid_coord_a = (point_a.coeff(d) - min.coeff(d)) * extent_inv;
      unsigned int grid_coord_b = (point_b.coeff(d) - min.coeff(d)) * extent_inv;
      unsigned int differences = grid_coord_a ^ grid_coord_b;  // XOR operator.
      if (differences > max_differences) {
        max_differences = differences;
      }
    }

    if (max_differences == 0) {
      return 0;
    }
    
    return (8 * sizeof(unsigned int)) - __builtin_clz(max_differences);
  }
  
  // Deletes a node with only a single child and puts the child into its place.
  // Returns the child.
  OctreeNode* RemoveSingleChildNode(OctreeNode* node);
  
  // Deletes an empty leaf node and potentially removes the parent if it turns
  // into a single-child node then.
  void RemoveEmptyLeaf(OctreeNode* node);
  
  // Deletes a node and (recursively) its children. node must not be null.
  void DeleteNode(OctreeNode* node);
  
#ifdef KEEP_TRIANGLES_IN_OCTREE
  void FindNearestTrianglesIntersectingBoxImpl(const Vec3f& min, const Vec3f& max, vector<u32>* result_indices);  // Unlimited result count
#endif
  
  void FindNearestTrianglesViaSurfelsImpl(const Vec3f& position, float radius_squared, int max_surfel_count, vector<u32>* result_indices);  // Unlimited result count
  
  
  // Pointer owned by this class.
  OctreeNode* root_;
  
  // Maximum number of surfels in a single node. If more surfels fall into the
  // same node, it is split up.
  usize max_surfels_per_node_;
  
  // List of surfels. Pointer to external data, not owned.
  vector<Surfel>* surfels_;
  
  // List of triangles. Pointer to external data, not owned.
  vector<SurfelTriangle>* triangles_;
  
  // Is used temporarily only, but cached here to avoid re-allocation.
  mutable vector<OctreeNode*> nodes_to_search_;
  vector<float> surfel_distances_squared_;
  vector<u32> surfel_indices_;
  
  // Debug counter.
  usize numerical_issue_counter_;
};

}
