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

#include "surfel_meshing/octree.h"

#include <libvis/logging.h>
#include <libvis/timing.h>

namespace vis {

usize OctreeNode::CountSurfelsRecursive() const {
  usize result = surfels.size();
  for (int i = 0; i < 8; ++ i) {
    if (children[i]) {
      result += children[i]->CountSurfelsRecursive();
    }
  }
  return result;
}


CompressedOctree::CompressedOctree(
    usize max_surfels_per_node,
    std::vector<Surfel>* surfels,
    vector<SurfelTriangle>* triangles) {
  root_ = nullptr;
  max_surfels_per_node_ = max_surfels_per_node;
  surfels_ = surfels;
  triangles_ = triangles;
  
  numerical_issue_counter_ = 0;
}

CompressedOctree::~CompressedOctree() {
  if (root_) {
    DeleteNode(root_);
  }
}

void CompressedOctree::AddSurfel(u32 surfel_index, Surfel* surfel) {
  // Check if the root node has to be created.
  if (!root_) {
    CreateRootForSurfel(surfel_index, surfel);
    return;
  }
  
  // Check whether the surfel is outside the root node.
  if (!root_->Contains(surfel->position())) {
    ExtendRootForSurfel(surfel_index, surfel->position());
    return;
  }
  
  // The new surfel is within the root node's extent. Lazily add it to the root
  // node's surfel list (it will be sorted down when needed).
  root_->AddSurfel(surfel_index, surfel);
}

void CompressedOctree::AddSurfelActive(u32 surfel_index, Surfel* surfel) {
  // Check if the root node has to be created.
  if (!root_) {
    CreateRootForSurfel(surfel_index, surfel);
    return;
  }
  
  // Check whether the surfel is outside the root node.
  if (!root_->Contains(surfel->position())) {
    ExtendRootForSurfel(surfel_index, surfel->position());
    return;
  }
  
  // The new surfel is within the root node's extent. Locate the smallest
  // node containing the surfel.
  OctreeNode* node = root_;
  int child_index = 0;
  while (true) {
    child_index = node->ComputeChildIndex(surfel->position());
    OctreeNode* child = node->children[child_index];
    
    if (child == nullptr) {
      // Found the smallest node.
      break;
    } else if (child->Contains(surfel->position())) {
      // Descend.
      node = child;
    } else {
      // There is a child for this quarter, but it does not contain the new
      // surfel. Insert a new level in-between. We know that the new level
      // will be between the parent and the child level, so we can start by
      // determining the coordinates of the old child and the new surfel
      // in units of the old child size, within the parent. Then looking
      // at the bit representation of the coordinates gives us the suitable
      // in-between level. This is similar to one of the cases below.
      OctreeNode* new_node;
      if (!InsertIntermediateLevel(surfel->position(), node, child_index, child, &new_node)) {
        LOG(WARNING) << "Encountered a case which should not happen.";
        node->AddSurfel(surfel_index, surfel);
        return;
      }
      
      // Insert the new surfel into a new quarter leaf of the new node.
      SortSurfelDownwards(new_node, surfel_index);
      
      return;
    }
  }
  
  if (node->IsLeaf()) {
    // If the maximum surfel count for this node is not reached yet, add the
    // new surfel to this node.
    if (node->surfels.size() < max_surfels_per_node_) {
      node->AddSurfel(surfel_index, surfel);
    } else {
      // There are too many surfels in this leaf node. Split up the node
      // such that at least one surfel gets split into a different child
      // than the others:
      // * Determine the bounding box of the contained surfels.
      // * Compute the smallest node that contains the box and is compatible
      //   with the containing node. This neccessarily splits up the
      //   surfels (into different ones of its quarters), since any smaller
      //   node would not contain the whole box anymore.
      // * Insert this node (if it is not equal to the containing node).
      // * Sort all surfels in the containing node into the new nodes
      //   (and possibly additional new quarter leaves).
      // * If the containing node only contains one child as a result,
      //   remove it (and directly use its child).
      Eigen::AlignedBox<float, 3> bbox;
      for (usize i = 0, size = node->surfels.size(); i < size; ++ i) {
        u32 index = node->surfels[i];
        bbox.extend(surfels_->at(index).position());
      }
      bbox.extend(surfel->position());
      
      // Compute the level having the largest cells such that the surfels
      // can definitely be separated (from the x, y, z distances of the
      // surfels: choose the cell size to be at most the minimum distance).
      float dist_x = bbox.max().x() - bbox.min().x();
      float dist_y = bbox.max().y() - bbox.min().y();
      float dist_z = bbox.max().z() - bbox.min().z();
      float dist_max;
      if (dist_x > dist_y) {
        if (dist_x > dist_z) {
          dist_max = dist_x;
        } else {
          dist_max = dist_z;
        }
      } else {
        if (dist_y > dist_z) {
          dist_max = dist_y;
        } else {
          dist_max = dist_z;
        }
      }
      
      if (dist_max == 0) {
        // We cannot separate these surfels. Give up and violate the maximum
        // surfel count (the next addition to this node will try again).
        // LOG(WARNING) << "Cannot separate surfels in a full cell, violating maximum surfel count";
        node->AddSurfel(surfel_index, surfel);
      } else {
        float node_extent = 2 * node->half_extent;
        
        // Compute floor(-1 * std::log2f(dist_max / node_extent)) in a fast way.
        // Slow code:
        // int level = -1 * std::log2f(dist_max / node_extent);  // Round down in cast to int.
        
        // Values of rounded_factor:
        // For dist_max in ]0.5, 1[ * node_extent: 1
        // For dist_max in ]0.25, 0.5] * node_extent: 2 or 3
        // For dist_max in ]0.125, 0.25] * node_extent: 4 or 5 or 6 or 7, ...
        unsigned int rounded_factor = node_extent / dist_max;  // Round down in cast to int.
        if (rounded_factor == 0) {
          rounded_factor = 1;
        }
        // Count leading zeros in rounded_factor and invert the result to get
        // the number of "occupied" bits. Subtract 1 to get log2(rounded_factor).
        // Note that __builtin_clz() is undefined if its argument is 0.
        int level = (8 * sizeof(unsigned int)) - __builtin_clz(rounded_factor) - 1;
        
        // level is 0 if sorting the surfels into the quarters of the current
        // node will definitely separate them (their max distance is in
        // ]half_extent, 2 * half_extent[), 1 if a new node within a quarter
        // will definitely separate them (their max distance is in
        // [half_extent, 0.5 * half_extent[), etc.
        OctreeNode* current_node = node;
        if (level > 0) {
          float min_cell_quarter_extent = node_extent / (1 << (level + 1));
          int level_difference_plus_one = 
              GetMinLevelContaining(bbox.min(), bbox.max(), node->min,
                                    min_cell_quarter_extent);
          
          if (level_difference_plus_one == 0) {
            // Should not happen. Give up and violate the maximum
            // surfel count (the next addition to this node will try again).
            // LOG(WARNING) << "Cannot split up a full node because the surfels are still in the same node on the computed resolution. Violating the maximum surfel count.";
            node->AddSurfel(surfel_index, surfel);
            return;
          } else {
            level -= level_difference_plus_one - 1;
            
            if (level > 0) {
              // Create the new node on the level that was computed.
              int current_node_child_index;
              current_node = node->CreateChild(level, bbox.min(), &current_node_child_index, /*add_as_child*/ true);
            }
          }
        }
        
        // Sort all surfels into current_node, respectively additional new
        // quarter leaves in current_node and the containing node.
        usize size = node->surfels.size();
        for (usize i = 0; i < size; ++ i) {
          u32 index = node->surfels[i];
          SortSurfelDownwards(current_node, index);
        }
        node->EraseSurfels(0, size - 1);  // Keep any surfels which may have been added to the back.
        SortSurfelDownwards(current_node, surfel_index);
        
        // If the containing node only contains one child as a result,
        // remove it (and directly use its child).
        if (node->child_count == 1 && node->IsEmpty()) {
          RemoveSingleChildNode(node);
        }
      }
    }
  } else {
    // Create a new leaf node in a previously unoccupied quarter.
    OctreeNode* child = new OctreeNode(node->GetQuarterMidpoint(child_index),
                                       0.5f * node->half_extent);
    node->AddChild(child, child_index);
    
    child->AddSurfel(surfel_index, surfel);
  }
}

void CompressedOctree::RemoveSurfel(u32 surfel_index) {
  Surfel* surfel = &surfels_->at(surfel_index);
  OctreeNode* node = surfel->node();
  
  #ifdef KEEP_TRIANGLES_IN_OCTREE
    // Remove the triangles connected to the surfel.
    for (int triangle_index_in_surfel = 0, end = surfel->GetTriangleCount();
        triangle_index_in_surfel < end; ++ triangle_index_in_surfel) {
      u32 triangle_index = surfel->GetTriangle(triangle_index_in_surfel);
      const SurfelTriangle& triangle = triangles_->at(triangle_index);
      if (triangle.index(0) == surfel_index ||
          triangle.index(1) == surfel_index ||
          triangle.index(2) == surfel_index) {
        RemoveTriangle(triangle_index, triangle);
      }
    }
  #endif
  
  // Remove the surfel.
  node->EraseSurfel(surfel->index_in_node(), surfels_);
  if (node->IsEmpty()) {
    if (node->IsLeaf()) {
      RemoveEmptyLeaf(node);
    } else if (node->child_count == 1) {
      RemoveSingleChildNode(node);
    }
  }
}

#ifdef KEEP_TRIANGLES_IN_OCTREE
void CompressedOctree::AddTriangle(u32 triangle_index, const SurfelTriangle& /*triangle*/) {
  if (!root_) {
    LOG(ERROR) << "AddTriangle() called, but no root node exists. This must not happen since at least 3 surfels (the triangle corners) should be in the octree. Not adding the triangle.";
    return;
  }
  root_->AddTriangle(triangle_index);
}

void CompressedOctree::RemoveTriangle(u32 triangle_index, const SurfelTriangle& triangle) {
  // Remove the triangle from the (up to 3) node(s) of its vertices.
  // This might not remove all references to it from the octree, but we don't
  // care about that (much): They will be removed later.
  surfels_->at(triangle.index(0)).node()->EraseTriangle(triangle_index);
  surfels_->at(triangle.index(1)).node()->EraseTriangle(triangle_index);
  surfels_->at(triangle.index(2)).node()->EraseTriangle(triangle_index);
}
#endif

// Returns the next node, or nullptr if the caller should break out of the loop.
template <bool include_completed_surfels, bool include_free_surfels>
inline OctreeNode* FindNearestSurfelsWithinRadiusImpl(
    OctreeNode* node,
    vector<OctreeNode*>& nodes_to_search,
    const Vec3f& position,
    float radius_squared,
    int max_result_count,
    float* result_distances_squared,
    u32* result_indices,
    int* result_count,
    float* max_distance_squared,
    const Surfel* surfels) {
  // Look for results in the current node.
  for (usize node_surfel_index = 0, size = node->surfels.size();
      node_surfel_index < size;
      ++ node_surfel_index) {
    u32 surfel_index = node->surfels[node_surfel_index];
    if (!include_completed_surfels && surfels[surfel_index].meshing_state() == Surfel::MeshingState::kCompleted) {
      continue;
    }
    if (!include_free_surfels && surfels[surfel_index].meshing_state() == Surfel::MeshingState::kFree) {
      continue;
    }
    float distance_squared =
        (surfels[surfel_index].position() - position).squaredNorm();
    if (distance_squared <= *max_distance_squared) {
      // Consider adding this to the results.
      if (*result_count < max_result_count) {
        ++ *result_count;
      }
      int i;
      for (i = *result_count - 1; i > 0; -- i) {
        if (result_distances_squared[i - 1] > distance_squared) {
          result_distances_squared[i] = result_distances_squared[i - 1];
          result_indices[i] = result_indices[i - 1];
        } else {
          break;
        }
      }
      result_distances_squared[i] = distance_squared;
      result_indices[i] = node->surfels[node_surfel_index];
      if (*result_count == max_result_count) {
        *max_distance_squared = result_distances_squared[max_result_count - 1];
      }
    }
  }
  
  // Remember relevant child nodes other than the one in the same quarter as
  // the query point.
  int child_index = node->ComputeChildIndex(position);
  
  Vec3f mid_dists_squared = (position - node->midpoint).array().square();  // Component-wise squaring.
  
  int index;
  if (mid_dists_squared.x() > radius_squared) {
    if (mid_dists_squared.y() > radius_squared) {
      if (mid_dists_squared.z() > radius_squared) {
        // We only need to check the child in the query's quarter.
      } else {
        // The children with the same x and y, but both z must be checked.
        index = child_index ^ (1 << 2); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      }
    } else {
      if (mid_dists_squared.z() > radius_squared) {
        // The children with the same x and z, but both y must be checked.
        index = child_index ^ (1 << 1); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      } else {
        // The children with the same x but both y and z must be checked.
        index = child_index ^ (1 << 2); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ (1 << 1); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 1) | (1 << 2)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      }
    }
  } else {
    if (mid_dists_squared.y() > radius_squared) {
      if (mid_dists_squared.z() > radius_squared) {
        // The children with the same y and z, but both x must be checked.
        index = child_index ^ (1 << 0); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      } else {
        // The children with the same y, but both x and z must be checked.
        index = child_index ^ (1 << 2); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ (1 << 0); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 0) | (1 << 2)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      }
    } else {
      if (mid_dists_squared.z() > radius_squared) {
        // The children with the same z, but both x and y must be checked.
        index = child_index ^ (1 << 1); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ (1 << 0); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 0) | (1 << 1)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      } else {
        // All children must be checked.
        index = child_index ^ (1 << 0); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ (1 << 1); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ (1 << 2); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 0) | (1 << 1)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 0) | (1 << 2)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 1) | (1 << 2)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
        index = child_index ^ ((1 << 0) | (1 << 1) | (1 << 2)); if (node->children[index]) { nodes_to_search.emplace_back(node->children[index]); }
      }
    }
  }
  
  // Descend into the child in the query's quarter, or abort.
  OctreeNode* child = node->children[child_index];
  if (!child) {
    return nullptr;
  } else {
    float dist_child_squared = (position - child->ClosestPointTo(position)).squaredNorm();
    if (dist_child_squared < std::numeric_limits<float>::epsilon()) {
      // Descend.
      return child;
    } else {
      nodes_to_search.emplace_back(child);
      return nullptr;
    }
  }
}

template <bool include_completed_surfels, bool include_free_surfels>
int CompressedOctree::FindNearestSurfelsWithinRadius(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) {
  int result_count = 0;
  float max_distance_squared = radius_squared;
  nodes_to_search_.resize(1);
  nodes_to_search_[0] = root_;
  
  // Iterate over the work stack.
  while (!nodes_to_search_.empty()) {
    OctreeNode* node = nodes_to_search_.back();
    nodes_to_search_.pop_back();
    
    // Can we skip this node?
    if (result_count == max_result_count) {
      float dist_node_squared = (position - node->ClosestPointTo(position)).squaredNorm();
      if (dist_node_squared >= max_distance_squared) {
        continue;
      }
    }
    
    // Find the smallest child node closest to the query point in this subtree,
    // while remembering other nodes if they might contain relevant results.
    while (true) {
      // If there are too many surfels in this node or there are surfels in a
      // non-leaf node, sort them down before continuing the search.
      if (node->surfels.size() > max_surfels_per_node_ ||
          (!node->IsLeaf() && node->surfels.size() > 0)) {
        node = SortSurfelsInNodeDownwardsOneStep(node);
      }
      
      node = FindNearestSurfelsWithinRadiusImpl<include_completed_surfels, include_free_surfels>(node, nodes_to_search_, position, radius_squared, max_result_count, result_distances_squared, result_indices, &result_count, &max_distance_squared, surfels_->data());
      if (!node) {
        break;
      }
    }
  }
  
  return result_count;
}

template int CompressedOctree::FindNearestSurfelsWithinRadius<false, false>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices);
template int CompressedOctree::FindNearestSurfelsWithinRadius<false, true>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices);
template int CompressedOctree::FindNearestSurfelsWithinRadius<true, false>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices);
template int CompressedOctree::FindNearestSurfelsWithinRadius<true, true>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices);

template <bool include_completed_surfels, bool include_free_surfels>
int CompressedOctree::FindNearestSurfelsWithinRadiusPassive(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) const {
  int result_count = 0;
  float max_distance_squared = radius_squared;
  nodes_to_search_.resize(1);
  nodes_to_search_[0] = root_;
  
  // Iterate over the work stack.
  while (!nodes_to_search_.empty()) {
    OctreeNode* node = nodes_to_search_.back();
    nodes_to_search_.pop_back();
    
    // Can we skip this node?
    if (result_count == max_result_count) {
      float dist_node_squared = (position - node->ClosestPointTo(position)).squaredNorm();
      if (dist_node_squared >= max_distance_squared) {
        continue;
      }
    }
    
    // Find the smallest child node closest to the query point in this subtree,
    // while remembering other nodes if they might contain relevant results.
    while (true) {
      node = FindNearestSurfelsWithinRadiusImpl<include_completed_surfels, include_free_surfels>(node, nodes_to_search_, position, radius_squared, max_result_count, result_distances_squared, result_indices, &result_count, &max_distance_squared, surfels_->data());
      if (!node) {
        break;
      }
    }
  }
  
  return result_count;
}

template int CompressedOctree::FindNearestSurfelsWithinRadiusPassive<false, false>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) const;
template int CompressedOctree::FindNearestSurfelsWithinRadiusPassive<false, true>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) const;
template int CompressedOctree::FindNearestSurfelsWithinRadiusPassive<true, false>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) const;
template int CompressedOctree::FindNearestSurfelsWithinRadiusPassive<true, true>(const Vec3f& position, float radius_squared, int max_result_count, float* result_distances_squared, u32* result_indices) const;

void CompressedOctree::FindNearestTrianglesViaSurfelsImpl(const Vec3f& position, float radius_squared, int max_surfel_count, vector<u32>* result_indices) {
  surfel_distances_squared_.resize(max_surfel_count);
  surfel_indices_.resize(max_surfel_count);
  int surfel_count = FindNearestSurfelsWithinRadius<true, false>(position, radius_squared, max_surfel_count, surfel_distances_squared_.data(), surfel_indices_.data());
  
  result_indices->clear();
  result_indices->reserve(2 * surfel_count);
  for (int i = 0; i < surfel_count; ++ i) {
    Surfel* surfel = &surfels_->at(surfel_indices_[i]);
    for (int t = 0; t < surfel->GetTriangleCount(); ++ t) {
      u32 triangle_index = surfel->GetTriangle(t);
      
      // Insert the triangle index into result_indices while avoiding duplicates.
      bool found = false;
      for (usize r = 0, size = result_indices->size(); r < size; ++ r) {
        if (result_indices->at(r) == triangle_index) {
          found = true;
          break;
        }
      }
      if (!found) {
        result_indices->push_back(triangle_index);
      }
    }
  }
}


#ifdef KEEP_TRIANGLES_IN_OCTREE
void CompressedOctree::FindNearestTrianglesIntersectingBoxImpl(const Vec3f& min, const Vec3f& max, vector<u32>* result_indices) {
  result_indices->clear();
  if (!root_) {
    return;
  }
  nodes_to_search_.resize(1);
  nodes_to_search_[0] = root_;
  
  // Iterate over the work stack.
  while (!nodes_to_search_.empty()) {
    OctreeNode* node = nodes_to_search_.back();
    nodes_to_search_.pop_back();
    
    const float kEpsilon = 1e-3f * node->half_extent;
    
    // Look for results in the current node. At the same time:
    // * Drop any triangle indices in the current node whose bbox does not
    //   intersect the node anymore, or which are invalid.
    // * Sort down any triangle indices whose bbox intersects only a child of
    //   the node.
    for (unordered_set<u32>::iterator it = node->triangles.begin(),
             end = node->triangles.end();
         it != end; ) {
      // Check whether the index is still valid.
      if (*it >= triangles_->size()) {
        it = node->triangles.erase(it);
        continue;
      }
      
      const SurfelTriangle& triangle = triangles_->at(*it);
      if (!triangle.IsValid()) {
        it = node->triangles.erase(it);
        continue;
      }
      
      // Compute the bbox of the triangle.
      Eigen::AlignedBox<float, 3> triangle_bbox(surfels_->at(triangle.index(0)).position());
      triangle_bbox.extend(surfels_->at(triangle.index(1)).position());
      triangle_bbox.extend(surfels_->at(triangle.index(2)).position());
      
      // Check whether it can be dropped because it does not intersect the node
      // anymore.
      if (triangle_bbox.min().coeff(0) > node->max.coeff(0) ||
          triangle_bbox.min().coeff(1) > node->max.coeff(1) ||
          triangle_bbox.min().coeff(2) > node->max.coeff(2) ||
          triangle_bbox.max().coeff(0) < node->min.coeff(0) ||
          triangle_bbox.max().coeff(1) < node->min.coeff(1) ||
          triangle_bbox.max().coeff(2) < node->min.coeff(2)) {
        it = node->triangles.erase(it);
        continue;
      }
      
      // Check whether it can be sorted down. Do not sort it down if its bbox
      // entirely contains the node bbox.
      if (!node->IsLeaf() &&
          (triangle_bbox.min().coeff(0) > node->min.coeff(0) ||
           triangle_bbox.min().coeff(1) > node->min.coeff(1) ||
           triangle_bbox.min().coeff(2) > node->min.coeff(2) ||
           triangle_bbox.max().coeff(0) < node->max.coeff(0) ||
           triangle_bbox.max().coeff(1) < node->max.coeff(1) ||
           triangle_bbox.max().coeff(2) < node->max.coeff(2))) {
        // Clamp the triangle bbox to the node bbox (minus an epsilon inset to
        // account for potentially slightly different decision boundaries of
        // child nodes).
        Eigen::AlignedBox<float, 3> clamped_triangle_bbox(
            Vec3f(std::max(triangle_bbox.min().coeff(0), node->min.coeff(0) + kEpsilon),
                  std::max(triangle_bbox.min().coeff(1), node->min.coeff(1) + kEpsilon),
                  std::max(triangle_bbox.min().coeff(2), node->min.coeff(2) + kEpsilon)),
            Vec3f(std::min(triangle_bbox.max().coeff(0), node->max.coeff(0) - kEpsilon),
                  std::min(triangle_bbox.max().coeff(1), node->max.coeff(1) - kEpsilon),
                  std::min(triangle_bbox.max().coeff(2), node->max.coeff(2) - kEpsilon)));
        
        // Sort it into all children which it intersects if it is entirely
        // contained by the union of the children. If not, do not sort it down.
        // Approach:
        // * Check which node quarters the triangle bbox intersects.
        // * For all intersected quarters, the child must entirely contain the
        //   part of the triangle bbox within this quarter.
        bool intersects_x[2] = {triangle_bbox.min().coeff(0) <= node->midpoint.coeff(0),
                                triangle_bbox.max().coeff(0) >= node->midpoint.coeff(0)};
        bool intersects_y[2] = {triangle_bbox.min().coeff(1) <= node->midpoint.coeff(1),
                                triangle_bbox.max().coeff(1) >= node->midpoint.coeff(1)};
        bool intersects_z[2] = {triangle_bbox.min().coeff(2) <= node->midpoint.coeff(2),
                                triangle_bbox.max().coeff(2) >= node->midpoint.coeff(2)};
        
        bool contained_in_children = true;
        int child_index = 0;
        for (int dz = 0; dz <= 1; ++ dz) {
          if (!intersects_z[dz]) {
            child_index += 4;
            continue;
          }
          for (int dy = 0; dy <= 1; ++ dy) {
            if (!intersects_y[dy]) {
              child_index += 2;
              continue;
            }
            for (int dx = 0; dx <= 1; ++ dx) {
              if (!intersects_x[dx]) {
                child_index += 1;
                continue;
              }
              
              // The triangle intersects the quarter with this child_index.
              // Check whether the child entirely contains the part of the
              // triangle bbox within this quarter.
              OctreeNode* child = node->children[child_index];
              if (!child) {
                contained_in_children = false;
                dy = 2;  // break 2
                dz = 2;  // break 3
                break;
              }
              
              // NOTE: This always checks for (float) equality with the midpoint,
              //       even in the 3 directions where it does not play a role.
              //       This is not harmful but unnecessary.
              if ((child->min.coeff(0) <= clamped_triangle_bbox.min().coeff(0) || fabsf(child->min.coeff(0) - node->midpoint.coeff(0)) <= kEpsilon) &&
                  (child->min.coeff(1) <= clamped_triangle_bbox.min().coeff(1) || fabsf(child->min.coeff(1) - node->midpoint.coeff(1)) <= kEpsilon) &&
                  (child->min.coeff(2) <= clamped_triangle_bbox.min().coeff(2) || fabsf(child->min.coeff(2) - node->midpoint.coeff(2)) <= kEpsilon) &&
                  (child->max.coeff(0) >= clamped_triangle_bbox.max().coeff(0) || fabsf(child->max.coeff(0) - node->midpoint.coeff(0)) <= kEpsilon) &&
                  (child->max.coeff(1) >= clamped_triangle_bbox.max().coeff(1) || fabsf(child->max.coeff(1) - node->midpoint.coeff(1)) <= kEpsilon) &&
                  (child->max.coeff(2) >= clamped_triangle_bbox.max().coeff(2) || fabsf(child->max.coeff(2) - node->midpoint.coeff(2)) <= kEpsilon)) {
                // Ok, triangle (part) contained in child.
              } else {
                contained_in_children = false;
                dy = 2;  // break 2
                dz = 2;  // break 3
                break;
              }
              
              child_index += 1;
            }
          }
        }
        
        if (contained_in_children) {
          // Sort the triangle down.
          child_index = 0;
          for (int dz = 0; dz <= 1; ++ dz) {
            if (!intersects_z[dz]) {
              child_index += 4;
              continue;
            }
            for (int dy = 0; dy <= 1; ++ dy) {
              if (!intersects_y[dy]) {
                child_index += 2;
                continue;
              }
              for (int dx = 0; dx <= 1; ++ dx) {
                if (!intersects_x[dx]) {
                  child_index += 1;
                  continue;
                }
                
                OctreeNode* child = node->children[child_index];
                child->AddTriangle(*it);
                
                child_index += 1;
              }
            }
          }
          
          it = node->triangles.erase(it);
          continue;
        }  // end if contained_in_children
      }  // end if it can be potentially sorted down
      
      // Test the triangle for intersection with the query box.
      if (!(min.coeff(0) > triangle_bbox.max().coeff(0) ||
            min.coeff(1) > triangle_bbox.max().coeff(1) ||
            min.coeff(2) > triangle_bbox.max().coeff(2) ||
            max.coeff(0) < triangle_bbox.min().coeff(0) ||
            max.coeff(1) < triangle_bbox.min().coeff(1) ||
            max.coeff(2) < triangle_bbox.min().coeff(2))) {
        // Make sure that each triangle is only returned once.
        // NOTE: Depending on how many results are returned usually, it might be
        //       faster to use a set or to manually do insertion sort into the 
        //       vector while keeping it ordered.
        //       Currently this assumes that usually only few triangles are
        //       returned.
        bool found = false;
        for (u32 index : *result_indices) {
          if (index == *it) {
            found = true;
            break;
          }
        }
        if (!found) {
          result_indices->push_back(*it);
        }
      }
      
      ++ it;
    }
    
    // DEBUG:
//     if (node->triangles.size() > 200) {
//       LOG(WARNING) << "Many triangles in node: " << node->triangles.size();
//       LOG(WARNING) << "IsLeaf: " << node->IsLeaf();
//       for (int i = 0; i < 8; ++ i) {
//         LOG(WARNING) << "Child " << i << ": " << node->children[i];
//       }
//     }
    
    // Descend into the child nodes.
    if (!node->IsLeaf()) {
      for (int i = 0; i < 8; ++ i) {
        OctreeNode* child = node->children[i];
        if (child &&
            !(min.coeff(0) > child->max.coeff(0) ||
              min.coeff(1) > child->max.coeff(1) ||
              min.coeff(2) > child->max.coeff(2) ||
              max.coeff(0) < child->min.coeff(0) ||
              max.coeff(1) < child->min.coeff(1) ||
              max.coeff(2) < child->min.coeff(2))) {
          nodes_to_search_.push_back(child);
        }
      }
    }
  }
}
#endif

usize CompressedOctree::CountSurfelsSlow() {
  if (root_) {
    return root_->CountSurfelsRecursive();
  } else {
    return 0;
  }
}

bool CompressedOctree::FindSurfelAnywhereSlow(u32 surfel_index, const Surfel& surfel, OctreeNode* start_node, OctreeNode** node, usize* index) const {
  if (!start_node) {
    return false;
  }
  
  // Check the current node.
  for (usize i = 0; i < start_node->surfels.size(); ++ i) {
    if (start_node->surfels[i] == surfel_index) {
      *node = start_node;
      *index = i;
      return true;
    }
  }
  
  // Check the children.
  for (int i = 0; i < 8; ++ i) {
    if (FindSurfelAnywhereSlow(surfel_index, surfel, start_node->children[i], node, index)) {
      return true;
    }
  }
  
  return false;
}

void CompressedOctree::SortSurfelDownwards(OctreeNode* node, u32 surfel_index) {
  Surfel* surfel = &surfels_->at(surfel_index);
  
  int child_index = 0;
  while (true) {
    child_index = node->ComputeChildIndex(surfel->position());
    if (node->children[child_index] == nullptr) {
      // Create a new leaf node for the surfel.
      OctreeNode* child = new OctreeNode(node->GetQuarterMidpoint(child_index),
                                         0.5f * node->half_extent);
      node->AddChild(child, child_index);
      child->AddSurfel(surfel_index, surfel);
      return;
    } else if (node->children[child_index]->Contains(surfel->position())) {
      // Descend.
      node = node->children[child_index];
      
      // Not checking for the maximum surfel count per leaf here.
      if (node->IsLeaf()) {
        // if (node->surfels.size() == max_surfels_per_node_) {
        //   LOG(WARNING) << "SortSurfelDownwards() is violating the maximum surfel count.";
        // }
        ++ numerical_issue_counter_;
        node->AddSurfel(surfel_index, surfel);
        return;
      }
    } else {
      // There is a child for this quarter, but it does not contain the new
      // surfel. In principle one should insert a new level in-between now, but
      // this function is lazy and just adds the surfel to the last node.
      ++ numerical_issue_counter_;
      // LOG(WARNING) << "CompressedOctree::SortSurfelDownwards(): Should create an intermediate level, but lazily adding the surfel to the top node instead.";
      node->AddSurfel(surfel_index, surfel);
      return;
    }
  }
}

OctreeNode* CompressedOctree::SortSurfelsInNodeDownwardsOneStep(OctreeNode* node) {
  Eigen::AlignedBox<float, 3> bbox;
  
  // Compute the quarter (child) index for each of the surfels in the node. If
  // they fall into an existing child node, directly put them into the child.
  // Otherwise, compute their bounding box.
  for (int i = node->surfels.size() - 1; i >= 0; -- i) {
    u32* surfel_index = &node->surfels[i];
    Surfel* surfel = &surfels_->at(*surfel_index);
    
    int child_index = node->ComputeChildIndex(surfel->position());
    OctreeNode* child = node->children[child_index];
    if (child && child->Contains(surfel->position())) {
      // Simply move the surfel into the child.
      child->AddSurfel(*surfel_index, surfel);
      node->EraseSurfel(i, surfels_);
    } else {
      // The surfel does not lie within the child node (if any). Add it to the
      // bounding box of new surfels for that quarter.
      bbox.extend(surfel->position());
    }
  }
  usize size = node->surfels.size();
  
  usize remaining_surfel_count = node->surfels.size();
  if (remaining_surfel_count == 0) {
    node->EraseAllSurfels();
    if (node->child_count == 1) {
      return RemoveSingleChildNode(node);
    } else {
      return node;
    }
  }
  
  // Create new nodes for surfels that did not fall into existing children.
  // The goal here for avoiding overly many steps is that at least two surfels
  // should fall into different nodes as a result.
  
  // Compute the level having the largest cells such that the surfels
  // can definitely be separated (from the x, y, z distances of the
  // surfels: choose the cell size to be at most the minimum distance).
  float dist_x = bbox.max().x() - bbox.min().x();
  float dist_y = bbox.max().y() - bbox.min().y();
  float dist_z = bbox.max().z() - bbox.min().z();
  float dist_max;
  if (dist_x > dist_y) {
    if (dist_x > dist_z) {
      dist_max = dist_x;
    } else {
      dist_max = dist_z;
    }
  } else {
    if (dist_y > dist_z) {
      dist_max = dist_y;
    } else {
      dist_max = dist_z;
    }
  }
  
  if (dist_max == 0) {
    // This is a single point only (that is maybe duplicated).
    // * If this is a leaf node, we cannot do anything since it is not possible
    //   to separate the point(s) further.
    if (node->IsLeaf()) {
      CHECK_EQ(remaining_surfel_count, node->surfels.size());
      return node;
    }
    
    // * If this node does not have a child yet in the points' quarter but it
    //   has children in other quarters, create a quarter leaf and put the
    //   point(s) inside to have them separated from the other children.
    int child_index = node->ComputeChildIndex(bbox.min());
    if (!node->children[child_index]) {
      OctreeNode* child = new OctreeNode(node->GetQuarterMidpoint(child_index),
                                         0.5f * node->half_extent);
      node->AddChild(child, child_index);
      
      child->surfels.reserve(remaining_surfel_count);
      for (usize i = 0; i < size; ++ i) {
        child->AddSurfel(node->surfels[i], &surfels_->at(node->surfels[i]));
      }
      node->EraseSurfels(0, size - 1);
      return node;
    }
    
    // * If there already is a child in the same quarter, use the same strategy
    //   as for active point insertion in this case: add an intermediate node.
    OctreeNode* new_node;
    if (!InsertIntermediateLevel(bbox.min(), node, child_index, node->children[child_index], &new_node)) {
      // Error. Give up.
      ++ numerical_issue_counter_;
      // LOG(WARNING) << "Probable numerical issue, tried to insert intermediate node but cannot.";
      return node;
    }
    
    // node->CheckContains(new_node, 0.001f);
    
    // Create a new quarter leaf in the new intermediate node and insert all
    // surfels there.
    int new_node_child_index = new_node->ComputeChildIndex(bbox.min());
    OctreeNode* child = new OctreeNode(new_node->GetQuarterMidpoint(new_node_child_index),
                                       0.5f * new_node->half_extent);
    // CHECK(new_node->children[new_node_child_index] == nullptr);
    new_node->AddChild(child, new_node_child_index);
    
    child->surfels.reserve(remaining_surfel_count);
    for (usize i = 0; i < size; ++ i) {
      child->AddSurfel(node->surfels[i], &surfels_->at(node->surfels[i]));
    }
    node->EraseSurfels(0, size - 1);  // Keep any surfels which may have been added to the back.
    return node;
  }
  
  float node_extent = 2 * node->half_extent;
  
  // Compute floor(-1 * std::log2f(dist_max / node_extent)) in a fast way.
  // Slow code:
  // int level = -1 * std::log2f(dist_max / node_extent);  // Round down in cast to int.
  
  // Values of rounded_factor:
  // For dist_max in ]0.5, 1[ * node_extent: 1
  // For dist_max in ]0.25, 0.5] * node_extent: 2 or 3
  // For dist_max in ]0.125, 0.25] * node_extent: 4 or 5 or 6 or 7, ...
  unsigned int rounded_factor = node_extent / dist_max;  // Round down in cast to int.
  if (rounded_factor == 0) {
    rounded_factor = 1;
  }
  // Count leading zeros in rounded_factor and invert the result to get
  // the number of "occupied" bits. Subtract 1 to get log2(rounded_factor).
  // Note that __builtin_clz() is undefined if its argument is 0.
  int level = (8 * sizeof(unsigned int)) - __builtin_clz(rounded_factor) - 1;
  
  // If level > 0, a new node in a quarter must be created to separate the
  // points. If level == 0, the base node itself separates the points using
  // its quarters.
  OctreeNode* current_node = node;
  if (level > 0) {
    float min_cell_quarter_extent = node_extent * 1.0f / (1 << (level + 1));
    int level_difference_plus_one =
        GetMinLevelContaining(bbox.min(), bbox.max(), node->min, min_cell_quarter_extent);
    
    if (level_difference_plus_one == 0) {
      // Should not happen. Give up.
      ++ numerical_issue_counter_;
      // LOG(WARNING) << "Encountered a case which should not happen.";
      return node;
    }
    
    level -= level_difference_plus_one - 1;
    
    if (level > 0) {
      // Create the new node on the level that was computed.
      int current_node_child_index;
      current_node =
          node->CreateChild(level, 0.5f * (bbox.min() + bbox.max()),
                            &current_node_child_index, /*add_as_child*/ false);
      
      // Robustness check: Due to numerical issues it can happen that this tries
      // to create a child in a place where there is one already. Catch this
      // case.
      OctreeNode* existing_child = node->children[current_node_child_index];
      float difference_factor =
          existing_child ? (existing_child->half_extent / current_node->half_extent) : 0;  // Should be 0.5 or less if the situation is valid.
      if (difference_factor > 0.75f && existing_child->Contains(current_node->midpoint)) {
        // Emergency handling: use the existing child instead of creating a new
        // one.
        delete current_node;
        current_node = existing_child;
      } else {
//         node->CheckContains(current_node, 0.001f);
        
        // Connect to the existing parent and child, if any.
        if (existing_child) {
          if (current_node->Contains(existing_child->midpoint)) {
            // Add the existing child as a child of the new node. Add the new node
            // as a child of the containing node:
            // node --> current_node --> existing_child
            current_node->children[current_node->ComputeChildIndex(existing_child->midpoint)] = existing_child;
            ++ current_node->child_count;
            current_node->CheckChildCount();
            existing_child->parent = current_node;
            
            node->children[current_node_child_index] = current_node;
            current_node->parent = node;
            
//             current_node->CheckContains(existing_child, 0.001f);
          } else {
            // Add an intermediate node as a child of the containing node, having
            // existing_child and current_node as children:
            // node --> intermediate_node --> existing_child
            //                            \-> current_node
            float smaller_extent;
            int smaller_level;
            if (current_node->half_extent < existing_child->half_extent) {
              smaller_extent = current_node->half_extent;
              smaller_level = level;
            } else {
              smaller_extent = existing_child->half_extent;
              int int_factor = (existing_child->half_extent / current_node->half_extent) + 0.5f;  // Round.
              smaller_level = level + (8 * sizeof(unsigned int)) - __builtin_clz(int_factor) - 1;
            }
            
            int level_difference_plus_one =
                GetMinLevelContaining(existing_child->midpoint,
                                      current_node->midpoint,
                                      node->min,
                                      smaller_extent);
            int intermediate_level = smaller_level - (level_difference_plus_one - 1);
            
            if (level_difference_plus_one <= 0 || intermediate_level <= 0) {
              // Error. Give up.
              ++ numerical_issue_counter_;
//               LOG(WARNING) << "Encountered a case which should not happen: level_difference_plus_one: " << level_difference_plus_one << ", intermediate_level: " << intermediate_level;
              delete current_node;
              return node;
            }
            
            int intermediate_node_child_index;
            OctreeNode* intermediate_node =
                node->CreateChild(intermediate_level,
                                  current_node->midpoint,
                                  &intermediate_node_child_index,
                                  /*add_as_child*/ false);
            
            // CHECK(intermediate_node->Contains(current_node->midpoint));
            // CHECK(intermediate_node->Contains(existing_child->midpoint));
            
            // Add the existing and new child as children of the intermediate node.
            int child_index_1 = intermediate_node->ComputeChildIndex(current_node->midpoint);
            int child_index_2 = intermediate_node->ComputeChildIndex(existing_child->midpoint);
            if (child_index_1 == child_index_2) {
              // Error. Give up.
              ++ numerical_issue_counter_;
              // LOG(ERROR) << "Encountered a case which should not happen.";
              delete current_node;
              delete intermediate_node;
              return node;
            }
            intermediate_node->AddChild(current_node, child_index_1);
            intermediate_node->AddChild(existing_child, child_index_2);
            
            // Set the intermediate node as child of the containing node.
            node->children[intermediate_node_child_index] = intermediate_node;
            intermediate_node->parent = node;
            
//             intermediate_node->CheckContains(current_node, 0.001f);
//             intermediate_node->CheckContains(existing_child, 0.001f);
          }
        } else {  // if (!existing_child) {
          // node --> current_node
          node->AddChild(current_node, current_node_child_index);
        }
      }
    }
  }
  
  // Sort the remaining surfels into current_node, creating new quarter leaves
  // if necessary that cover the whole quarter and include a potentially
  // existing child.
  bool created_quarter_leaf[8] = {false, false, false, false,
                                  false, false, false, false};
  for (usize i = 0; i < size; ++ i) {
    Surfel* surfel = &surfels_->at(node->surfels[i]);
    
//     // DEBUG
//     float distance_to_node = (surfel.position() - current_node->ClosestPointTo(surfel.position())).norm();
//     if (distance_to_node > 0.01f) {
//       LOG(ERROR) << "Inserting a surfel into a node which does not contain it! Distance: " << distance_to_node << ", bbox size: " << bbox.sizes().transpose();
//     }
    
    int child_index = current_node->ComputeChildIndex(surfel->position());
    if (!created_quarter_leaf[child_index]) {
      OctreeNode* existing_child = current_node->children[child_index];
      float child_half_extent = 0.5f * current_node->half_extent;
      
      // Robustness check: Due to numerical issues it can happen that this tries
      // to create a child in a place where there is one already. Catch this
      // case.
      float difference_factor =
          existing_child ? (existing_child->half_extent / child_half_extent) : 0;  // Should be 0.5 or less if the situation is valid.
      if (difference_factor > 0.75f) {
        // Emergency handling: use the existing child instead of creating a new
        // one.
      } else {
        OctreeNode* child = new OctreeNode(current_node->GetQuarterMidpoint(child_index),
                                          child_half_extent);
        
        if (existing_child) {
          // current_node --> child --> existing_child
          current_node->children[child_index] = child;
          child->parent = current_node;
          
          child->children[child->ComputeChildIndex(existing_child->midpoint)] = existing_child;
          ++ child->child_count;
          child->CheckChildCount();
          existing_child->parent = child;
          
//           current_node->CheckContains(child, 0.001f);
//           child->CheckContains(existing_child, 0.001f);
        } else {
          // current_node --> child
          current_node->AddChild(child, child_index);
          
//           current_node->CheckContains(child, 0.001f);
        }
      }
      created_quarter_leaf[child_index] = true;
    }
    
    // Add the surfel to the quarter leaf.
    current_node->children[child_index]->AddSurfel(node->surfels[i], surfel);
  }
  
  // Check whether the current node could be deleted.
  if (node->child_count == 1 &&
      node->surfels.size() == remaining_surfel_count) {
    return RemoveSingleChildNode(node);
  } else {
    node->EraseSurfels(0, size - 1);
    return node;
  }
}

void CompressedOctree::CreateRootForSurfel(u32 surfel_index, Surfel* surfel) {
  // The initial extent is arbitrary, it will be adapted once more surfels are added.
  root_ = new OctreeNode(surfel->position(), 1.0f);
  root_->parent = nullptr;
  root_->AddSurfel(surfel_index, surfel);
}

OctreeNode* CompressedOctree::RemoveSingleChildNode(OctreeNode* node) {
  // Find the single child.
  OctreeNode** single_child = &node->children[0];
  while (*single_child == nullptr) {
    ++ single_child;
  }
  CHECK_LE(single_child, &node->children[7]);
  
  // Find the child pointer in the parent.
  OctreeNode** child_pointer;
  OctreeNode* triangle_dest;
  if (node->parent == nullptr) {
    triangle_dest = *single_child;
    child_pointer = &root_;
  } else {
    triangle_dest = node->parent;
    child_pointer = &triangle_dest->children[0];
    while (*child_pointer != node) {
      ++ child_pointer;
    }
  }
  
//   // DEBUG
//   CHECK_EQ(node->child_count, 1);
//   CHECK_EQ((*single_child)->parent, node);
//   CHECK_EQ(node->surfel_count, 0u);
  
  #ifdef KEEP_TRIANGLES_IN_OCTREE
  // Move all triangles (if any) to the triangle_dest node.
    for (unordered_set<u32>::iterator it = node->triangles.begin(),
            end = node->triangles.end();
        it != end; ++ it) {
      triangle_dest->AddTriangle(*it);
    }
  #endif
  
  // Connect the parent to the child directly and delete the node
  // which was in-between.
  *child_pointer = *single_child;
  (*single_child)->parent = node->parent;
  OctreeNode* single_child_ptr = *single_child;  // Copy pointer before deleting the object containing it.
  delete node;
  
  return single_child_ptr;
}

void CompressedOctree::RemoveEmptyLeaf(OctreeNode* node) {
//   // DEBUG
//   CHECK_EQ(node->surfel_count, 0u);
//   CHECK_EQ(node->child_count, 0u);
  
  if (node->parent == nullptr) {
    // The node is the root node.
    delete node;
    root_ = nullptr;
    return;
  }
  
  OctreeNode* parent = node->parent;
  
  #ifdef KEEP_TRIANGLES_IN_OCTREE
    // Move all triangles (if any) to the parent node.
    for (unordered_set<u32>::iterator it = node->triangles.begin(),
            end = node->triangles.end();
        it != end; ++ it) {
      parent->AddTriangle(*it);
    }
  #endif
  
  // Find the child pointer in the parent and set it to nullptr.
  OctreeNode** child_pointer;
  child_pointer = &parent->children[0];
  while (*child_pointer != node) {
    ++ child_pointer;
  }
  *child_pointer = nullptr;
  -- parent->child_count;
  parent->CheckChildCount();
  
  delete node;
  
  // If the parent became a single-child node, remove it.
  if (parent->child_count == 1 && parent->IsEmpty()) {
    RemoveSingleChildNode(parent);
  }
}

void CompressedOctree::DeleteNode(OctreeNode* node) {
  if (node->child_count > 0) {
    for (int i = 0; i < 8; ++ i) {
      if (node->children[i]) {
        DeleteNode(node->children[i]);
      }
    }
  }
  delete node;
}

}
