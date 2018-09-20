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

#include <libvis/eigen.h>
#include <libvis/libvis.h>

namespace vis {

struct OctreeNode;

// Stores the two edges connected to a front surfel.
struct Front {
  inline Front() {}
  
  inline Front(u32 left_, u32 right_)
      : left(left_), right(right_) {}
  
  // Index of the left surfel when looking at the front from the top and there
  // is free space in the forward direction.
  u32 left;
  
  // Index of the right surfel when looking at the front from the top and there
  // is free space in the forward direction.
  u32 right;
};


// Represents a surfel on the CPU. Only contains attributes which are relevant
// for meshing.
class Surfel {
 public:
  // Generally used to mark invalid indices to surfels.
  constexpr static u32 kInvalidIndex = std::numeric_limits<u32>::max();
  // Used in octree node free lists for free-list entries (instead of surfel indices).
  constexpr static u32 kInvalidBit = 1u << (32 - 1);
  
  enum class MeshingState : u8 {
    kFree = 0,  // No triangles are incident to this surfel.
    kFront = 1,  // Some triangles are incident to this surfel.
    kCompleted = 2  // The surfel is completely surrounded by triangles.
  };
  
  template <typename DerivedA, typename DerivedB>
  inline Surfel(const MatrixBase<DerivedA>& position, float radius_squared, const MatrixBase<DerivedB>& normal, u32 last_update_stamp)
      : position_(position),
        radius_squared_(radius_squared),
        normal_(normal),
        last_update_stamp_(last_update_stamp),
        meshing_state_(MeshingState::kFree) {}
  
  inline void SetMeshingState(MeshingState state) {
    meshing_state_ = state;
  }
  
  template <typename Derived>
  inline void SetPosition(const MatrixBase<Derived>& position) {
    position_ = position;
  }
  
  template <typename Derived>
  inline void SetNormal(const MatrixBase<Derived>& normal) {
    normal_ = normal;
  }
  
  inline void SetOctreeNode(OctreeNode* node, u32 index_in_node) {
    node_ = node;
    index_in_node_ = index_in_node;
  }
  
  inline void SetOctreeNodeIndex(u32 index_in_node) {
    index_in_node_ = index_in_node;
  }
  
  inline void AddTriangle(u32 triangle_index) {
    triangles_.emplace_back(triangle_index);
  }
  
  inline void RemoveTriangle(u32 triangle_index) {
    for (usize i = 0, size = triangles_.size(); i < size; ++ i) {
      if (triangles_[i] == triangle_index) {
        triangles_[i] = triangles_[size - 1];
        triangles_.pop_back();
        return;
      }
    }
    LOG(FATAL) << "RemoveTriangle() in a surfel did not find the triangle.";
  }
  
  inline void RemoveAllTriangles() {
    triangles_.clear();
  }
  
  inline int GetTriangleCount() const {
    return triangles_.size();
  }
  
  inline u32 GetTriangle(int index) const {
    return triangles_[index];
  }
  
  inline void SetLastUpdateStamp(u32 last_update_stamp) {
    last_update_stamp_ = last_update_stamp;
  }
  
  inline void SetRadiusSquared(float radius_squared) {
    radius_squared_ = radius_squared;
  }
  
  inline void SetCanBeRemeshed(bool can_be_remeshed) {
    flags_ = (flags_ & ~(1<<1)) | (can_be_remeshed << 1);
  }
  
  inline void SetCanBeReset(bool can_be_reset) {
    flags_ = (flags_ & ~1) | can_be_reset;
  }
  
  inline void SetFlags(bool can_be_remeshed, bool can_be_reset) {
    flags_ = (can_be_remeshed << 1) | can_be_reset;
  }
  
  inline const Vec3f& position() const { return position_; }
  inline float radius_squared() const { return radius_squared_; }
  inline const Vec3f& normal() const { return normal_; }
  inline OctreeNode* node() const { return node_; }
  inline vector<Front>& fronts() { return fronts_; }
  inline const vector<Front>& fronts() const { return fronts_; }
  inline u32 index_in_node() const { return index_in_node_; }
  inline u32 last_update_stamp() const { return last_update_stamp_; }
  inline MeshingState meshing_state() const { return meshing_state_; }
  inline bool can_be_remeshed() const { return flags_ & (1<<1); }
  inline bool can_be_reset() const { return flags_ & 1; }
  
 private:
  friend std::ostream& operator<<(std::ostream& os, const Surfel& s);
  
  Vec3f position_;
  float radius_squared_;
  Vec3f normal_;
  u32 index_in_node_;
  OctreeNode* node_;
  vector<u32> triangles_;
  vector<Front> fronts_;
  u32 last_update_stamp_;
  u8 flags_;
  MeshingState meshing_state_;
};

inline std::ostream& operator<<(std::ostream& os, const Surfel& s) {
  return os << "[Surfel position: " << s.position().transpose()
            << ", radius_squared: " << s.radius_squared()
            << ", normal: " << s.normal().transpose()
            << ", meshing state (free: 0, front: 1, completed: 2): " << static_cast<int>(s.meshing_state()) << "]";
}


class SurfelTriangle {
 public:
  // Value used to indicate missing adjacent triangles.
  constexpr static u32 kInvalid = std::numeric_limits<u32>::max();
  
  // Value used to indicate that this entry is not a valid triangle, but a free list entry.
  constexpr static u32 kFreeListEntryMark = std::numeric_limits<u32>::max();
  
  inline SurfelTriangle() {}
  
  inline SurfelTriangle(u32 i0, u32 i1, u32 i2)
      : indices_{i0, i1, i2} {}
  
  inline bool EqualsRotated(u32 a, u32 b, u32 c) {
    return (indices_[0] == a && indices_[1] == b && indices_[2] == c) ||
           (indices_[0] == b && indices_[1] == c && indices_[2] == a) ||
           (indices_[0] == c && indices_[1] == a && indices_[2] == b);
  }
  
  inline void DeleteFromSurfels(u32 my_triangle_index, vector<Surfel>* surfels) {
    for (int i = 0; i < 3; ++ i) {
      Surfel* surfel = &surfels->at(indices_[i]);
      surfel->RemoveTriangle(my_triangle_index);
    }
  }
  
  inline void MakeFreeListEntry(u32 next_free_entry_index) {
    // At least two indices must be the same, otherwise this would create
    // artifacts if it gets transferred to the GPU directly.
    indices_[0] = next_free_entry_index;
    indices_[1] = 0;
    indices_[2] = kFreeListEntryMark;
  }
  
  inline u32 free_list_value() const {
    return indices_[0];
  }
  
  inline bool IsValid() const { return indices_[2] != kFreeListEntryMark; }
  
  inline u32 index(int i) const { return indices_[i]; }
  
 private:
  // Indices of the surfels at the corners of this triangle. They are ordered
  // counter-clockwise when looking at the triangle from the front side.
  u32 indices_[3];
};

}
