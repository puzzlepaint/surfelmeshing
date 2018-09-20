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


#include <glog/logging.h>
#include <gtest/gtest.h>

#include "libvis/eigen.h"
#include "libvis/lm_optimizer.h"
#include "libvis/sophus.h"

using namespace vis;

namespace {

struct SimpleLineFitting {
  aligned_vector<Vector2f> data_points;
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const Vector2f& state,
      Accumulator* accumulator) const {
    for (const Vector2f& data_point : data_points) {
      const float x = data_point.x();
      const float y_state = state(0) * x + state(1);
      const float y_data = data_point.y();
      
      // Residual: m * x + t - y .
      const float residual = y_state - y_data;
      if (compute_jacobians) {
        const float dr_dm = x;
        const float dr_dt = 1;
        accumulator->AddJacobian(residual, 0, Vector2f(dr_dm, dr_dt));
      }
      accumulator->AddResidual(residual);
    }
  }
};

}

// Tests (and demonstrates) simple line fitting.
TEST(LMOptimizer, SimpleLineFitting) {
  // Define the residuals.
  SimpleLineFitting problem;
  const float kM = 3;
  const float kT = 2;
  problem.data_points.push_back(Vector2f(0, kM * 0 + kT));
  problem.data_points.push_back(Vector2f(1, kM * 1 + kT));
  problem.data_points.push_back(Vector2f(2, kM * 2 + kT));
  
  // Use a state consisting of 2 floats.
  LMOptimizer<float, Vector2f, SimpleLineFitting> optimizer;
  
  // Set the initial estimate.
  optimizer.state() = Vector2f(kM - 1, kT + 1);
  
  // Verify that the analytical Jacobian at this state equals the numerical
  // Jacobian (with default step size and precision threshold for all state
  // components).
  EXPECT_TRUE(optimizer.VerifyAnalyticalJacobian(1.f, numeric_limits<float>::epsilon(), problem));
  
  // Run the optimization.
  optimizer.Optimize(/*max_iteration_count*/ 100, problem,
                     /*print_progress*/ false);
  
  // Verify that the correct result is returned.
  EXPECT_FLOAT_EQ(kM, optimizer.state()(0));
  EXPECT_FLOAT_EQ(kT, optimizer.state()(1));
}

namespace {

struct SE3fState {
  // Initializes the transformation to identity.
  SE3fState() {}
  
  SE3fState(const SE3fState& other)
      : dest_TR_src(other.dest_TR_src) {}
  
  int variable_count() const {
    return SE3f::DoF;
  }
  
  template <typename Derived>
  void operator-=(const MatrixBase<Derived>& delta) {
    // Using minus delta here since we are subtracting. Using
    // left-multiplication of the delta must be consistent with the way the
    // Jacobian is computed.
    dest_TR_src = SE3f::exp(-delta) * dest_TR_src;
  }
  
  SE3f dest_TR_src;
};

struct MatchedPointsSE3Optimization {
  aligned_vector<Vec3f> src_points;
  aligned_vector<Vec3f> dest_points;
  
  template<bool compute_jacobians, class Accumulator>
  inline void Compute(
      const SE3fState& state,
      Accumulator* accumulator) const {
    for (usize i = 0, size = src_points.size(); i < size; ++ i) {
      const Vec3f& src = src_points[i];
      const Vec3f& dest_data = dest_points[i];
      
      // Transformed source point.
      const Vec3f dest_state = state.dest_TR_src * src;
      
      // Residuals: (T * src)[c] - dest[c] for each of the 3 vector components
      // c. Together, this is the squared norm of the vector difference.
      const Vec3f residuals = dest_state - dest_data;
      accumulator->AddResidual(residuals(0));
      accumulator->AddResidual(residuals(1));
      accumulator->AddResidual(residuals(2));
      if (compute_jacobians) {
        // Jacobian of: exp(hat(delta)) * T * src - dest , wrt. delta.
        // The derivation is in:
        // scripts/LMOptimizer SE3Optimization Test Jacobian derivation.ipynb.
        // The raw Jacobian is:
        // 1, 0, 0, 0, dest_state(2), -dest_state(1)
        // 0, 1, 0, -dest_state(2), 0, dest_state(0)
        // 0, 0, 1, dest_state(1), -dest_state(0), 0
        // The AddJacobian() calls avoid including the zeros.
        // Note that a one-element matrix constructor does not exist in Eigen,
        // thus we use (Matrix<...>() << element_value).finished().
        accumulator->AddJacobian(residuals(0), 0, (Matrix<float, 1, 1>() << 1).finished(), 4, Vec2f(dest_state(2), -dest_state(1)));
        accumulator->AddJacobian(residuals(1), 1, (Matrix<float, 1, 1>() << 1).finished(), 3, Vec3f(-dest_state(2), 0, dest_state(0)));
        accumulator->AddJacobian(residuals(2), 2, Vec3f(1, dest_state(1), -dest_state(0)));
      }
    }
  }
};

}

// Tests (and demonstrates) SE3 pose optimization with point correspondences.
// NOTE: This implementation can only rotate around the coordinate system
//       origin, therefore the choice of origin is important! If the actual
//       rotation origin is hard to simulate via rotation around the origin and
//       translation, the optimization will likely not converge.
TEST(LMOptimizer, SE3Optimization) {
  // Define the residuals.
  MatchedPointsSE3Optimization problem;
  problem.src_points.push_back(Vec3f(1, 2, 3));
  problem.src_points.push_back(Vec3f(3, 2, 1));
  problem.src_points.push_back(Vec3f(1, 1, 2));
  problem.src_points.push_back(Vec3f(4, 2, 2));
  problem.src_points.push_back(Vec3f(2, 2, 1));
  
  SE3f ground_truth_dest_TR_src =
      SE3f(Quaternionf(AngleAxisf(0.42f, Vec3f(1, 3, 2).normalized())),
           Vec3f(0.5f, 0.6f, 0.7f));
  problem.dest_points.resize(problem.src_points.size());
  for (usize i = 0; i < problem.src_points.size(); ++ i) {
    problem.dest_points[i] = ground_truth_dest_TR_src * problem.src_points[i];
  }
  
  // Use a custom state for the pose.
  LMOptimizer<float, SE3fState, MatchedPointsSE3Optimization> optimizer;
  
  // Set the initial estimate to identity.
  optimizer.state().dest_TR_src = SE3f();
  
  // Verify that the analytical Jacobian at this state equals the numerical
  // Jacobian (with default step size and precision threshold for all state
  // components).
  EXPECT_TRUE(optimizer.VerifyAnalyticalJacobian(1e-3f, 1.1e-2f, problem));
  
  // Run the optimization.
  optimizer.Optimize(/*max_iteration_count*/ 100, problem,
                     /*print_progress*/ false);
  
  // Verify that the correct result is returned.
  constexpr float kErrorTolerance = 0.0001f;
  SE3f::Tangent error = SE3f::log(optimizer.state().dest_TR_src.inverse() * ground_truth_dest_TR_src);
  EXPECT_LE(error(0), kErrorTolerance);
  EXPECT_LE(error(1), kErrorTolerance);
  EXPECT_LE(error(2), kErrorTolerance);
  EXPECT_LE(error(3), kErrorTolerance);
  EXPECT_LE(error(4), kErrorTolerance);
  EXPECT_LE(error(5), kErrorTolerance);
}
