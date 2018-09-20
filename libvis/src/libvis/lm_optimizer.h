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

#include <glog/logging.h>

#include "libvis/eigen.h"
#include "libvis/libvis.h"
#include "libvis/lm_optimizer_impl.h"

namespace vis {

// Generic class for non-linear continuous optimization with the
// Levenberg-Marquardt method.
//
// The general form of the optimization problem is to minimize a cost function
// consisting of a sum of residuals r_i:
//   C(x) = \sum_{i} p(r_i(x)) .
// Here, p(r) is either the square function p(r) = r^2 or a robust cost function
// such as Huber's function or Tukey's biweight function. The
// Levenberg-Marquardt algorithm implemented by this class works best if the
// individual residuals r_i are zero-mean and small when close to the optimum.
// In general, the residuals must be continuous and differentiable as this is a
// gradient-based optimization method. However, if individual residuals do not
// fulfill this over the whole domain, it is usually negligible since the other
// residuals still provide sufficient information. The gradient must be non-zero
// for the optimization to work, as otherwise the direction of the update cannot
// be determined. The optimization will find a saddle point or a local minimum,
// which is only guaranteed to be the global minimum if the cost function is
// convex.
// 
// The derivation of the method is as follows for squared residuals.
// TODO: Derivation of the update.
// 
// The user of the class defines the cost function. Optionally, analytical
// Jacobian computation can also be provided by the user for increased
// performance.
// 
// The template parameters must be given as follows:
// * Scalar should be float or double, determining the numerical precision used
//   for computing the cost and Jacobians.
// * State holds the optimization problem state and is typically an
//   Eigen::Vector. However, for optimization using Lie algebras of SE3 or SO3,
//   for example, one can provide a custom type which stores the rotation
//   part(s) of the state as quaternion while applying updates using the
//   exponential map on the corresponding Lie algebra element.
// * CostAndJacobianCalculator computes the cost for a given state and can
//   optionally provide analytical Jacobians for increased optimization speed.
// 
// The class passed for State must provide the following:
// 
// class State {
//  public:
//   // Default constructor.
//   State();
//   
//   // Copy constructor.
//   State(const State& other);
//   
//   // Returns the number of variables in the optimization problem. For
//   // example, for fitting a 2D line represented by the equation m * x + t,
//   // this should return 2 (as the parameters are m and t). Note that as an
//   // exception, this function can also be named rows(), which makes it
//   // possible to use an Eigen::Vector for the state.
//   int variable_count() const;
//   
//   // Subtracts a delta vector from the state. The delta is computed by
//   // LMOptimizer and its row count equals the return value of
//   // variable_count(). In the simplest case, the State class will subtract
//   // the corresponding delta vector component from each state variable,
//   // however for cases such as optimization over Lie groups, the
//   // implementation can differ.
//   template <typename Derived>
//   void operator-=(const MatrixBase<Derived>& delta);
// };
// 
// The class passed for CostAndJacobianCalculator must provide the following:
//
// class CostAndJacobianCalculator {
//  public:
//   // Computes the cost for a given state by providing the values of all
//   // residuals, and optionally the Jacobians of the residuals wrt. the
//   // variables if supported.
//   template<bool compute_jacobians, class Accumulator>
//   inline void Compute(
//       const State& state,
//       Accumulator* accumulator) const {
//     // To add a residual (r_i in the generic cost term above) to the cost,
//     // call the following. This expects the non-squared residual.
//     accumulator->AddResidual(residual);
//     
//     // If computing analytical Jacobians is supported and compute_jacobians
//     // is true, for each call to AddResidual(), a corresponding call to a
//     // variant of AddJacobian() must be made. Otherwise, these calls are not
//     // necessary and the corresponding computations should be omitted for
//     // better performance.
//     if (compute_jacobians) {
//       accumulator->AddJacobian(index, residual, jacobian);
//     }
//   }
// };
// 
// To use this class, first assign the initial state to the return value of
// state(), then call Optimize(). The result can again be retrieved from
// state().
// 
// TODO:
// - Would it be simpler to make the initial state a parameter of Optimize()
//   and return the final state? Or would this cause unnecessary state
//   duplicates?
// - Implement robust cost functions.
// - Implement using numerical derivatives for optimization. How to efficiently
//   get the derivatives of individual residuals? Can we optionally require
//   additional methods for this? Is it possible to initiate the computations
//   in a variant of the AddJacobian() call?
template<typename Scalar, class State, class CostAndJacobianCalculator>
class LMOptimizer {
 public:
  class UpdateEquationAccumulator {
   public:
    UpdateEquationAccumulator(
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b)
        : cost_(0), H_(H), b_(b) {
      if (H_) {
        H_->setZero();
      }
      if (b_) {
        b_->setZero();
      }
    }
    
    // To be called by CostAndJacobianCalculator to add a residual to the cost.
    inline void AddResidual(Scalar residual) {
      // TODO: Support robust cost functions.
      
      // Squared residuals.
      cost_ += residual * residual;
    }
    
    // To be called by CostAndJacobianCalculator to add the Jacobian corresponding
    // to one residual. The Jacobian corresponds to the entries
    // [index, index + jacobian.rows() - 1] of the state. If the Jacobian entries
    // are all non-zero, it will result in one dense block in H. This function is
    // for fixed-size Jacobians only and will not compile otherwise. If your
    // Jacobian contains zeros, consider using another variant of this function
    // to increase performance.
    template <typename Derived>
    inline void AddJacobian(Scalar residual, u32 index,
                            const MatrixBase<Derived>& jacobian) {
      // TODO: Support robust cost functions.
      constexpr Scalar weight = 1;
      const Scalar weighted_residual = residual;  // TODO: see above.
      
      H_->template block<Derived::RowsAtCompileTime, Derived::RowsAtCompileTime>(
          index, index)
              .template triangularView<Eigen::Upper>() +=
                  (weight * jacobian * jacobian.transpose());
      
      b_->template segment<Derived::RowsAtCompileTime>(index) +=
          (weighted_residual * jacobian);
    }
    
    // Variant of AddJacobian() for Jacobians consisting of two dense blocks
    // with zeros in-between. Avoids processing the zeros and can therefore
    // achieve higher performance in this case. It must hold: index0 < index1.
    template <typename Derived0, typename Derived1>
    inline void AddJacobian(Scalar residual, u32 index0,
                            const MatrixBase<Derived0>& jacobian0,
                            u32 index1, const MatrixBase<Derived1>& jacobian1) {
      // TODO: Support robust cost functions.
      constexpr Scalar weight = 1;
      const Scalar weighted_residual = residual;  // TODO: see above.
      
      // Block (0, 0) in H.
      H_->template block<Derived0::RowsAtCompileTime, Derived0::RowsAtCompileTime>(
          index0, index0)
              .template triangularView<Eigen::Upper>() +=
                  (weight * jacobian0 * jacobian0.transpose());
      
      // Block (0, 1) in H.
      H_->template block<Derived0::RowsAtCompileTime, Derived1::RowsAtCompileTime>(
          index0, index1) +=
              (weight * jacobian0 * jacobian1.transpose());
      
      // Block (1, 1) in H.
      H_->template block<Derived1::RowsAtCompileTime, Derived1::RowsAtCompileTime>(
          index1, index1)
              .template triangularView<Eigen::Upper>() +=
                  (weight * jacobian1 * jacobian1.transpose());
      
      // Block 0 in b.
      b_->template segment<Derived0::RowsAtCompileTime>(index0) +=
          (weighted_residual * jacobian0);
      
      // Block 1 in b.
      b_->template segment<Derived1::RowsAtCompileTime>(index1) +=
          (weighted_residual * jacobian1);
    }
    
    inline Scalar cost() const { return cost_; }
    
   private:
    Scalar cost_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* H_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* b_;
  };
  
  class JacobianVerificationHelper {
   public:
    JacobianVerificationHelper(int variable_count)
        : non_squared_cost_(0) {
      analytical_jacobian_.resize(variable_count, Eigen::NoChange);
      analytical_jacobian_.setZero();
    }
    
    inline void AddResidual(Scalar residual) {
      non_squared_cost_ += residual;
    }
    
    template <typename Derived>
    inline void AddJacobian(Scalar /*residual*/, u32 index,
                            const MatrixBase<Derived>& jacobian) {
      analytical_jacobian_.template segment<Derived::RowsAtCompileTime>(index) += jacobian;
    }
    
    template <typename Derived0, typename Derived1>
    inline void AddJacobian(Scalar /*residual*/, u32 index0,
                            const MatrixBase<Derived0>& jacobian0, u32 index1,
                            const MatrixBase<Derived1>& jacobian1) {
      analytical_jacobian_.template segment<Derived0::RowsAtCompileTime>(index0) += jacobian0;
      analytical_jacobian_.template segment<Derived1::RowsAtCompileTime>(index1) += jacobian1;
    }
    
    inline Scalar non_squared_cost() const { return non_squared_cost_; }
    
    inline const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& analytical_jacobian() const { return analytical_jacobian_; }
    
   private:
    Scalar non_squared_cost_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> analytical_jacobian_;
  };
  
  
  // Runs the optimization until convergence is assumed.
  // TODO: Allow to specify the strategy for initialization and update of
  //       lambda. Can also easily provide a Gauss-Newton implementation then by
  //       checking for the special case lambda = 0 and not retrying the update
  //       then.
  void Optimize(int max_iteration_count, const CostAndJacobianCalculator& cost_and_jac_calculator, bool print_progress) {
    // Determine the variable count of the optimization problem.
    const int variable_count = VariableCountGetter<State>::eval(state_);
    CHECK_GT(variable_count, 0);
    
    // Allocate H and b.
    // Matrix holding the Gauss-Newton Hessian approximation.
    // TODO: Also implement using sparse matrix storage for H.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H;
    H.resize(variable_count, variable_count);
  
    // Vector for the right hand side of the update linear equation system.
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
    b.resize(variable_count, Eigen::NoChange);
    
    // Do optimization iterations.
    Scalar lambda = 0;
    Scalar last_cost = numeric_limits<float>::quiet_NaN();
    bool applied_update = true;
    int iteration;
    for (iteration = 0; iteration < max_iteration_count; ++ iteration) {
      // Compute cost and Jacobians (which get accumulated on H and b).
      // TODO: Support numerical Jacobian. How to best find out which is supported?
      UpdateEquationAccumulator update_eq(&H, &b);
      cost_and_jac_calculator.template Compute<true>(state_, &update_eq);
      last_cost = update_eq.cost();
      
      if (print_progress) {
        if (iteration == 0) {
          LOG(INFO) << "LMOptimizer: [0] Initial cost == " << update_eq.cost();
        } else {
          LOG(INFO) << "LMOptimizer: [" << iteration << "] cost == " << update_eq.cost();
        }
      }
      
      // Initialize lambda based on the average diagonal element size in H.
      if (iteration == 0) {
        lambda = 0;
        for (int i = 0; i < variable_count; ++ i) {
          lambda += H(i, i);
        }
        lambda = static_cast<Scalar>(0.1) * lambda / variable_count;
      }
      
      applied_update = false;
      constexpr int kNumLMTries = 10;
      for (int lm_iteration = 0; lm_iteration < kNumLMTries; ++ lm_iteration) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H_plus_I;
        H_plus_I = H;
        // Add to the diagonal according to the Levenberg-Marquardt method.
        H_plus_I.diagonal().array() += lambda;
        
        // Using .ldlt() for a symmetric positive semi-definite matrix.
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> delta = H_plus_I.template selfadjointView<Eigen::Upper>().ldlt().solve(b);
        
        // Apply the update to create a temporary state.
        // Note the inversion of the delta here.
        State test_state(state_);
        test_state -= delta;
        
        // Test whether taking over the update will decrease the cost.
        UpdateEquationAccumulator test_cost(nullptr, nullptr);
        cost_and_jac_calculator.template Compute<false>(test_state, &test_cost);
        
        if (test_cost.cost() < update_eq.cost()) {
          // Take over the update.
          if (print_progress && lm_iteration > 0) {
            LOG(INFO) << "LMOptimizer:   [" << (iteration + 1) << "] update accepted";
          }
          state_ = test_state;
          lambda = 0.5f * lambda;
          applied_update = true;
          last_cost = test_cost.cost();
          break;
        } else {
          lambda = 2.f * lambda;
          if (print_progress) {
            LOG(INFO) << "LMOptimizer:   [" << (iteration + 1) << ", " << (lm_iteration + 1) << " of " << kNumLMTries
                      << "] update rejected (bad cost: " << test_cost.cost()
                      << "), new lambda: " << lambda;
          }
        }
      }
      
      if (!applied_update) {
        if (print_progress) {
          LOG(INFO) << "LMOptimizer: Cannot find an update which decreases the cost, aborting.";
        }
        iteration += 1;  // For correct display only.
        break;
      }
    }
    
    if (print_progress) {
      if (applied_update) {
        LOG(INFO) << "LMOptimizer: Maximum iteration count reached, aborting.";
      }
      LOG(INFO) << "LMOptimizer: [" << iteration << "] Final cost == " << last_cost;
    }
  }
  
  // Verifies the analytical cost Jacobian provided by CostAndJacobianCalculator
  // by comparing it to the numerically calculated value. This is done for
  // the current state. NOTE: This refers to the Jacobian of the total cost wrt.
  // the state variables. It does not check each residual's individual Jacobian.
  // TODO: Allow setting step size and precision threshold for each state
  //       component.
  bool VerifyAnalyticalJacobian(Scalar step_size, Scalar error_threshold,
                                const CostAndJacobianCalculator& cost_and_jac_calculator) {
    // Determine the variable count of the optimization problem.
    const int variable_count = VariableCountGetter<State>::eval(state_);
    CHECK_GT(variable_count, 0);
    
    // Determine cost at current state.
    JacobianVerificationHelper helper(variable_count);
    cost_and_jac_calculator.template Compute<true>(state_, &helper);
    const Scalar base_cost = helper.non_squared_cost();
    
    // NOTE: Using forward differences only for now.
    bool have_error = false;
    for (int variable_index = 0; variable_index < variable_count;
         ++ variable_index) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> delta;
      delta.resize(variable_count, Eigen::NoChange);
      delta.setZero();
      // Using minus step size since the delta will be subtracted.
      delta(variable_index) = -step_size;
      
      State test_state(state_);
      test_state -= delta;
      JacobianVerificationHelper test_helper(variable_count);
      cost_and_jac_calculator.template Compute<false>(test_state, &test_helper);
      const Scalar test_cost = test_helper.non_squared_cost();
      
      Scalar analytical_jacobian_component = helper.analytical_jacobian()(variable_index);
      Scalar numerical_jacobian_component = (test_cost - base_cost) / step_size;
      
      Scalar error = fabs(analytical_jacobian_component - numerical_jacobian_component);
      if (error > error_threshold) {
        LOG(ERROR) << "VerifyAnalyticalJacobian(): Component " << variable_index
                   << " differs: Analytical: " << analytical_jacobian_component
                   << ", numerical: " << numerical_jacobian_component
                   << " (base_cost: " << base_cost << ", test_cost: "
                   << test_cost << ")";
        have_error = true;
      }
    }
    return !have_error;
  }
  
  // Returns the current state of the optimization problem.
  inline const State& state() const { return state_; }
  
  // Returns the current state of the optimization problem.
  inline State& state() { return state_; }
  
 private:
  // The current state.
  State state_;
};

}
