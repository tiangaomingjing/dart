/*
 * Copyright (c) 2011-2018, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef DART_OPTIMIZER_MULTIOBJECTIVEPROBLEM_HPP_
#define DART_OPTIMIZER_MULTIOBJECTIVEPROBLEM_HPP_

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "dart/optimizer/Function.hpp"

namespace dart {
namespace optimizer {

class MultiObjectiveProblem
{
public:
  /// Constructor
  explicit MultiObjectiveProblem(
      std::size_t _dim = 0, std::size_t numSolutions = 100);

  /// Destructor
  virtual ~MultiObjectiveProblem() = default;

  /// \{ \name Problem Setting

  /// Sets dimension. Note: Changing the dimension will clear out the
  /// initial guess and any seeds that have been added.
  void setDimension(std::size_t dim);

  /// Returns dimension
  std::size_t getDimension() const;

  void setNumSolutions(std::size_t numSolutions);

  std::size_t getNumSolutions() const;

  /// Sets initial guess for opimization parameters
  void setInitialGuess(const Eigen::VectorXd& initGuess);

  /// Sets initial guess for opimization parameters
  const Eigen::VectorXd& getInitialGuess() const;

  /// Adds a seed for the Solver to use as a hint for the neighborhood of
  /// the solution.
  void addSeed(const Eigen::VectorXd& seed);

  /// Returns a mutable reference of the seed for the specified index. If an
  /// out-of-bounds index is provided a warning will print, and a reference to
  /// the initial guess will be returned instead.
  Eigen::VectorXd& getSeed(std::size_t index);

  /// An immutable version of getSeed(std::size_t)
  const Eigen::VectorXd& getSeed(std::size_t index) const;

  /// Returns a mutable reference to the full vector of seeds that this
  /// Problem currently contains
  std::vector<Eigen::VectorXd>& getSeeds();

  /// An immutable version of getSeeds()
  const std::vector<Eigen::VectorXd>& getSeeds() const;

  /// Clears the seeds that this Problem currently contains
  void clearAllSeeds();

  /// Sets lower bounds for optimization parameters
  void setLowerBounds(const Eigen::VectorXd& lb);

  /// Returns lower bounds for optimization parameters
  const Eigen::VectorXd& getLowerBounds() const;

  /// Sets upper bounds for optimization parameters
  void setUpperBounds(const Eigen::VectorXd& ub);

  /// Returns upper bounds for optimization parameters
  const Eigen::VectorXd& getUpperBounds() const;

  /// Sets objective functions to be minimized.
  void setObjectives(const std::vector<FunctionPtr>& objectives);

  /// Adds a minimum objective function
  void addObjective(FunctionPtr obj);

  std::size_t getNumObjectives() const;

  /// Returns objective functions
  const std::vector<FunctionPtr>& getObjectives() const;

  /// Adds equality constraint
  void addEqConstraint(FunctionPtr eqConst);

  /// Adds inequality constraint. Inequality constraints must evaluate
  /// to LESS THAN or equal to zero (within some tolerance) to be satisfied.
  void addIneqConstraint(FunctionPtr _ineqConst);

  /// Returns number of equality constraints
  std::size_t getNumEqConstraints() const;

  /// Returns number of inequality constraints
  std::size_t getNumIneqConstraints() const;

  /// Returns equality constraint
  FunctionPtr getEqConstraint(std::size_t idx) const;

  /// Returns inequality constraint
  FunctionPtr getIneqConstraint(std::size_t idx) const;

  /// Removes equality constraint
  void removeEqConstraint(FunctionPtr eqConst);

  /// Removes inequality constraint
  void removeIneqConstraint(FunctionPtr ineqConst);

  /// Removes all equality constraints
  void removeAllEqConstraints();

  /// Removes all inequality constraints
  void removeAllIneqConstraints();

  /// \}

  /// \{ \name Result

  /// Sets optimum value of the objective function. This function called by
  /// Solver.
  void setOptimumValue(double val);

  /// Returns optimum value of the objective function
  double getOptimumValue() const;

  /// Sets optimal solution. This function called by Solver.
  void setOptimalSolution(const Eigen::VectorXd& optParam);

  /// Returns optimal solution
  const Eigen::VectorXd& getOptimalSolution();

  /// \}

protected:
  /// Dimension of this problem
  std::size_t mDimension;

  /// Initial guess for optimization parameters
  Eigen::VectorXd mInitialGuess;

  /// Additional guess hints for the Solver.
  std::vector<Eigen::VectorXd> mSeeds;

  /// Lower bounds for optimization parameters
  Eigen::VectorXd mLowerBounds;

  /// Upper bounds for optimization parameters
  Eigen::VectorXd mUpperBounds;

  /// Objective function
  std::vector<FunctionPtr> mObjectives;

  /// Equality constraint functions
  std::vector<FunctionPtr> mEqConstraints;

  /// Inequality constraint functions
  std::vector<FunctionPtr> mIneqConstraints;

  /// Optimal objective value
  double mOptimumValue;

  /// Optimal solution
  Eigen::VectorXd mOptimalSolution;

  /// Number of Pareto-optimal solutions
  std::size_t mNumSolutions;
};

} // namespace optimizer
} // namespace dart

#endif // DART_OPTIMIZER_MULTIOBJECTIVEPROBLEM_HPP_
