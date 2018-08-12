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

#ifndef DART_OPTIMIZER_PAGMO_PAGMOSOLVER_HPP_
#define DART_OPTIMIZER_PAGMO_PAGMOSOLVER_HPP_

#include <random>
#include <pagmo/pagmo.hpp>
#include "dart/optimizer/Solver.hpp"
#include "dart/optimizer/pagmo/PagmoOptionList.hpp"

#define DART_PAGMO_DEFAULT_SOLVER Algorithm::Global_MOEAD

namespace dart {
namespace optimizer {

class Problem;

class PagmoSolver : public Solver
{
public:
  /// Reference: https://esa.github.io/pagmo2/docs/algorithm_list.html
  enum class Algorithm
  {
    Local_CompassSearch,
    Local_nlopt_COBYLA,
    Global_NullAlgorithm,
    Global_DifferentialEvolution,
    Global_ImprovedHarmonySearch,
    Global_SelfAdaptiveDifferentialEvolution,
    Global_MOEAD,
    Global_NSGA2,
  };

  struct UniqueProperties
  {

    std::size_t mPopulationSize;

    std::size_t mNumThreads; // or Island or NumPopulations

    /// Number of attempts to make before quitting. Each attempt will start from
    /// the next seed provided by the problem. Once there are no more seeds,
    /// random starting configurations will be used.
    ///
    /// Set this to 0 to keep trying until a solution is found (the program will
    /// need to be interrupted in order to stop if no solution is being found).
    std::size_t mMaxAttempts;

    /// The number of steps between random perturbations being applied to the
    /// configuration. Set this to 0 to never apply randomized perturbations.
    std::size_t mPerturbationStep;

    /// The random perturbation works as follows: A random point in the domain
    /// of the Problem is selected, and then a random step size between 0 and
    /// mMaxPerturbationFactor is selected. The configuration will take a step
    /// of that random step size towards the random point.
    ///
    /// A maximum value of 1.0 is recommended for mMaxPerturbationFactor. A
    /// smaller value will result in smaller randomized perturbations. A value
    /// significantly larger than 1.0 could bias the configuration towards the
    /// boundary of the Problem domain.
    double mMaxPerturbationFactor;

    /// The largest permittable change in value when randomizing a configuration
    double mMaxRandomizationStep;

    /// This is the weight that will be applied to any constraints that do not
    /// have a corresponding weight specified by mEqConstraintWeights or by
    /// mIneqConstraintWeights
    double mDefaultConstraintWeight;

    /// Vector of weights that should be applied to the equality constraints.
    /// If there are fewer components in this vector than there are equality
    /// constraints in the Problem, then the remaining equality constraints will
    /// be assigned a weight of mDefaultConstraintWeight.
    Eigen::VectorXd mEqConstraintWeights;

    /// Vector of weights that should be applied to the inequality constraints.
    /// If there are fewer components in this vector than there are inequality
    /// constraints in the Problem, then the remaining inequality constraints
    /// will be assigned a weight of mDefaultConstraintWeight.
    Eigen::VectorXd mIneqConstraintWeights;

    /// Algorithm to be used by the pagmo
    Algorithm mAlgorithm;

    PagmoOptionList mOptions;

    UniqueProperties(
        std::size_t numThreads = 1,
        std::size_t maxAttempts = 1,
        std::size_t perturbationStep = 0,
        double maxPerturbationFactor = 1.0,
        double maxRandomizationStep = 1e10,
        double defaultConstraintWeight = 1.0,
        Eigen::VectorXd eqConstraintWeights = Eigen::VectorXd(),
        Eigen::VectorXd ineqConstraintWeights = Eigen::VectorXd());
  };

  struct Properties : Solver::Properties, UniqueProperties
  {
    Properties(
        const Solver::Properties& solverProperties = Solver::Properties(),
        const UniqueProperties& descentProperties = UniqueProperties());
  };

  /// Default Constructor
  PagmoSolver(const Properties& properties = Properties());

  /// Alternative Constructor
  PagmoSolver(std::shared_ptr<Problem> problem);

  /// Destructor
  ~PagmoSolver() override;

  // Documentation inherited
  bool solve() override;

  // Documentation inherited
  std::string getType() const override;

  // Documentation inherited
  std::shared_ptr<Solver> clone() const override;

  /// Sets the Properties of this PagmoSolver
  void setProperties(const Properties& properties);

  /// Sets the Properties of this PagmoSolver
  void setProperties(const UniqueProperties& properties);

  /// Returns the Properties of this GradientDescentSolver
  Properties getGradientDescentProperties() const;

  /// Copies the Properties of another PagmoSolver
  void copy(const PagmoSolver& other);

  /// Copies the Properties of another PagmoSolver
  PagmoSolver& operator=(const PagmoSolver& other);

  /// Sets the algorithm that is to be used by the solver
  void setAlgorithm(Algorithm alg);

  /// Returns the algorithm that is to be used by the pagmo solver
  Algorithm getAlgorithm() const;

  Eigen::VectorXd getLastConfiguration() const;

protected:
  void randomizeConfiguration(Eigen::VectorXd& x);

  /// Generates
  std::vector<pagmo::population> generatePopulations(
      std::size_t index, std::size_t numRequestedPopulations);

  /// PagmoSolver properties
  UniqueProperties mPagmoSolverP;

  /// Randomization device
  std::random_device mRD;

  /// Mersenne twister method
  std::mt19937 mMT;

  /// Distribution
  std::uniform_real_distribution<double> mDistribution;
};

} // namespace optimizer
} // namespace dart

#endif // DART_OPTIMIZER_PAGMO_PAGMOSOLVER_HPP_
