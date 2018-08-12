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

#include "dart/optimizer/pagmo/PagmoSolver.hpp"

#include <memory>
#include <Eigen/Dense>
#include "dart/common/Console.hpp"
#include "dart/common/StlHelpers.hpp"
#include "dart/math/Constants.hpp"
#include "dart/optimizer/Function.hpp"
#include "dart/optimizer/Problem.hpp"
#include "dart/optimizer/pagmo/PagmoProblemAdaptor.hpp"

namespace dart {
namespace optimizer {

//==============================================================================
PagmoSolver::UniqueProperties::UniqueProperties(
    std::size_t numThreads,
    std::size_t maxAttempts,
    std::size_t perturbationStep,
    double maxPerturbationFactor,
    double maxRandomizationStep,
    double defaultConstraintWeight,
    Eigen::VectorXd eqConstraintWeights,
    Eigen::VectorXd ineqConstraintWeights)
  : mNumThreads(numThreads),
    mMaxAttempts(maxAttempts),
    mPerturbationStep(perturbationStep),
    mMaxPerturbationFactor(maxPerturbationFactor),
    mMaxRandomizationStep(maxRandomizationStep),
    mDefaultConstraintWeight(defaultConstraintWeight),
    mEqConstraintWeights(eqConstraintWeights),
    mIneqConstraintWeights(ineqConstraintWeights)
{
  // Do nothing
}

//==============================================================================
PagmoSolver::Properties::Properties(
    const Solver::Properties& solverProperties,
    const PagmoSolver::UniqueProperties& uniqueProperties)
  : Solver::Properties(solverProperties), UniqueProperties(uniqueProperties)
{
  // Do nothing
}

//==============================================================================
PagmoSolver::PagmoSolver(const Properties& properties)
  : Solver(properties), mPagmoSolverP(properties), mRD(), mMT(mRD())
{
  // Do nothing
}

//==============================================================================
PagmoSolver::PagmoSolver(std::shared_ptr<Problem> problem)
  : Solver(std::move(problem)), mRD(), mMT(mRD())
{
  // Do nothing
}

//==============================================================================
PagmoSolver::~PagmoSolver()
{
  // Do nothing
}

//==============================================================================
static std::vector<double> convertToStd(const Eigen::VectorXd& v)
{
  return std::vector<double>(v.data(), v.data() + v.size());
}

//==============================================================================
static Eigen::VectorXd convertToEigen(const std::vector<double>& v)
{
  Eigen::VectorXd result(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    result[i] = v[i];

  return result;
}

//==============================================================================
std::pair<double, Eigen::VectorXd> computeBestSolution(
    const pagmo::archipelago& archi)
{
  double bestF = math::constantsd::inf();
  Eigen::VectorXd bestX;

  for (const auto& island : archi)
  {
    assert(island.get_population().champion_f().size() == 1u);
    const auto champF = island.get_population().champion_f()[0];
    if (champF < bestF)
    {
      bestF = champF;
      bestX = Eigen::VectorXd::Map(
          island.get_population().champion_x().data(),
          island.get_population().champion_x().size());
    }
  }

  return std::make_pair(bestF, bestX);
}

//==============================================================================
pagmo::algorithm createNloptCobyla(const PagmoSolver::Properties& properties)
{
  pagmo::algorithm alg(pagmo::nlopt("cobyla"));

  return alg;
}

//==============================================================================
pagmo::algorithm createMoead(const PagmoSolver::Properties& properties)
{
  pagmo::algorithm alg(pagmo::moead(properties.mNumMaxIterations));

  return alg;
}

//==============================================================================
pagmo::algorithm createNsga2(const PagmoSolver::Properties& properties)
{
  pagmo::algorithm alg(pagmo::nsga2(properties.mNumMaxIterations));

  return alg;
}

//==============================================================================
pagmo::algorithm createPagmoAlgorithm(const PagmoSolver::Properties& properties)
{
  switch (properties.mAlgorithm)
  {
    case PagmoSolver::Algorithm::Local_nlopt_COBYLA:
    {
      return createNloptCobyla(properties);
    }
    case PagmoSolver::Algorithm::Global_MOEAD:
    {
      return createMoead(properties);
    }
    case PagmoSolver::Algorithm::Global_NSGA2:
    {
      return createNsga2(properties);
    }
    default:
    {
      // TODO(JS): Emit warning
      return pagmo::algorithm(pagmo::null_algorithm());
    }
  }
}

//==============================================================================
std::vector<pagmo::population> PagmoSolver::generatePopulations(
    std::size_t index, std::size_t numRequestedPopulations)
{
  std::shared_ptr<Problem> problem = mProperties.mProblem;
  std::vector<pagmo::population> populations(numRequestedPopulations);
  std::size_t numGeneratedPopulations = 0u;

  if (index == 0)
  {
    populations[0].push_back(convertToStd(problem->getInitialGuess()));
    numGeneratedPopulations++;
  }

  const std::vector<Eigen::VectorXd>& seeds = problem->getSeeds();
  while (numGeneratedPopulations < numRequestedPopulations)
  {
    assert(numGeneratedPopulations < populations.size());

    if (1u + seeds.size() < index + numGeneratedPopulations)
    {
      assert(index + numGeneratedPopulations >= 1u);
      populations[numGeneratedPopulations].push_back(
          convertToStd(seeds[index + numGeneratedPopulations - 1u]));
    }
    else
    {
      Eigen::VectorXd x;
      randomizeConfiguration(x);
      populations[numGeneratedPopulations].push_back(convertToStd(x));
    }

    numGeneratedPopulations++;
  }
  assert(populations.size() == numRequestedPopulations);

  return populations;
}

//==============================================================================
bool PagmoSolver::solve()
{
  if (!mProperties.mProblem)
    return true;

  pagmo::problem prob(PagmoProblemAdaptor(mProperties.mProblem));
  pagmo::algorithm algo = createPagmoAlgorithm(mProperties);

  std::size_t attemptCount = 0u;
  while (attemptCount <= mPagmoSolverP.mMaxAttempts)
  {
    auto pops = generatePopulations(
        mPagmoSolverP.mNumThreads * attemptCount, mPagmoSolverP.mNumThreads);

    pagmo::archipelago archi(mPagmoSolverP.mNumThreads, algo, prob, 0);
    for (auto i = 0u; i < archi.size(); ++i)
    {
      pagmo::island& isl = archi[i];
      isl.set_population(pops[i]);
    }

    archi.evolve(mProperties.mNumMaxIterations);
    archi.wait();

    const auto minimumAndX = computeBestSolution(archi);
    mProperties.mProblem->setOptimumValue(minimumAndX.first);
    mProperties.mProblem->setOptimalSolution(minimumAndX.second);

    attemptCount += mPagmoSolverP.mNumThreads;
  }

  return true;
}

//==============================================================================
Eigen::VectorXd PagmoSolver::getLastConfiguration() const
{
  return mProperties.mProblem->getOptimalSolution();
}

//==============================================================================
void PagmoSolver::randomizeConfiguration(Eigen::VectorXd& x)
{
  if (!mProperties.mProblem)
    return;

  if (x.size() < static_cast<int>(mProperties.mProblem->getDimension()))
    x.setZero(mProperties.mProblem->getDimension());

  for (int i = 0; i < x.size(); ++i)
  {
    double lower = mProperties.mProblem->getLowerBounds()[i];
    double upper = mProperties.mProblem->getUpperBounds()[i];
    double step = upper - lower;
    if (step > mPagmoSolverP.mMaxRandomizationStep)
    {
      step = 2 * mPagmoSolverP.mMaxRandomizationStep;
      lower = x[i] - step / 2.0;
    }

    x[i] = step * mDistribution(mMT) + lower;
  }
}

//==============================================================================
std::string PagmoSolver::getType() const
{
  return "PagmoSolver";
}

//==============================================================================
std::shared_ptr<Solver> PagmoSolver::clone() const
{
  return std::make_shared<PagmoSolver>(getSolverProperties());
}

//==============================================================================
void PagmoSolver::setProperties(const Properties& properties)
{
  Solver::setProperties(properties);
  setProperties(static_cast<const UniqueProperties&>(properties));
}

//==============================================================================
void PagmoSolver::setProperties(const UniqueProperties& properties)
{
  setAlgorithm(properties.mAlgorithm);
  // TODO(JS): Add more
}

//==============================================================================
PagmoSolver::Properties PagmoSolver::getGradientDescentProperties() const
{
  return Properties(getSolverProperties(), mPagmoSolverP);
}

//==============================================================================
void PagmoSolver::copy(const PagmoSolver& other)
{
  setProperties(other.getSolverProperties());
  setAlgorithm(other.getAlgorithm());
}

//==============================================================================
PagmoSolver& PagmoSolver::operator=(const PagmoSolver& other)
{
  copy(other);
  return *this;
}

//==============================================================================
void PagmoSolver::setAlgorithm(Algorithm alg)
{
  mPagmoSolverP.mAlgorithm = alg;
}

//==============================================================================
PagmoSolver::Algorithm PagmoSolver::getAlgorithm() const
{
  return mPagmoSolverP.mAlgorithm;
}

} // namespace optimizer
} // namespace dart
