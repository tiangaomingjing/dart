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

#include "dart/optimizer/MultiObjectiveProblem.hpp"

#include <algorithm>
#include <limits>

#include "dart/common/Console.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/optimizer/Function.hpp"

namespace dart {
namespace optimizer {

//==============================================================================
template <typename T>
static T getVectorObjectIfAvailable(std::size_t _idx, const std::vector<T>& vec)
{
  // TODO: Should we have an out-of-bounds assertion or throw here?
  if (_idx < vec.size())
    return vec[_idx];

  return nullptr;
}

//==============================================================================
MultiObjectiveProblem::MultiObjectiveProblem(
    std::size_t dim, std::size_t numSolutions)
  : mDimension(0), mOptimumValue(0.0)
{
  setDimension(dim);
}

//==============================================================================
void MultiObjectiveProblem::setDimension(std::size_t dim)
{
  if (dim == mDimension)
    return;

  const double inf = std::numeric_limits<double>::infinity();
  const auto dimension = static_cast<Eigen::VectorXd::Index>(dim);

  mDimension = dim;

  mInitialGuess = Eigen::VectorXd::Zero(dimension);
  mLowerBounds = Eigen::VectorXd::Constant(dimension, -inf);
  mUpperBounds = Eigen::VectorXd::Constant(dimension, inf);
  mOptimalSolution = Eigen::VectorXd::Zero(dimension);

  clearAllSeeds();
}

//==============================================================================
std::size_t MultiObjectiveProblem::getDimension() const
{
  return mDimension;
}

//==============================================================================
void MultiObjectiveProblem::setNumSolutions(std::size_t numSolutions)
{
  mNumSolutions = numSolutions;
}

//==============================================================================
std::size_t MultiObjectiveProblem::getNumSolutions() const
{
  return mNumSolutions;
}

//==============================================================================
void MultiObjectiveProblem::setInitialGuess(const Eigen::VectorXd& initGuess)
{
  assert(
      static_cast<std::size_t>(initGuess.size()) == mDimension
      && "Invalid size.");

  if (initGuess.size() != static_cast<int>(mDimension))
  {
    dterr << "[Problem::setInitialGuess] Attempting to set the initial guess "
          << "of a Problem of dimension [" << mDimension << "] to a vector of "
          << "dimension [" << initGuess.size() << "]. This initial guess "
          << "will not be used!\n";
    return;
  }

  mInitialGuess = initGuess;
}

//==============================================================================
const Eigen::VectorXd& MultiObjectiveProblem::getInitialGuess() const
{
  return mInitialGuess;
}

//==============================================================================
void MultiObjectiveProblem::addSeed(const Eigen::VectorXd& seed)
{
  if (seed.size() == static_cast<int>(mDimension))
  {
    mSeeds.push_back(seed);
  }
  else
  {
    dtwarn << "[Problem::addSeed] Attempting to add a seed of dimension ["
           << seed.size() << "] a Problem of dimension [" << mDimension
           << "]. The seed will not be added.\n";
  }
}

//==============================================================================
Eigen::VectorXd& MultiObjectiveProblem::getSeed(std::size_t index)
{
  if (index < mSeeds.size())
    return mSeeds[index];

  if (mSeeds.size() == 0)
    dtwarn << "[Problem::getSeed] Requested seed at index [" << index << "], "
           << "but there are currently no seeds. Returning the problem's "
           << "initial guess instead.\n";
  else
    dtwarn << "[Problem::getSeed] Requested seed at index [" << index << "], "
           << "but the current max index is [" << mSeeds.size() - 1 << "]. "
           << "Returning the Problem's initial guess instead.\n";

  return mInitialGuess;
}

//==============================================================================
const Eigen::VectorXd& MultiObjectiveProblem::getSeed(std::size_t index) const
{
  return const_cast<MultiObjectiveProblem*>(this)->getSeed(index);
}

//==============================================================================
std::vector<Eigen::VectorXd>& MultiObjectiveProblem::getSeeds()
{
  return mSeeds;
}

//==============================================================================
const std::vector<Eigen::VectorXd>& MultiObjectiveProblem::getSeeds() const
{
  return mSeeds;
}

//==============================================================================
void MultiObjectiveProblem::clearAllSeeds()
{
  mSeeds.clear();
}

//==============================================================================
void MultiObjectiveProblem::setLowerBounds(const Eigen::VectorXd& lb)
{
  assert(static_cast<std::size_t>(lb.size()) == mDimension && "Invalid size.");
  mLowerBounds = lb;
}

//==============================================================================
const Eigen::VectorXd& MultiObjectiveProblem::getLowerBounds() const
{
  return mLowerBounds;
}

//==============================================================================
void MultiObjectiveProblem::setUpperBounds(const Eigen::VectorXd& ub)
{
  assert(static_cast<std::size_t>(ub.size()) == mDimension && "Invalid size.");
  mUpperBounds = ub;
}

//==============================================================================
const Eigen::VectorXd& MultiObjectiveProblem::getUpperBounds() const
{
  return mUpperBounds;
}

//==============================================================================
void MultiObjectiveProblem::setObjectives(
    const std::vector<FunctionPtr>& objectives)
{
  mObjectives = objectives;
}

//==============================================================================
void MultiObjectiveProblem::addObjective(FunctionPtr obj)
{
  assert(obj && "nullptr pointer is not allowed.");
  mObjectives.emplace_back(std::move(obj));
}

//==============================================================================
std::size_t MultiObjectiveProblem::getNumObjectives() const
{
  return mObjectives.size();
}

//==============================================================================
const std::vector<FunctionPtr>& MultiObjectiveProblem::getObjectives() const
{
  return mObjectives;
}

//==============================================================================
void MultiObjectiveProblem::addEqConstraint(FunctionPtr eqConst)
{
  assert(eqConst);
  mEqConstraints.push_back(eqConst);
}

//==============================================================================
void MultiObjectiveProblem::addIneqConstraint(FunctionPtr _ineqConst)
{
  assert(_ineqConst);
  mIneqConstraints.push_back(_ineqConst);
}

//==============================================================================
std::size_t MultiObjectiveProblem::getNumEqConstraints() const
{
  return mEqConstraints.size();
}

//==============================================================================
std::size_t MultiObjectiveProblem::getNumIneqConstraints() const
{
  return mIneqConstraints.size();
}

//==============================================================================
FunctionPtr MultiObjectiveProblem::getEqConstraint(std::size_t idx) const
{
  assert(idx < mEqConstraints.size());
  return getVectorObjectIfAvailable<FunctionPtr>(idx, mEqConstraints);
}

//==============================================================================
FunctionPtr MultiObjectiveProblem::getIneqConstraint(std::size_t idx) const
{
  assert(idx < mIneqConstraints.size());
  return getVectorObjectIfAvailable<FunctionPtr>(idx, mIneqConstraints);
}

//==============================================================================
void MultiObjectiveProblem::removeEqConstraint(FunctionPtr eqConst)
{
  // TODO(JS): Need to delete?
  mEqConstraints.erase(
      std::remove(mEqConstraints.begin(), mEqConstraints.end(), eqConst),
      mEqConstraints.end());
}

//==============================================================================
void MultiObjectiveProblem::removeIneqConstraint(FunctionPtr ineqConst)
{
  // TODO(JS): Need to delete?
  mIneqConstraints.erase(
      std::remove(mIneqConstraints.begin(), mIneqConstraints.end(), ineqConst),
      mIneqConstraints.end());
}

//==============================================================================
void MultiObjectiveProblem::removeAllEqConstraints()
{
  // TODO(JS): Need to delete?
  mEqConstraints.clear();
}

//==============================================================================
void MultiObjectiveProblem::removeAllIneqConstraints()
{
  // TODO(JS): Need to delete?
  mIneqConstraints.clear();
}

//==============================================================================
void MultiObjectiveProblem::setOptimumValue(double val)
{
  mOptimumValue = val;
}

//==============================================================================
double MultiObjectiveProblem::getOptimumValue() const
{
  return mOptimumValue;
}

//==============================================================================
void MultiObjectiveProblem::setOptimalSolution(const Eigen::VectorXd& optParam)
{
  assert(
      static_cast<std::size_t>(optParam.size()) == mDimension
      && "Invalid size.");
  mOptimalSolution = optParam;
}

//==============================================================================
const Eigen::VectorXd& MultiObjectiveProblem::getOptimalSolution()
{
  return mOptimalSolution;
}

} // namespace optimizer
} // namespace dart
