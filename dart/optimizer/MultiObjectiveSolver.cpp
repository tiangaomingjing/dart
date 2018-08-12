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

#include "dart/optimizer/MultiObjectiveSolver.hpp"

#include <fstream>
#include "dart/common/Console.hpp"
#include "dart/optimizer/MultiObjectiveProblem.hpp"

namespace dart {
namespace optimizer {

//==============================================================================
MultiObjectiveSolver::Properties::Properties(
    std::shared_ptr<MultiObjectiveProblem> problem,
    double tolerance,
    std::size_t numMaxIterations,
    std::size_t iterationsPerPrint,
    std::ostream* ostream,
    bool printFinalResult,
    const std::string& resultFile,
    const std::string& logFile)
  : mProblem(std::move(problem)),
    mTolerance(tolerance),
    mNumMaxIterations(numMaxIterations),
    mIterationsPerPrint(iterationsPerPrint),
    mOutStream(ostream),
    mPrintFinalResult(printFinalResult),
    mResultFile(resultFile),
    mLogFilePath(logFile)
{
  // Do nothing
}

//==============================================================================
MultiObjectiveSolver::MultiObjectiveSolver(const Properties& properties)
  : mProperties(properties)
{
  // Do nothing
}

//==============================================================================
void MultiObjectiveSolver::setProperties(
    const MultiObjectiveSolver::Properties& properties)
{
  mProperties = properties;
}

//==============================================================================
const MultiObjectiveSolver::Properties&
MultiObjectiveSolver::getSolverProperties() const
{
  return mProperties;
}

//==============================================================================
void MultiObjectiveSolver::computeNondominance(
    const Eigen::MatrixXd& objectives,
    Eigen::Array<bool, Eigen::Dynamic, 1>& nondominance)
{
  assert(objectives.rows() == nondominance.rows());
  unsigned int numSamples = objectives.rows();

  // assume all non-dominant
  nondominance.setConstant(true);
  for (unsigned int i = 0; i < numSamples - 1; ++i)
  {
    // if it is nondominated; compare it with uncompared...
    if (nondominance(i) == true)
    {
      for (unsigned int j = i + 1; j < numSamples; ++j)
      {
        Eigen::VectorXd delta = objectives.row(i) - objectives.row(j);
        if ((delta.array() < 0.0).all())
        {
          // i dominates j
          nondominance(j) = false;
        }
        else if ((delta.array() > 0.0).all())
        {
          // j dominates i
          nondominance(i) = false;
          // all the solutions i dominate will also be dominated by j
          // so no need to continue checking i
          continue;
        }
      }
    }
  }
}

//==============================================================================
void MultiObjectiveSolver::setProblem(
    std::shared_ptr<MultiObjectiveProblem> problem)
{
  mProperties.mProblem = std::move(problem);
}

//==============================================================================
std::shared_ptr<MultiObjectiveProblem> MultiObjectiveSolver::getProblem() const
{
  return mProperties.mProblem;
}

//==============================================================================
void MultiObjectiveSolver::setNumMaxIterations(std::size_t maxIterations)
{
  mProperties.mNumMaxIterations = maxIterations;
}

//==============================================================================
std::size_t MultiObjectiveSolver::getNumMaxIterations() const
{
  return mProperties.mNumMaxIterations;
}

//==============================================================================
void MultiObjectiveSolver::setLogname(const std::string& logName)
{
  mProperties.mLogFilePath = logName;
}

//==============================================================================
const std::string& MultiObjectiveSolver::getLogName()
{
  return mProperties.mLogFilePath;
}

//==============================================================================
void MultiObjectiveSolver::saveEliteLog()
{
  if (mElitePopulation.rows() == 0)
  {
    dtmsg << "EliteLog is not saved because it's not available.\n";
    return;
  }

  const static Eigen::IOFormat CSVFormat(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

  const std::string popFilename = mProperties.mLogFilePath + "-Elitepop.csv";
  const std::string objFilename = mProperties.mLogFilePath + "-Eliteobj.csv";
  const std::string nondomFilename
      = mProperties.mLogFilePath + "-Elitenondom.csv";

  std::ofstream popFile(popFilename.c_str());
  if (popFile.is_open())
    popFile << mElitePopulation.format(CSVFormat);
  popFile.close();

  std::ofstream objFile(objFilename.c_str());
  if (objFile.is_open())
    objFile << mEliteObjectives.format(CSVFormat);
  objFile.close();

  std::ofstream nondomFile(nondomFilename.c_str());
  if (nondomFile.is_open())
    nondomFile << mEliteNotDominated.format(CSVFormat);
  nondomFile.close();
}

//==============================================================================
void MultiObjectiveSolver::setRecordHistory(bool on)
{
  mRecordPopulationHistory = on;
}

//==============================================================================
void MultiObjectiveSolver::getNondominatedPopulation(
    const Eigen::MatrixXd& population,
    const Eigen::MatrixXd& objectives,
    const Eigen::Array<bool, Eigen::Dynamic, 1>& nondominance,
    Eigen::MatrixXd& nonDominatedPopulation,
    Eigen::MatrixXd& nonDominatedObjectives)
{
  const auto numSamples = population.rows();
  const auto numObjectives = mProperties.mProblem->getObjectives().size();
  const auto dimension = mProperties.mProblem->getDimension();

  int numOfNonDominated = (nondominance == true).count();
  nonDominatedPopulation.setZero(numOfNonDominated, dimension);
  nonDominatedObjectives.setZero(numOfNonDominated, numObjectives);

  std::size_t idx = 0;
  for (auto i = 0; i < numSamples; i++)
  {
    if (nondominance[i] == true)
    {
      nonDominatedPopulation.row(idx) = population.row(i);
      nonDominatedObjectives.row(idx) = objectives.row(i);
      idx++;
    }
  }
}

//==============================================================================
void MultiObjectiveSolver::updateElitePopulation(
    const Eigen::MatrixXd& nonDominatedPopulation,
    const Eigen::MatrixXd& nonDominatedObjectives)
{
  if (mElitePopulation.rows() == 0)
  {
    mElitePopulation = nonDominatedPopulation;
    mEliteObjectives = nonDominatedObjectives;
    return;
  }

  Eigen::MatrixXd newPop(
      nonDominatedPopulation.rows() + mElitePopulation.rows(),
      mElitePopulation.cols());
  newPop << nonDominatedPopulation, mElitePopulation;
  Eigen::MatrixXd newObj(
      nonDominatedObjectives.rows() + mEliteObjectives.rows(),
      mEliteObjectives.cols());
  newObj << nonDominatedObjectives, mEliteObjectives;

  Eigen::Array<bool, Eigen::Dynamic, 1> newNondominance(newPop.rows());
  Eigen::MatrixXd tmpPopulation;
  Eigen::MatrixXd tmpObjectives;
  computeNondominance(newObj, newNondominance);
  getNondominatedPopulation(
      newPop, newObj, newNondominance, tmpPopulation, tmpObjectives);

  if (static_cast<std::size_t>(tmpPopulation.rows())
      > mProperties.mProblem->getNumSolutions())
  {
    std::vector<int> l(tmpPopulation.rows());
    for (auto i = 0; i < tmpPopulation.rows(); ++i)
    {
      l[i] = i;
    }
    std::random_shuffle(l.begin(), l.end());

    mElitePopulation.resize(
        mProperties.mProblem->getNumSolutions(),
        mProperties.mProblem->getDimension());
    mEliteObjectives.resize(
        mProperties.mProblem->getNumSolutions(),
        mProperties.mProblem->getNumObjectives());
    for (auto i = 0u; i < mProperties.mProblem->getNumSolutions(); ++i)
    {
      mElitePopulation.row(i) = tmpPopulation.row(l[i]);
      mEliteObjectives.row(i) = tmpObjectives.row(l[i]);
    }
  }
  else
  {
    mElitePopulation = tmpPopulation;
    mEliteObjectives = tmpObjectives;
  }
}

} // namespace optimizer
} // namespace dart
