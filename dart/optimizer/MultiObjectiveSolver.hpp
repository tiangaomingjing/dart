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

#ifndef DART_OPTIMIZER_MULTIOBJECTIVESOLVER_HPP_
#define DART_OPTIMIZER_MULTIOBJECTIVESOLVER_HPP_

#include <iostream>
#include <memory>

#include <Eigen/Dense>

namespace dart {
namespace optimizer {

class MultiObjectiveProblem;

/// Abstract class that provides a common interface for different
/// multi-objective optimization solvers.
///
/// The different MultiObjectiveSolver implementations each use a different
/// Pareto-optimization library, which could lead to differences in performance
/// for various problem types. This base class allows the different
/// MultiObjectiveSolver implementations to be swapped out with each other
/// quickly and easily to help with testing, benchmarking, and experimentation.
class MultiObjectiveSolver
{
public:
  /// The MultiObjectiveSolver::Properties class contains Solver parameters that
  /// are common to all MultiObjectiveSolver types. Most (but not necessarily
  /// all) Solvers will make use of these parameters, and these parameters can
  /// be directly copied or transferred between all Solver types.
  struct Properties
  {
    /// Multi-objective optimization problem to be solved
    std::shared_ptr<MultiObjectiveProblem> mProblem;

    /// The maximum step size allowed for the Problem to be considered converged
    double mTolerance;

    /// The maximum number of iterations that the solver should use. Use 0 for
    /// infinite iterations.
    std::size_t mNumMaxIterations;

    /// How many iterations between printing the Solver's progress to the
    /// terminal. Use 0 for no printing.
    std::size_t mIterationsPerPrint;

    /// Stream for printing the Solver's progress. Default is std::cout.
    std::ostream* mOutStream;

    /// Set to true if the final result should be printed to the terminal.
    bool mPrintFinalResult;

    /// Publish the results of the optimization to a file. Leave this string
    /// empty to avoid publishing anything.
    std::string mResultFile;

    /// Publish the progress of the optimization to a file. Leave this string
    /// empty to avoid publishing anything.
    std::string mLogFilePath;

    Properties(
        std::shared_ptr<MultiObjectiveProblem> problem = nullptr,
        double tolerance = 1e-9,
        std::size_t numMaxIterations = 500,
        std::size_t iterationsPerPrint = 0,
        std::ostream* ostream = &std::cout,
        bool printFinalResult = false,
        const std::string& resultFile = "",
        const std::string& logFile = "");
  };

  /// Default constructor
  MultiObjectiveSolver(const Properties& properties = Properties());

  /// Destructor
  virtual ~MultiObjectiveSolver() = default;

  /// Solve optimization problem
  virtual bool solve() = 0;

  /// Set the generic Properties of this Solver
  void setProperties(const Properties& properties);

  /// Get the generic Properties of this Solver
  const Properties& getSolverProperties() const;

  virtual void setProblem(std::shared_ptr<MultiObjectiveProblem> problem);

  /// Get nonlinear optimization problem
  std::shared_ptr<MultiObjectiveProblem> getProblem() const;

  virtual const Eigen::MatrixXd& getPopulation() const = 0;
  virtual Eigen::VectorXd getPopulation(std::size_t sampleIndex) const = 0;

  virtual const Eigen::MatrixXd& getEliteSamples() const = 0;
  virtual Eigen::VectorXd getEliteSample(std::size_t sampleIndex) const = 0;

  virtual const Eigen::MatrixXd& getPopulationObjectives() const = 0;
  virtual Eigen::VectorXd getPopulationObjectives(
      std::size_t sampleIndex) const = 0;
  virtual double getPopulationObjective(
      std::size_t sampleIndex, std::size_t objectiveIndex) const = 0;

  virtual const Eigen::MatrixXd& getEliteObjectives() const = 0;
  virtual Eigen::VectorXd getEliteObjectives(std::size_t sampleIndex) const = 0;

  virtual const Eigen::Array<bool, Eigen::Dynamic, 1>&
  getPopulationNonDominance() const = 0;
  virtual bool getPopulationNonDominance(std::size_t sampleIndex) const = 0;

  virtual const Eigen::Array<bool, Eigen::Dynamic, 1>& getEliteNonDominance()
      const = 0;
  virtual bool getEliteNonDominance(std::size_t sampleIndex) const = 0;

  void setNumMaxIterations(std::size_t maxIterations);
  std::size_t getNumMaxIterations() const;

  void setLogname(const std::string& logName);
  const std::string& getLogName();

  void computeNondominance(
      const Eigen::MatrixXd& objectives,
      Eigen::Array<bool, Eigen::Dynamic, 1>& nondominance);

  void getNondominatedPopulation(
      const Eigen::MatrixXd& population,
      const Eigen::MatrixXd& objectives,
      const Eigen::Array<bool, Eigen::Dynamic, 1>& nondominance,
      Eigen::MatrixXd& nonDominatedPopulation,
      Eigen::MatrixXd& nonDominatedObjectives);

  void updateElitePopulation(
      const Eigen::MatrixXd& nonDominatedPopulation,
      const Eigen::MatrixXd& nonDominatedObjectives);

  void saveEliteLog();

  const Eigen::MatrixXd& getElitePopulation()
  {
    return mElitePopulation;
  }
  const Eigen::MatrixXd& getEliteObjectives()
  {
    return mEliteObjectives;
  }

  void setRecordHistory(bool on);

protected:
  Properties mProperties;

  Eigen::MatrixXd mElitePopulation;
  Eigen::MatrixXd mEliteObjectives;
  Eigen::Array<bool, Eigen::Dynamic, 1> mEliteNotDominated;

  bool mRecordPopulationHistory;
  std::size_t mHistorySteps;
};

} // namespace optimizer
} // namespace dart

#endif // DART_OPTIMIZER_MULTIOBJECTIVESOLVER_HPP_
