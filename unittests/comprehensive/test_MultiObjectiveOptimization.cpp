#include <fstream>
//#include <Python.h>
#include <gtest/gtest.h>
#include <dart/common/Memory.hpp>
#include <dart/optimizer/Function.hpp>
#include <dart/optimizer/MultiObjectiveProblem.hpp>

using namespace dart;
using dart::optimizer::Function;
using dart::optimizer::FunctionPtr;
using dart::optimizer::UniqueFunctionPtr;

//==============================================================================
int dimension = 10;
Eigen::VectorXd lowerLimits = Eigen::VectorXd::Zero(dimension);
Eigen::VectorXd upperLimits = Eigen::VectorXd::Constant(dimension, 1.0);

//==============================================================================
class Func1 : public Function
{
public:
  Func1() = default;

  double eval(const Eigen::VectorXd& x) override
  {
    return x[0];
  }

  UniqueFunctionPtr clone() const
  {
    return dart::common::make_unique<Func1>(*this);
  }

  std::size_t getParameterDimension() const
  {
    return 1u;
  }
};

//==============================================================================
class Func2 : public Function
{
public:
  Func2() = default;

  double eval(const Eigen::VectorXd& x) override
  {
    double g = 1.0 + 9 * (x.sum() - x[0]) / double(dimension - 1);
    return g * (1.0 - std::sqrt(x[0] / g));
  }

  UniqueFunctionPtr clone() const
  {
    return dart::common::make_unique<Func2>(*this);
  }

  std::size_t getParameterDimension() const
  {
    return 1u;
  }
};

//==============================================================================
template <typename SolverType>
void testZDT1(bool initialize, bool finalize)
{
  auto pFunc1 = std::make_shared<Func1>();
  auto pFunc2 = std::make_shared<Func2>();

  std::vector<FunctionPtr> pFuncs;
  pFuncs.push_back(pFunc1);
  pFuncs.push_back(pFunc2);

#ifdef NDEBUG // release mode
  int numSamples = 200;
#else
  int numSamples = 100;
#endif
  double desiredAR = 0.2;
#ifdef NDEBUG // release mode
  std::size_t iterationNum = 1000;
#else
  std::size_t iterationNum = 200;
#endif

  auto problem = std::make_shared<optimizer::MultiObjectiveProblem>(
      dimension, numSamples);
  problem->setObjectives(pFuncs);
  problem->setLowerBounds(lowerLimits);
  problem->setUpperBounds(upperLimits);

  // auto sampler =
  // std::make_shared<planner::optimization::DeterministicSampler>(
  //    dimension, "population1.csv");
  // planner::optimization::MomcmcOptimizer solver(
  //    problem, desiredAR, sampler);

  SolverType solver(problem, desiredAR);
  solver.setNumMaxIterations(iterationNum);
  solver.setF(0.5);
  solver.setCR(0.7);
  solver.solve();

  Eigen::MatrixXd population = solver.getPopulation();
  Eigen::MatrixXd popObjectives = solver.getPopulationObjectives();

  Eigen::MatrixXd elitePopulation = solver.getElitePopulation();
  Eigen::MatrixXd eliteObjectives = solver.getEliteObjectives();

  const static Eigen::IOFormat CSVFormat(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

  std::ofstream popFile;
  popFile.open("ZDT1_pop.csv");
  popFile << population.format(CSVFormat);
  popFile.close();

  std::ofstream objFile;
  objFile.open("ZDT1_obj.csv");
  objFile << popObjectives.format(CSVFormat);
  objFile.close();

  std::ofstream elitePopFile;
  elitePopFile.open("ZDT1_elite_pop.csv");
  elitePopFile << elitePopulation.format(CSVFormat);
  elitePopFile.close();

  std::ofstream eliteObjFile;
  eliteObjFile.open("ZDT1_elite_obj.csv");
  eliteObjFile << eliteObjectives.format(CSVFormat);
  eliteObjFile.close();

//  // Initialize the Python Interpreter
//  if (initialize)
//    Py_Initialize();

//  PyRun_SimpleString(
//      "from __future__ import print_function                             \n"
//      "import matplotlib.pyplot as plt                                   \n"
//      "import numpy as np                                                \n"
//      "                                                                  \n"
//      "if __name__ == '__main__':                                        \n"
//      "                                                                  \n"
//      "                                                                  \n"
//      "    samplesObjectives = np.loadtxt('ZDT1_obj.csv', delimiter=',') \n"
//      "    eliteSampleObjectives = np.loadtxt('ZDT1_elite_obj.csv', "
//      "delimiter=',') \n"
//      "                                                                  \n"
//      "    pX = np.arange(0.0, 1.0, 0.01)                                \n"
//      "    pY = 1 - np.sqrt(pX)                                          \n"
//      "                                                                  \n"
//      "    fig = plt.figure(1)                                           \n"
//      "    ax = fig.add_subplot(111)                                     \n"
//      "    ax.plot(pX, pY,'r-')                                          \n"
//      "    ax.plot(samplesObjectives[:,0], samplesObjectives[:,1],'o')   \n"
//      "    fig1 = plt.figure(2)                                          \n"
//      "    ax2 = fig1.add_subplot(111)                                   \n"
//      "    ax2.plot(eliteSampleObjectives[:,0], "
//      "eliteSampleObjectives[:,1],'o')  \n"
//      "    #ax.set_xlim([-0.5, 1.5])                                     \n"
//      "    #ax.set_ylim([-0.5, 1.5])                                     \n"
//      "                                                                  \n"
//      "    plt.show()                                                    \n");

//  // Finish the Python Interpreter
//  if (finalize)
//    Py_Finalize();
}

//==============================================================================
TEST(ZDT1, Basic)
{
//  testZDT1<optimizer::MomcmcSolver>(true, false);
//  testZDT1<planner::optimization::ProgressiveMomcmcOptimizer>(false, true);
}
