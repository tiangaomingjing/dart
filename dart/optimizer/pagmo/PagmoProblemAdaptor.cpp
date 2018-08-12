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

#include "dart/optimizer/pagmo/PagmoProblemAdaptor.hpp"

namespace dart {
namespace optimizer {

//==============================================================================
PagmoProblemAdaptor::PagmoProblemAdaptor()
{
  // Do nothing
}

//==============================================================================
PagmoProblemAdaptor::PagmoProblemAdaptor(std::shared_ptr<Problem> problem)
  : mDimension(problem->getDimension()), mObjNum(1), mProb(std::move(problem))
{
  // Do Nothing
}

//==============================================================================
pagmo::vector_double PagmoProblemAdaptor::fitness(
    const pagmo::vector_double& x) const
{
  const Eigen::VectorXd val = Eigen::VectorXd::Map(x.data(), x.size());
  return {mProb->getObjective()->eval(val)};
}

//==============================================================================
pagmo::vector_double::size_type PagmoProblemAdaptor::get_nobj() const
{
  return mObjNum;
}

//==============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double>
PagmoProblemAdaptor::get_bounds() const
{
  const Eigen::VectorXd& lb1 = mProb->getLowerBounds();
  const Eigen::VectorXd& ub1 = mProb->getUpperBounds();

  pagmo::vector_double lb2(lb1.data(), lb1.data() + lb1.size());
  pagmo::vector_double ub2(ub1.data(), ub1.data() + ub1.size());

  return std::make_pair(lb2, ub2);
}

//==============================================================================
pagmo::vector_double::size_type PagmoProblemAdaptor::get_nix() const
{
  pagmo::vector_double::size_type retval = 0u;
  return retval;
}

//==============================================================================
std::string PagmoProblemAdaptor::get_name() const
{
  // TODO(JS): Add name field to problem
  return "PagmoProblem";
}

} // namespace optimizer
} // namespace dart
