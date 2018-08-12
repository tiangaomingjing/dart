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

#ifndef DART_OPTIMIZER_PAGMO_PAGMOMULTIOBJECTIVEPROBLEMADAPTOR_HPP_
#define DART_OPTIMIZER_PAGMO_PAGMOMULTIOBJECTIVEPROBLEMADAPTOR_HPP_

#include <string>
#include <pagmo/pagmo.hpp>
#include "dart/optimizer/MultiObjectiveProblem.hpp"

namespace dart {
namespace optimizer {

class PagmoMultiObjectiveProblemAdaptor
{
public:
  PagmoMultiObjectiveProblemAdaptor();

  PagmoMultiObjectiveProblemAdaptor(
      std::shared_ptr<MultiObjectiveProblem> problem);

  pagmo::vector_double fitness(const pagmo::vector_double& x) const;

  pagmo::vector_double::size_type get_nobj() const;

  std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;

  pagmo::vector_double::size_type get_nix() const;

  std::string get_name() const;

  template <typename Archive>
  void serialize(Archive& ar)
  {
    ar(mDimension, mObjNum);
  }

protected:
  unsigned int mDimension;
  unsigned int mObjNum;
  std::shared_ptr<MultiObjectiveProblem> mProb;
};

} // namespace optimizer
} // namespace dart

#endif // DART_OPTIMIZER_PAGMO_PAGMOMULTIOBJECTIVEPROBLEMADAPTOR_HPP_
