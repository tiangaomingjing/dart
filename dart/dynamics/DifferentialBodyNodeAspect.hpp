/*
 * Copyright (c) 2011-2017, The DART development contributors
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

#ifndef DART_DYNAMICS_DIFFERENTIALBODYNODEASPECT_HPP_
#define DART_DYNAMICS_DIFFERENTIALBODYNODEASPECT_HPP_

#include <Eigen/Dense>

#include "dart/common/AspectWithVersion.hpp"
#include "dart/common/Signal.hpp"
#include "dart/common/SpecializedForAspect.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EllipsoidShape.hpp"
#include "dart/dynamics/FixedFrame.hpp"
#include "dart/dynamics/TemplatedJacobianNode.hpp"
#include "dart/dynamics/detail/ShapeFrameAspect.hpp"
#include "dart/dynamics/Frame.hpp"

namespace dart {
namespace dynamics {

namespace detail {

struct DifferenctialBodyNodeState
{
  mutable std::vector<Eigen::Vector6d> mVq;
  mutable std::vector<Eigen::Vector6d> mVdq;

  //  mutable std::vector<std::vector<Eigen::Vector6d>> mVqq;
  //  mutable std::vector<std::vector<Eigen::Vector6d>> mVqdq;
  //  mutable std::vector<std::vector<Eigen::Vector6d>> mVdqdq;

  mutable std::vector<bool> mIsVelocityDerivWrtQDirty;
  mutable std::vector<bool> mIsVelocityDerivWrtQqDirty;

  //  mutable std::vector<std::vector<bool>> mIsVelocityDerivWrtQQDirty;
  //  mutable std::vector<std::vector<bool>> mIsVelocityDerivWrtQDqDirty;
  //  mutable std::vector<std::vector<bool>> mIsVelocityDerivWrtDqDqDirty;

  /// Constructor
  DifferenctialBodyNodeState();

  /// Destructor
  virtual ~DifferenctialBodyNodeState() = default;
};

} // namespace detail

//==============================================================================
class DifferentiableBodyNodeAspect final
    : public common::AspectWithState<DifferentiableBodyNodeAspect,
                                     detail::DifferenctialBodyNodeState,
                                     Frame>
{
public:
  using Base = common::AspectWithState<DifferentiableBodyNodeAspect,
                                       detail::DifferenctialBodyNodeState,
                                       Frame>;

  /// Constructor
  DifferentiableBodyNodeAspect(const StateData& state = StateData());

  DifferentiableBodyNodeAspect(const DifferentiableBodyNodeAspect&) = delete;

  const Eigen::Vector6d&
  getSpatialVelocityDerivWrtQ(std::size_t dofIndex) const;

protected:
  Frame* getFrame();

  const Frame* getFrame() const;
};

} // namespace dynamics
} // namespace dart

#endif // DART_DYNAMICS_DIFFERENTIALBODYNODEASPECT_HPP_
