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

#include "dart/dynamics/DifferentialBodyNodeAspect.hpp"

namespace dart {
namespace dynamics {

namespace detail {

//==============================================================================
DifferenctialBodyNodeState::DifferenctialBodyNodeState()
{
  // Do nothing
}

} // namespace detail

//==============================================================================
DifferentiableBodyNodeAspect::DifferentiableBodyNodeAspect(
    const StateData& state)
  : DifferentiableBodyNodeAspect::Base(state)
{
  // Do nothing
}

//==============================================================================
const Eigen::Vector6d&
DifferentiableBodyNodeAspect::getSpatialVelocityDerivWrtQ(
    std::size_t dofIndex) const
{
//  auto* frame = getFrame();

//  if (frame->isWorld())
//    return mState.mVq[dofIndex];

//  if (mState.mIsVelocityDerivWrtQDirty[dofIndex])
//  {
//    auto* parentFrame = frame->getParentFrame();
//    const Eigen::Isometry3d& tf = frame->getRelativeTransform();
////    auto parentAspect = parentFrame->getAspect<ThisClass>();
////    const Eigen::Vector6d& parentVq =
////        parentAspect->getSpatialVelocityDerivWrtQ(dofIndex);

////    mState.mVq[dofIndex] = math::AdInvT(tf, parentVq);

//    mState.mIsVelocityDerivWrtQDirty[dofIndex] = false;
//  }

  return mState.mVq[dofIndex];
}

//==============================================================================
Frame* DifferentiableBodyNodeAspect::getFrame()
{
  return mComposite;
}

//==============================================================================
const Frame* DifferentiableBodyNodeAspect::getFrame() const
{
  return mComposite;
}

} // namespace dynamics
} // namespace dart
