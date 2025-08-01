/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include <pcl/registration/warp_point_rigid.h>

namespace pcl {
namespace registration {
/** \brief @b WarpPointRigid4D enables 4D (1D rotation + 3D translation)
 * transformations for points.
 *
 * \note The class is templated on the source and target point types as well as on the
 * output scalar of the transformation matrix (i.e., float or double). Default: float.
 * \author Radu B. Rusu, modified by Saad Ejaz
 * \ingroup registration
 */
template <typename PointSourceT, typename PointTargetT, typename Scalar = float>
class WarpPointRigid4D : public WarpPointRigid<PointSourceT, PointTargetT, Scalar> {
public:
  using Matrix4 = typename WarpPointRigid<PointSourceT, PointTargetT, Scalar>::Matrix4;
  using VectorX = typename WarpPointRigid<PointSourceT, PointTargetT, Scalar>::VectorX;

  using Ptr = shared_ptr<WarpPointRigid4D<PointSourceT, PointTargetT, Scalar>>;
  using ConstPtr =
      shared_ptr<const WarpPointRigid4D<PointSourceT, PointTargetT, Scalar>>;

  /** \brief Constructor. */
  WarpPointRigid4D() : WarpPointRigid<PointSourceT, PointTargetT, Scalar>(4) {}

  /** \brief Empty destructor */
  ~WarpPointRigid4D() override = default;

  /** \brief Set warp parameters.
   * \param[in] p warp parameters (tx ty tz rz)
   */
  void
  setParam(const VectorX& p) override
  {
    assert(p.rows() == this->getDimension());
    Matrix4& trans = this->transform_matrix_;

    trans = Matrix4::Zero();
    trans(3, 3) = 1;
    trans(2, 2) = 1; // Rotation around the Z-axis
  

    // Copy the rotation and translation components
    trans.template block<4, 1>(0, 3) = Eigen::Matrix<Scalar, 4, 1>(p[0], p[1], p[2], 1.0);

    // Compute w from the unit quaternion
    Eigen::Rotation2D<Scalar> r(p[3]);
    trans.template topLeftCorner<2, 2>() = r.toRotationMatrix();
  }
};
} // namespace registration
} // namespace pcl