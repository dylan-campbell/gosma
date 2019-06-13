/*
 * Globally-Optimal Spherical Mixture Alignment (GOSMA)
 * Copyright (c) 2019 Dylan John Campbell
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Contact Details:
 * dylan.campbell@anu.edu.au
 * Brian Anderson Building 115, North Road,
 * Australian National University, Canberra ACT 2600, Australia
 *
 * Any publications resulting from the use of this code should cite the
 * following paper:
 *
 * Dylan Campbell, Lars Petersson, Laurent Kneip, Hongdong Li and Stephen Gould,
 * "The Alignment of the Spheres: Globally-Optimal Spherical Mixture Alignment
 * for Camera Pose Estimation", IEEE Conference on Computer Vision and Pattern
 * Recognition (CVPR), Long Beach, USA, IEEE, Jun. 2019
 *
 * For the full license, see LICENCE.md in the root directory
 * 
 * Author: Dylan Campbell
 * Date: 20190612
 * Revision: 2.0
 */

#ifndef POINT_SET_H_
#define POINT_SET_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm> // random_shuffle, count
#include <Eigen/Dense>

class PointSet {
 public:
  static const int kNumDimensions;  // Dimension of the point

  // Constructors
  PointSet();

  // Destructors
  ~PointSet();

  // Accessors
  int num_points() const {
    return num_points_;
  }
  int max_num_points() const {
    return max_num_points_;
  }
  Eigen::MatrixX3f points() const {
    return points_;
  }

  // Mutators
  void set_num_points(int num_points) {
    num_points_ = num_points;
  }
  void set_max_num_points(int max_num_points) {
    max_num_points_ = max_num_points;
  }
  void set_points(Eigen::MatrixX3f& points) {
    points_ = points;
  }

  // Public Class Functions
  int Load(std::string filename);
  void GetBoundingBox(Eigen::Vector3f centre, Eigen::Vector3f widths);
  void Translate(Eigen::RowVector3f translation);
  void Rotate(Eigen::Matrix3f rotation);
  void Scale(float scale);

 private:
  int num_points_;
  int max_num_points_;
  Eigen::MatrixX3f points_;

  // Private Class Functions
};

#endif /* POINT_SET_H_ */
