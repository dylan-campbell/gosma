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

#include "point_set.h"

const int PointSet::kNumDimensions = 3;

PointSet::PointSet() {
  num_points_ = -1;
  max_num_points_ = 10000000;
}

PointSet::~PointSet() {
}

/*
 * Public Class Functions
 */

/*
 * Load whitespace-separated point-set from a file
 * Determines point-set size and dimension automatically
 */
int PointSet::Load(std::string filename) {
  std::string line, s;
  std::istringstream ss;

  std::ifstream ifile(filename.c_str(), std::ios_base::in);
  if (ifile.is_open()) {
    // Ascertain the number of rows and columns
    // Includes newlines at end of document
    int num_rows = std::count(std::istreambuf_iterator<char>(ifile),
                              std::istreambuf_iterator<char>(), '\n');
    ifile.seekg(0);  // Return to beginning of file
    getline(ifile, line);
    ss.str(line);
    int num_cols = 0;
    while (ss >> s)
      num_cols++;
    ifile.seekg(0);  // Return to beginning of file

    // If user specifies to use all points, do not randomise the point-set
    if (max_num_points_ == 0 || num_rows <= max_num_points_) {
      num_points_ = num_rows;
      points_.resize(num_points_, Eigen::NoChange);
      for (int i = 0; i < num_points_; ++i) {
        for (int j = 0; j < kNumDimensions; ++j) {
          ifile >> points_(i, j);
        }
        // Ignore any additional data in the row
        ifile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
      // Otherwise, randomly sample the point-set
    } else {
      Eigen::MatrixX3f points_all(num_points_, 3);
      for (int i = 0; i < num_points_; ++i) {
        for (int j = 0; j < kNumDimensions; ++j) {
          ifile >> points_all(i, j);
        }
        // Ignore any additional data in the row
        ifile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
      std::vector<int> random_indices;
      for (int i = 0; i < num_rows; ++i)
        random_indices.push_back(i);
      std::random_shuffle(random_indices.begin(), random_indices.end());
      num_points_ = std::min(num_rows, max_num_points_);
      points_.resize(num_points_, Eigen::NoChange);
      for (int i = 0; i < num_points_; ++i) {
        for (int j = 0; j < kNumDimensions; ++j) {
          points_(i, j) = points_all(random_indices[i], j);
        }
      }
    }
    ifile.close();
  } else {
    num_points_ = -1;
    std::cout << "Unable to open point-set file '" << filename << "'"
              << std::endl;
  }

  return num_points_;
}

void PointSet::GetBoundingBox(Eigen::Vector3f centre, Eigen::Vector3f widths) {
  Eigen::Vector3f points_min = points_.colwise().minCoeff().transpose();
  Eigen::Vector3f points_max = points_.colwise().maxCoeff().transpose();
  centre = (points_max + points_min) / 2;
  widths = points_max - points_min;
}

void PointSet::Translate(Eigen::RowVector3f translation) {
  points_ = points_.rowwise() - translation;
}

void PointSet::Rotate(Eigen::Matrix3f rotation) {
  points_ *= rotation.transpose();
}

void PointSet::Scale(float scale) {
  points_ *= scale;
}

/*
 * Private Class Functions
 */
