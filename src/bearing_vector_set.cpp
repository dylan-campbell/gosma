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

#include "bearing_vector_set.h"

const int BearingVectorSet::kNumDimensions = 3;

BearingVectorSet::BearingVectorSet() {
  num_bearing_vectors_ = -1;
  max_num_bearing_vectors_ = 10000000;
}

BearingVectorSet::~BearingVectorSet() {
}

/*
 * Public Class Functions
 */

/*
 * Load whitespace-separated bearing vector set from a file
 * Determines bearing vector set size and dimension automatically
 */
int BearingVectorSet::Load(std::string filename) {
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

    // If user specifies to use all bearing vectors,
    // do not randomise the bearing vectors set
    if (max_num_bearing_vectors_ == 0 || num_rows <= max_num_bearing_vectors_) {
      num_bearing_vectors_ = num_rows;
      bearing_vectors_.resize(num_bearing_vectors_, Eigen::NoChange);
      for (int i = 0; i < num_bearing_vectors_; ++i) {
        for (int j = 0; j < kNumDimensions; ++j) {
          ifile >> bearing_vectors_(i, j);
        }
        // Ignore any additional data in the row
        ifile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
      // Otherwise, randomly sample the set
    } else {
      Eigen::MatrixX3f bearing_vectors_all(num_bearing_vectors_, 3);
      for (int i = 0; i < num_bearing_vectors_; ++i) {
        for (int j = 0; j < kNumDimensions; ++j) {
          ifile >> bearing_vectors_all(i, j);
        }
        // Ignore any additional data in the row
        ifile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
      std::vector<int> random_indices;
      for (int i = 0; i < num_rows; ++i)
        random_indices.push_back(i);
      std::random_shuffle(random_indices.begin(), random_indices.end());
      num_bearing_vectors_ = std::min(num_rows, max_num_bearing_vectors_);
      bearing_vectors_.resize(num_bearing_vectors_, Eigen::NoChange);
      for (int i = 0; i < num_bearing_vectors_; ++i) {
        for (int j = 0; j < kNumDimensions; ++j) {
          bearing_vectors_(i, j) = bearing_vectors_all(random_indices[i], j);
        }
      }
    }
    ifile.close();
  } else {
    num_bearing_vectors_ = -1;
    std::cout << "Unable to open bearing-vector-set file '" << filename << "'"
              << std::endl;
  }
  return num_bearing_vectors_;
}

void BearingVectorSet::Rotate(Eigen::Matrix3f rotation) {
  bearing_vectors_ *= rotation.transpose();
}

void BearingVectorSet::NormaliseBearings() {
  for (int i = 0; i < num_bearing_vectors_; ++i) {
    bearing_vectors_.row(i).normalize();
  }
}

/*
 * Private Class Functions
 */
