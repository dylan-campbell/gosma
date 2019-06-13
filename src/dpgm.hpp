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

#ifndef DPGM_H_
#define DPGM_H_

#include <iostream>
#include <Eigen/Dense>
#include "dpmeans.hpp"

class DPGM {
 public:
  // Constructors
  DPGM()
      : num_points_(0),
        num_components_(0),
        lambda_(0.0f) {
  }

  DPGM(const Eigen::MatrixX3f& points, float lambda)
      : num_points_(points.rows()),
        num_components_(0),
        lambda_(lambda),
        min_variance_(0.1f),
        points_(points) {
  }

  DPGM(const Eigen::MatrixX3f& points, float lambda, float min_variance)
      : num_points_(points.rows()),
        num_components_(0),
        lambda_(lambda),
        min_variance_(min_variance),
        points_(points) {
  }

  // Destructors
  ~DPGM() {
  }

  // Accessors
  int num_points() const {
    return num_points_;
  }
  int num_components() const {
    return num_components_;
  }
  float lambda() const {
    return lambda_;
  }
  float min_variance() const {
    return min_variance_;
  }
  Eigen::MatrixX3f points() const {
    return points_;
  }
  Eigen::MatrixX3f mus() const {
    return mus_;
  }
  Eigen::VectorXf variances() const {
    return variances_;
  }
  Eigen::VectorXf phis() const {
    return phis_;
  }

  // Mutators
  void set_num_points(int num_points) {
    num_points_ = num_points;
  }
  void set_num_components(int num_components) {
    num_components_ = num_components;
  }
  void set_lambda(float lambda) {
    lambda_ = lambda;
  }
  void set_min_variance(float min_variance) {
    min_variance_ = min_variance;
  }
  void set_points(Eigen::MatrixX3f points) {
    points_ = points;
  }
  void set_mus(Eigen::MatrixX3f mus) {
    mus_ = mus;
  }
  void set_variances(Eigen::VectorXf variances) {
    variances_ = variances;
  }
  void set_phis(Eigen::VectorXf phis) {
    phis_ = phis;
  }

  // Public Class Functions
  void Construct();

 private:
  int num_points_;
  int num_components_;
  float lambda_;
  float min_variance_;
  Eigen::MatrixX3f points_;
  Eigen::MatrixX3f mus_;
  Eigen::VectorXf variances_;
  Eigen::VectorXf phis_;

  // Private Class Functions
};

//============================= Implementation =================================

void DPGM::Construct() {
  DPMeans<float, Euclidean<float>, 3> dpmeans(lambda_);
  // Add points to clusterer
  for (int i = 0; i < num_points_; ++i) {
    dpmeans.AddObservation(points_.row(i).transpose());
  }
  // Run the clustering algorithm for at most 10 steps
  dpmeans.IterateToConvergence(10);
  // Filter clusters by minimum size
  int min_num_points_per_cluster = 1;
  int num_clusters_removed = dpmeans.FilterClusters(min_num_points_per_cluster);
  // Get outputs
  int K = dpmeans.K();
  Eigen::VectorXi labels = dpmeans.zs();
  Eigen::VectorXf counts = dpmeans.Ns().cast<float>();
  Eigen::MatrixX3f mus = dpmeans.mus();

  // Each point contributes an equal weight
  // ToDo: density-based point weights
  Eigen::VectorXf weights = Eigen::VectorXf::Constant(num_points_, 1.0f);

  // Compute Gaussian statistics
  Eigen::VectorXf weight_sum_per_cluster = Eigen::VectorXf::Zero(K); // Sum of weights per label
  Eigen::VectorXf sum_squared_distances = Eigen::VectorXf::Zero(K); // Sum of weighted squared point-centroid distances
  Eigen::MatrixX3f point_sum_per_cluster = Eigen::MatrixX3f::Zero(
      K, Eigen::NoChange);  // For weighted mu calc
  // Compute weight sums and weighted point sums
  for (int i = 0; i < num_points_; ++i) {
    if (labels(i) < K) {
      weight_sum_per_cluster(labels(i)) += weights(i);
      point_sum_per_cluster.row(labels(i)) += weights(i) * points_.row(i);
    }
  }
  // Compute weighted centroids
  for (int k = 0; k < K; ++k) {
    mus.row(k) = point_sum_per_cluster.row(k) / weight_sum_per_cluster(k);
  }
  // Compute variance statistics
  for (int i = 0; i < points_.rows(); ++i) {
    if (labels(i) < K) {
      sum_squared_distances(labels(i)) += weights(i)
          * (points_.row(i) - mus.row(labels(i)))
          * (points_.row(i) - mus.row(labels(i))).transpose();
    }
  }
  // Update class members
  num_components_ = K;
  mus_ = mus;
  variances_ = (sum_squared_distances.array() / weight_sum_per_cluster.array())
      .matrix();
  // Impose minimum variance (to avoid div0 errors)
  variances_ = (variances_.array() < min_variance_)
      .select(min_variance_, variances_);
  // Use constant phi
  // ToDo: consider other ways to choose phi
  phis_.setConstant(num_components_, 1, 1.0f / num_components_);
}

#endif /* DPGM_H_ */
