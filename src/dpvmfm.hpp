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

#ifndef DPVMFM_H_
#define DPVMFM_H_

#include <iostream>
#include <Eigen/Dense>
#include "dpmeans.hpp"

class DPVMFM {
 public:
  // Constructors
  DPVMFM()
      : num_bearing_vectors_(0),
        num_components_(0),
        lambda_(0.0f) {
  }

  DPVMFM(const Eigen::MatrixX3f& bearing_vectors, float lambda)
      : num_bearing_vectors_(bearing_vectors.rows()),
        num_components_(0),
        lambda_(lambda),
        max_kappa_(1000.0f),
        bearing_vectors_(bearing_vectors) {
  }

  DPVMFM(const Eigen::MatrixX3f& bearing_vectors, float lambda, float max_kappa)
      : num_bearing_vectors_(bearing_vectors.rows()),
        num_components_(0),
        lambda_(lambda),
        max_kappa_(max_kappa),
        bearing_vectors_(bearing_vectors) {
  }

  // Destructors
  ~DPVMFM() {
  }

  // Accessors
  int num_bearing_vectors() const {
    return num_bearing_vectors_;
  }
  int num_components() const {
    return num_components_;
  }
  float lambda() const {
    return lambda_;
  }
  float max_kappa() const {
    return max_kappa_;
  }
  Eigen::MatrixX3f bearing_vectors() const {
    return bearing_vectors_;
  }
  Eigen::MatrixX3f mus() const {
    return mus_;
  }
  Eigen::VectorXf kappas() const {
    return kappas_;
  }
  Eigen::VectorXf phis() const {
    return phis_;
  }

  // Mutators
  void set_num_bearing_vectors(int num_bearing_vectors) {
    num_bearing_vectors_ = num_bearing_vectors;
  }
  void set_num_components(int num_components) {
    num_components_ = num_components;
  }
  void set_lambda(float lambda) {
    lambda_ = lambda;
  }
  void set_max_kappa(float max_kappa) {
    max_kappa_ = max_kappa;
  }
  void set_bearing_vectors(Eigen::MatrixX3f bearing_vectors) {
    bearing_vectors_ = bearing_vectors;
  }
  void set_mus(Eigen::MatrixX3f mus) {
    mus_ = mus;
  }
  void set_kappas(Eigen::VectorXf kappas) {
    kappas_ = kappas;
  }
  void set_phis(Eigen::VectorXf phis) {
    phis_ = phis;
  }

  // Public Class Functions
  void Construct();

 private:
  int num_bearing_vectors_;
  int num_components_;
  float lambda_;
  float max_kappa_;
  Eigen::MatrixX3f bearing_vectors_;
  Eigen::MatrixX3f mus_;
  Eigen::VectorXf kappas_;
  Eigen::VectorXf phis_;

  // Private Class Functions
  float MLEstimateKappa(float norm);
};

//============================= Implementation =================================

void DPVMFM::Construct() {
  DPMeans<float, Spherical<float>, 3> dpmeans(lambda_);
  // Add bearing_vectors to clusterer
  for (int i = 0; i < num_bearing_vectors_; ++i) {
    dpmeans.AddObservation(bearing_vectors_.row(i).transpose());
  }
  // Run the clustering algorithm for at most 10 steps
  dpmeans.IterateToConvergence(10);
  // Filter clusters by minimum size
  int min_num_bearing_vectors_per_cluster = 1;
  int num_clusters_removed = dpmeans.FilterClusters(
      min_num_bearing_vectors_per_cluster);
  // Get outputs
  int K = dpmeans.K();
  Eigen::VectorXi labels = dpmeans.zs();
  Eigen::VectorXf counts = dpmeans.Ns().cast<float>();
  Eigen::MatrixX3f mus = dpmeans.mus();

  // Each bearing_vector contributes an equal weight
  // ToDo: density-based bearing_vector weights
  Eigen::VectorXf weights =
      Eigen::VectorXf::Constant(num_bearing_vectors_, 1.0f);

  // Compute vMF statistics: area-weighted sum over bearing vectors associated
  // with respective cluster
  Eigen::VectorXf weight_sum_per_cluster = Eigen::VectorXf::Zero(K); // Sum of weights per label
  Eigen::MatrixX3f bearing_vector_sum_per_cluster = Eigen::MatrixX3f::Zero(
      K, Eigen::NoChange);  // For weighted mu calc
  // Compute weight sums and weighted bearing_vector sums
  for (int i = 0; i < num_bearing_vectors_; ++i) {
    if (labels(i) < K) {
      weight_sum_per_cluster(labels(i)) += weights(i);
      bearing_vector_sum_per_cluster.row(labels(i)) += weights(i)
          * bearing_vectors_.row(i);
    }
  }
  // Compute weighted centroids and kappa
  mus_.resize(K, Eigen::NoChange);
  kappas_.resize(K);
  for (int k = 0; k < K; ++k) {
    mus_.row(k) = bearing_vector_sum_per_cluster.row(k)
        / weight_sum_per_cluster(k);
    float norm = mus_.row(k).norm();  // norm in [0, 1]
    mus_.row(k) /= norm;
    float kappa = MLEstimateKappa(norm);
    if (kappa > max_kappa_) {
      kappas_(k) = max_kappa_;
    } else {
      kappas_(k) = kappa;
    }
  }
  // Update class members
  num_components_ = K;
  // Use constant phi
  // ToDo: consider other ways to set phi
  phis_.setConstant(num_components_, 1, 1.0f / num_components_);
}

float DPVMFM::MLEstimateKappa(float norm) {
  // norm in [0, 1] -> kappa in [0, inf), therefore cap norm
  if (norm >= 0.999999f)
    norm = 0.999999f;
  double kappa = 1.0;
  double prev_kappa = 0.;
  double eps = 1e-8;
  double R = static_cast<double>(norm);
  while (fabs(kappa - prev_kappa) > eps) {
    double inv_tanh_kappa = 1.0f / tanh(kappa);
    double inv_kappa = 1.0f / kappa;
    double f = -inv_kappa + inv_tanh_kappa - R;
    double df = inv_kappa * inv_kappa - inv_tanh_kappa * inv_tanh_kappa + 1.0f;
    prev_kappa = kappa;
    kappa -= f / df;
  }
  return static_cast<float>(kappa);
}

#endif /* DPVMFM_H_ */
