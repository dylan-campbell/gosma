/*
 * Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 * github.com/jstraub/dpMMlowVar/blob/master/include/dpMMlowVar/dpmeans.hpp
 */

/*
 * File modified by Dylan Campbell
 * Date of modification: 20180928
 * Nature of modifications: minor, removed dependency on boost and simplified
 */

#ifndef DPMEANS_HPP_
#define DPMEANS_HPP_

#include <iostream>
#include <algorithm> // find
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "spherical_data.hpp"
#include "euclidean_data.hpp"

template<class T, class DS, int D>
class DPMeans {
 public:
  // Constructors
  DPMeans(double lambda);  // lambda = cos(lambda_in_degree * M_PI/180) - 1
  DPMeans(const DPMeans<T, DS, D>& o);

  // Destructors
  ~DPMeans();

  // Accessors
  int K() const {
    return static_cast<int>(K_);
  }
  Eigen::VectorXi Ns() const {
    Eigen::VectorXi Ns(Ns_.size());
    for (int i = 0; i < Ns_.size(); ++i)
      Ns(i) = Ns_[i];
    return Ns;
  }
  Eigen::VectorXi zs() const {
    Eigen::VectorXi zs(zs_.size());
    for (int i = 0; i < zs_.size(); ++i)
      zs(i) = zs_[i];
    return zs;
  }
  Eigen::MatrixXf mus() const {
    Eigen::MatrixXf mus(mus_.size(), D);
    for (int i = 0; i < mus_.size(); ++i)
      mus.row(i) = mus_[i].transpose();
    return mus;
  }

  bool GetCenter(uint32_t k, Eigen::Matrix<T, D, 1>& mu) const {
    if (k < K_) {
      mu = mus_[k];
      return true;
    } else {
      return false;
    }
  }
  bool GetX(uint32_t i, Eigen::Matrix<T, D, 1>& x) const {
    if (i < xs_.size()) {
      x = xs_[i];
      return true;
    } else {
      return false;
    }
  }

  // Adds an observation (adds obs, computes label, and potentially adds new
  // cluster depending on label assignment)
  void AddObservation(const Eigen::Matrix<T, D, 1>& x);
  // Updates all labels of all data currently stored with the object
  void UpdateLabels();
  // Updates all centers based on the current data and label assignments
  void UpdateCenters();
  // Iterate updates for centers and labels until cost function convergence
  bool IterateToConvergence(uint32_t max_iter);
  // Compute the current cost function value
  double Cost();
  // Filter clusters by minimum size
  int FilterClusters(int min_cluster_size);

  DPMeans<T, DS, D>& operator=(const DPMeans<T, DS, D>& o);

 protected:
  double lambda_;
  uint32_t K_;
  std::vector<Eigen::Matrix<T, D, 1>,
      Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>xs_;
  std::vector<Eigen::Matrix<T,D,1>,
      Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>> mus_;
  std::vector<uint32_t> zs_;
  std::vector<uint32_t> Ns_;

  // Resets all clusters (mus_ and Ks_) and resizes them to K_
  void ResetClusters();
  // Removes all empty clusters
  void RemoveEmptyClusters();
  // Computes the index of the closest cluster (may be K_ in which case a new
  // cluster has to be added)
  uint32_t IndOfClosestCluster(const Eigen::Matrix<T,D,1>& x, T& sim_closest);
};

// -------------------------------- Implementation -----------------------------
template<class T, class DS, int D>
DPMeans<T, DS, D>::DPMeans(double lambda)
    : lambda_(lambda),
      K_(0) {
}

template<class T, class DS, int D>
DPMeans<T, DS, D>::DPMeans(const DPMeans<T, DS, D>& o)
    : lambda_(o.lambda_),
      K_(o.K_),
      xs_(o.xs_),
      mus_(o.mus_),
      zs_(o.zs_),
      Ns_(o.Ns_) {
}

template<class T, class DS, int D>
DPMeans<T, DS, D>::~DPMeans() {
}

template<class T, class DS, int D>
DPMeans<T, DS, D>& DPMeans<T, DS, D>::operator=(const DPMeans<T, DS, D>& o) {
  if (&o == this)
    return *this;
  lambda_ = o.lambda_;
  K_ = o.K_;
  xs_ = o.xs_;
  zs_ = o.zs_;
  Ns_ = o.Ns_;
  if (o.mus_.empty()) {
    mus_.clear();
  } else {
    mus_ = o.mus_;
  }
  return *this;
}

template<class T, class DS, int D>
void DPMeans<T, DS, D>::AddObservation(const Eigen::Matrix<T, D, 1>& x) {
  xs_.push_back(x);
  T sim_closest = 0;
  uint32_t z = IndOfClosestCluster(x, sim_closest);
  if (z == K_) {
    mus_.push_back(x);
    ++K_;
  }
  zs_.push_back(z);
}

template<class T, class DS, int D>
uint32_t DPMeans<T, DS, D>::IndOfClosestCluster(
    const Eigen::Matrix<T, D, 1>& x, T& sim_closest) {
  uint32_t z_i = K_;
  // lambda for Euclidean, lambda + 1 for Spherical (DJC modified)
  sim_closest = lambda_ + DS::LambdaOffset();
  for (uint32_t k = 0; k < K_; ++k) {
    T sim_k = DS::Dist(mus_[k], x);
    if (DS::Closer(sim_k, sim_closest)) {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

template<class T, class DS, int D>
void DPMeans<T, DS, D>::UpdateLabels() {
  if (xs_.size() == 0)
    return;
  for (uint32_t i = 0; i < xs_.size(); ++i) {
    T sim_closest = 0;
    uint32_t z = IndOfClosestCluster(xs_[i], sim_closest);
    if (z == K_) {
      mus_.push_back(xs_[i]);
      ++K_;
    }
    zs_[i] = z;
  }
}

template<class T, class DS, int D>
void DPMeans<T, DS, D>::UpdateCenters() {
  if (xs_.size() == 0)
    return;
  ResetClusters();
  for (uint32_t i = 0; i < xs_.size(); ++i) {
    ++Ns_[zs_[i]];
  }
  DS::ComputeCenters(xs_, zs_, K_, mus_);
  RemoveEmptyClusters();
}

template<class T, class DS, int D>
void DPMeans<T, DS, D>::ResetClusters() {
  Ns_.resize(K_, 0);
  for (uint32_t k = 0; k < K_; ++k) {
    mus_[k].fill(0);
    Ns_[k] = 0;
  }
}

template<class T, class DS, int D>
void DPMeans<T, DS, D>::RemoveEmptyClusters() {
  if (K_ < 1)
    return;
  uint32_t kNew = K_;
  std::vector<bool> toDelete(K_, false);
  for (int32_t k = K_ - 1; k > -1; --k) {
    if (Ns_[k] == 0) {
      toDelete[k] = true;
      for (uint32_t i = 0; i < xs_.size(); ++i)
        if (static_cast<int32_t>(zs_[i]) >= k)
          zs_[i] -= 1;
      kNew--;
    }
  }
  uint32_t j = 0;
  for (uint32_t k = 0; k < K_; ++k) {
    if (toDelete[k]) {
      mus_[j] = mus_[k];
      Ns_[j] = Ns_[k];
      ++j;
    }
  }
  K_ = kNew;
  Ns_.resize(K_);
  mus_.resize(K_);
}

template<class T, class DS, int D>
double DPMeans<T, DS, D>::Cost() {
  double f = lambda_ * K_;
  for (uint32_t i = 0; i < xs_.size(); ++i) {
    f += DS::Dist(mus_[zs_[i]], xs_[i]);
  }
  return f;
}

template<class T, class DS, int D>
bool DPMeans<T, DS, D>::IterateToConvergence(uint32_t max_iter) {
  uint32_t iter = 0;
  double fPrev = 1e99;
  double f = Cost();
  while (iter < max_iter && fabs(fPrev - f) > 0.0f) {
    UpdateCenters();
    UpdateLabels();
    fPrev = f;
    f = Cost();
    ++iter;
  }
  return iter < max_iter;
}

template<class T, class DS, int D>
int DPMeans<T, DS, D>::FilterClusters(int min_cluster_size) {
  uint32_t K_new = 0;
  std::vector<Eigen::Matrix<T, D, 1>,
      Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>mus_new;
  std::vector<uint32_t> Ns_new;
  std::vector<uint32_t> removed_labels;
  for (int k = 0; k < K_; ++k) {
    if (Ns_[k] >= min_cluster_size) {
      ++K_new;
      mus_new.push_back(mus_[k]);
      Ns_new.push_back(Ns_[k]);
    } else {
      removed_labels.push_back(k);
    }
  }
  for (int i = 0; i < zs_.size(); ++i) {
    if (std::find(removed_labels.begin(), removed_labels.end(), zs_[i])
        != removed_labels.end()) {  // if point is in a removed cluster
      zs_[i] = K_;
    }
  }
  for (int i = 0; i < zs_.size(); ++i) {
    int label_difference = 0;
    for (auto l : removed_labels) {
      if (zs_[i] > l && zs_[i] != K_) {
        label_difference++;
      }
    }
    zs_[i] -= label_difference;
  }
  K_ = K_new;
  mus_ = mus_new;
  Ns_ = Ns_new;
  return removed_labels.size();
}

#endif /* DPMEANS_HPP_ */
