/*
 * Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 * github.com/jstraub/dpMMlowVar/blob/master/include/dpMMlowVar/euclideanData.hpp
 */

/*
 * File modified by Dylan Campbell
 * Date of modification: 20180928
 * Nature of modifications: minor, removed dependency on jsCore and simplified
 */

#ifndef EUCLIDEAN_DATA_HPP_
#define EUCLIDEAN_DATA_HPP_

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>

using Eigen::Dynamic;
using Eigen::Matrix;
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> VectorXu;

template<typename T>
struct Euclidean {
  class Cluster {
   protected:
    Matrix<T, Dynamic, 1> centroid_;
    Matrix<T, Dynamic, 1> xSum_;
    uint32_t N_;

   public:
    Cluster()
        : centroid_(0, 1),
          xSum_(0, 1),
          N_(0) {
    }

    Cluster(uint32_t D)
        : centroid_(D, 1),
          xSum_(0, 1),
          N_(0) {
    }

    Cluster(const Matrix<T, Dynamic, 1>& x_i)
        : centroid_(x_i),
          xSum_(x_i),
          N_(1) {
    }

    Cluster(const Matrix<T, Dynamic, 1>& xSum, uint32_t N)
        : centroid_(xSum),
          xSum_(xSum),
          N_(N) {
      if (N > 0)
        centroid_ /= N_;
    }

    T Dist(const Matrix<T, Dynamic, 1>& x_i) const {
      return Euclidean::Dist(this->centroid_, x_i);
    }

    void ComputeSS(
        const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z,
        const uint32_t k) {
      const uint32_t D = x.rows();
      const uint32_t N = x.cols();
      N_ = 0;
      xSum_.setZero(D);
      for (uint32_t i = 0; i < N; ++i) {
        if (z(i) == k) {
          xSum_ += x.col(i);
          ++N_;
        }
      }
      if (N_ == 0)
        xSum_ = x.col(k);
    }

    void UpdateCenter() {
      assert(this->centroid()(0) == this->centroid()(0));
      if (N_ > 0)
        centroid_ = xSum_ / N_;
    }

    void ComputeCenter(
        const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z,
        const uint32_t k) {
      ComputeSS(x, z, k);
      UpdateCenter();
    }

    bool IsInstantiated() const {
      return this->N_ > 0;
    }

    uint32_t N() const {
      return N_;
    }
    uint32_t& N() {
      return N_;
    }
    const Matrix<T, Dynamic, 1>& centroid() const {
      return centroid_;
    }
    Matrix<T, Dynamic, 1>& centroid() {
      return centroid_;
    }
    const Matrix<T, Dynamic, 1>& xSum() const {
      return xSum_;
    }
  };

  class DependentCluster : public Cluster {
   protected:
    // Variables
    T t_;
    T w_;
    // Parameters
    T tau_;
    T lambda_;
    T Q_;
    Matrix<T, Dynamic, 1> prev_centroid_;

   public:
    DependentCluster()
        : Cluster(),
          t_(0),
          w_(0),
          tau_(1),
          lambda_(1),
          Q_(1),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(uint32_t D)
        : Cluster(D),
          t_(0),
          w_(0),
          tau_(1),
          lambda_(1),
          Q_(1),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const Matrix<T, Dynamic, 1>& x_i)
        : Cluster(x_i),
          t_(0),
          w_(0),
          tau_(1),
          lambda_(1),
          Q_(1),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const Matrix<T, Dynamic, 1>& x_i, T tau, T lambda, T Q)
        : Cluster(x_i),
          t_(0),
          w_(0),
          tau_(tau),
          lambda_(lambda),
          Q_(Q),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(
        const Matrix<T, Dynamic, 1>& x_i, const DependentCluster& cl0)
        : Cluster(x_i),
          t_(0),
          w_(0),
          tau_(cl0.tau()),
          lambda_(cl0.lambda()),
          Q_(cl0.Q()),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(T tau, T lambda, T Q)
        : Cluster(),
          t_(0),
          w_(0),
          tau_(tau),
          lambda_(lambda),
          Q_(Q),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const DependentCluster& b)
        : Cluster(b.xSum(), b.N()),
          t_(b.t()),
          w_(b.w()),
          tau_(b.tau()),
          lambda_(b.lambda()),
          Q_(b.Q()),
          prev_centroid_(b.prev_centroid()) {
      this->centroid_ = b.centroid();
    }

    bool IsDead() const {
      return t_ * Q_ > lambda_;
    }
    bool IsNew() const {
      return t_ == 0;
    }

    void IncAge() {
      ++t_;
    }

    const Matrix<T, Dynamic, 1>& prev_centroid() const {
      return prev_centroid_;
    }
    Matrix<T, Dynamic, 1>& prev_centroid() {
      return prev_centroid_;
    }

    void NextTimeStep() {
      this->N_ = 0;
      this->prev_centroid_ = this->centroid_;
    }

    void UpdateWeight() {
      w_ = (w_ == 0) ? this->N_ : 1.0f / (1.0f / w_ + t_ * tau_) + this->N_;
      t_ = 0;
    }

    void Print() const {
      std::cout << "cluster globId = " << globalId << "\tN = " << this->N_
          << "\tage = " << t_ << "\tweight = " << w_ << "\t dead? "
          << this->IsDead() << "  center: " << this->centroid().transpose()
          << std::endl << "  xSum: " << this->xSum_.transpose() << std::endl;
      assert(this->centroid()(0) == this->centroid()(0));
      assert(!(this->centroid().array() == 0).all());
    }

    DependentCluster* Clone() {
      return new DependentCluster(*this);
    }

    void ReInstantiate() {
      const T gamma = 1.0f / (1.0f / w_ + t_ * tau_);
      this->centroid_ = (this->centroid_ * gamma + this->xSum_)
          / (gamma + this->N_);
    }

    void ReInstantiate(const Matrix<T, Dynamic, Dynamic>& x_i) {
      this->xSum_ = x_i;
      this->N_ = 1;
      ReInstantiate();
    }

    T MaxDist() const {
      return this->lambda_;
    }

    T Dist(const Matrix<T, Dynamic, 1>& x_i) const {
      if (this->IsInstantiated()) {
        return Euclidean::Dist(this->centroid_, x_i);
      } else {
        return Euclidean::Dist(this->centroid_, x_i)
            / (tau_ * t_ + 1.0f + 1.0f / w_) + Q_ * t_;
      }
    }

    T tau() const {
      return tau_;
    }
    T lambda() const {
      return lambda_;
    }
    T Q() const {
      return Q_;
    }
    T t() const {
      return t_;
    }
    T w() const {
      return w_;
    }

    uint32_t globalId;  // id globally - only increasing id
  };

  static T LambdaOffset() {  // DJC added
    return 0.0f;
  }

  static T Dist(const Matrix<T, Dynamic, 1>& a,
                const Matrix<T, Dynamic, 1>& b) {
    return (a - b).squaredNorm();
  }

  static T Dissimilarity(const Matrix<T, Dynamic, 1>& a,
                         const Matrix<T, Dynamic, 1>& b) {
    return (a - b).squaredNorm();
  }

  static bool Closer(const T a, const T b) {
    return a < b;
  }

  template<int D>
  static void ComputeCenters(
      const std::vector<Matrix<T, D, 1>,
          Eigen::aligned_allocator<Matrix<T, D, 1>>>& xs,
      const std::vector<uint32_t> zs, uint32_t K,
      std::vector<Matrix<T,D,1>, Eigen::aligned_allocator<Matrix<T,D,1>>>& mus);

  static Matrix<T,Dynamic,1> ComputeSum(
      const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t k,
      uint32_t* N_k);

  static Matrix<T,Dynamic,Dynamic> ComputeCenters(
      const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K,
      VectorXu& Ns);

  static Matrix<T,Dynamic,1> ComputeCenter(
      const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t k,
      uint32_t* N_k);

  static T DistToUninstantiated(
      const Matrix<T,Dynamic,1>& x_i, const Matrix<T,Dynamic,1>& ps_k,
      const T t_k, const T w_k, const T tau, const T Q) {
    return Dist(x_i, ps_k) / (tau * t_k + 1.0f + 1.0f / w_k) + Q * t_k;
  }

  static bool ClusterIsDead(const T t_k, const T lambda, const T Q) {
    return t_k * Q > lambda;
  }

  static Matrix<T,Dynamic,1> ReInstantiatedOldCluster(
      const Matrix<T,Dynamic,1>& xSum, const T N_k,
      const Matrix<T,Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau);

  static T UpdateWeight(
      const Matrix<T,Dynamic,1>& xSum, const uint32_t N_k,
      const Matrix<T,Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau) {
    return 1.0f / (1.0f / w_k + t_k * tau) + N_k;
  }
};

//============================= Implementation =================================

template<typename T> template<int D>
void Euclidean<T>::ComputeCenters(
    const std::vector<Matrix<T, D, 1>,
        Eigen::aligned_allocator<Matrix<T, D, 1>>>& xs,
    const std::vector<uint32_t> zs, uint32_t K,
    std::vector<Matrix<T,D,1>, Eigen::aligned_allocator<Matrix<T,D,1>>>& mus) {
  for (uint32_t k = 0; k < K; ++k) mus[k].fill(0);
  std::vector<uint32_t> Ns(K, 0);
  for (uint32_t i = 0; i < xs.size(); ++i) {
    mus[zs[i]] += xs[i];
    ++Ns[zs[i]];
  }
  for (uint32_t k = 0; k < K; ++k) mus[k] /= Ns[k];
}

template<typename T>
Matrix<T, Dynamic, 1> Euclidean<T>::ComputeSum(
    const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z, const uint32_t k,
    uint32_t* N_k) {
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  Matrix<T, Dynamic, 1> xSum(D);
  xSum.setZero(D);
  if (N_k)
    *N_k = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (z(i) == k) {
      xSum += x.col(i);
      if (N_k)
        (*N_k)++;
    }
  }
  return xSum;
}

template<typename T>
Matrix<T, Dynamic, Dynamic> Euclidean<T>::ComputeCenters(
    const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z, const uint32_t K,
    VectorXu& Ns) {
  const uint32_t D = x.rows();
  Matrix<T, Dynamic, Dynamic> centroids(D, K);
#pragma omp parallel for
  for (uint32_t k = 0; k < K; ++k) {
    centroids.col(k) = ComputeCenter(x, z, k, &Ns(k));
  }
  return centroids;
}

template<typename T>
Matrix<T, Dynamic, 1> Euclidean<T>::ComputeCenter(
    const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z, const uint32_t k,
    uint32_t* N_k) {
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  if (N_k)
    *N_k = 0;
  Matrix<T, Dynamic, 1> mean_k(D);
  mean_k.setZero(D);
  for (uint32_t i = 0; i < N; ++i) {
    if (z(i) == k) {
      mean_k += x.col(i);
      if (N_k)
        (*N_k)++;
    }
  }
  if (!N_k) {
    return mean_k;
  }
  if (*N_k > 0) {
    return mean_k / (*N_k);
  } else {
    return x.col(k);
  }
}

template<typename T>
Matrix<T, Dynamic, 1> Euclidean<T>::ReInstantiatedOldCluster(
    const Matrix<T, Dynamic, 1>& xSum, const T N_k,
    const Matrix<T, Dynamic, 1>& ps_k, const T t_k, const T w_k, const T tau) {
  const T gamma = 1.0f / (1.0f / w_k + t_k * tau);
  return (ps_k * gamma + xSum) / (gamma + N_k);
}

#endif /* EUCLIDEAN_DATA_HPP_ */
