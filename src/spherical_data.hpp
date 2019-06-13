/*
 * Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 * github.com/jstraub/dpMMlowVar/blob/master/include/dpMMlowVar/sphericalData.hpp
 */

/*
 * File modified by Dylan Campbell
 * Date of modification: 20180928
 * Nature of modifications: minor, removed dependency on jsCore and simplified
 */

#ifndef SPHERICAL_DATA_HPP_
#define SPHERICAL_DATA_HPP_

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::min;
using std::max;
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> VectorXu;

// Rotation from point A to B; percentage specifies how far the rotation will
// bring us towards B [0,1]
template<typename T>
inline Matrix<T, Dynamic, Dynamic> RotationFromAtoB(
    const Matrix<T, Dynamic, 1>& a, const Matrix<T, Dynamic, 1>& b,
    T percentage = 1.0) {
  assert(b.size() == a.size());
  uint32_t D_ = b.size();
  Matrix<T, Dynamic, Dynamic> bRa(D_, D_);

  T dot = b.transpose() * a;
  if (fabs(dot) > 1.0) {
    std::cout << "a = " << a.transpose() << " |.| " << a.norm() << " b = "
        << b.transpose() << " |.| " << b.norm() << " -> " << dot << std::endl;
    assert(false);
  }
  dot = max(static_cast<T>(-1.0), min(static_cast<T>(1.0), dot));
  // If points are almost the same, just put identity
  if (fabs(dot - 1.0f) < 1e-6f) {
    bRa = Matrix<T, Dynamic, Dynamic>::Identity(D_, D_);
    // If points are antipodal, direction does not matter, pick one and rotate
    // by percentage
  } else if (fabs(dot + 1.0f) < 1e-6f) {
    bRa = -Matrix<T, Dynamic, Dynamic>::Identity(D_, D_);
    bRa(0, 0) = cos(percentage * M_PI * 0.5);
    bRa(1, 1) = cos(percentage * M_PI * 0.5);
    bRa(0, 1) = -sin(percentage * M_PI * 0.5);
    bRa(1, 0) = sin(percentage * M_PI * 0.5);
  } else {
    T alpha = acos(dot) * percentage;
    Matrix<T, Dynamic, 1> c(D_);
    c = a - b * dot;
    if (c.norm() <= 1e-5) {
      std::cout << "c = " << c.transpose() << " |.| " << c.norm() << std::endl;
      assert(false);
    }
    c /= c.norm();
    Matrix<T, Dynamic, Dynamic> A = b * c.transpose() - c * b.transpose();
    bRa = Matrix<T, Dynamic, Dynamic>::Identity(D_, D_) + sin(alpha) * A
        + (cos(alpha) - 1.0f) * (b * b.transpose() + c * c.transpose());
  }
  return bRa;
}

template<typename T>
struct Spherical {
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
        : centroid_(xSum / xSum.norm()),
          xSum_(xSum),
          N_(N) {
    }

    T Dist(const Matrix<T, Dynamic, 1>& x_i) const {
      return Spherical::Dist(this->centroid_, x_i);
    }

    void ComputeSS(const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z,
                   const uint32_t k) {
      Spherical::ComputeSum(x, z, k, &N_);
      if (N_ == 0) {
        const uint32_t D = x.rows();
        xSum_ = Matrix<T, Dynamic, 1>::Zero(D);
        xSum_(0) = 1.;
      }
    }

    void UpdateCenter() {
      if (N_ > 0)
        centroid_ = xSum_ / xSum_.norm();
    }

    void ComputeCenter(const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z,
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
    T beta_;
    T lambda_;
    T Q_;
    Matrix<T, Dynamic, 1> prev_centroid_;

   public:
    DependentCluster()
        : Cluster(),
          t_(0),
          w_(0),
          beta_(1),
          lambda_(1),
          Q_(1),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(uint32_t D)
        : Cluster(D),
          t_(0),
          w_(0),
          beta_(1),
          lambda_(1),
          Q_(1),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const Matrix<T, Dynamic, 1>& x_i)
        : Cluster(x_i),
          t_(0),
          w_(0),
          beta_(1),
          lambda_(1),
          Q_(1),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const Matrix<T, Dynamic, 1>& x_i, T beta, T lambda, T Q)
        : Cluster(x_i),
          t_(0),
          w_(0),
          beta_(beta),
          lambda_(lambda),
          Q_(Q),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const Matrix<T, Dynamic, 1>& x_i,
                     const DependentCluster& cl0)
        : Cluster(x_i),
          t_(0),
          w_(0),
          beta_(cl0.beta()),
          lambda_(cl0.lambda()),
          Q_(cl0.Q()),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(T beta, T lambda, T Q)
        : Cluster(),
          t_(0),
          w_(0),
          beta_(beta),
          lambda_(lambda),
          Q_(Q),
          prev_centroid_(this->centroid_) {
    }

    DependentCluster(const DependentCluster& b)
        : Cluster(b.xSum(), b.N()),
          t_(b.t()),
          w_(b.w()),
          beta_(b.beta()),
          lambda_(b.lambda()),
          Q_(b.Q()),
          prev_centroid_(b.prev_centroid()) {
    }

    DependentCluster* Clone() {
      return new DependentCluster(*this);
    }

    bool IsDead() const {
      return t_ * Q_ < lambda_;
    }
    bool IsNew() const {
      return t_ == 0;
    }

    void IncAge() {
      ++t_;
    }

    void Print() const {
      std::cout << "cluster globId = " << globalId << "\tN = " << this->N_
          << "\tage = " << t_ << "\tweight = " << w_ << "\t dead? "
          << this->IsDead() << "  center: " << this->centroid().transpose()
          << std::endl;
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
      T phi, theta, eta;
      T zeta = acos(
          max(static_cast<T>(-1.0f), min(static_cast<T>(1.0f),
          Spherical::Dist(this->xSum_, this->centroid_) / this->xSum_.norm())));
      Spherical::SolveProblem2(
          this->xSum_, zeta, t_, w_, beta_, phi, theta, eta);
      w_ = (w_ == 0.0) ? this->xSum_.norm() :
          w_ * cos(theta) + beta_ * t_ * cos(phi)
          + this->xSum_.norm() * cos(eta);
      t_ = 0;
    }

    void ReInstantiate() {
      T phi, theta, eta;
      T zeta = acos(max(static_cast<T>(-1.0f), min(static_cast<T>(1.0f),
          Spherical::Dist(this->xSum_, this->prev_centroid_)
          / this->xSum_.norm())));
      Spherical::SolveProblem2(
          this->xSum_, zeta, t_, w_, beta_, phi, theta, eta);
      this->centroid_ = RotationFromAtoB<T>(
          this->xSum_ / this->xSum_.norm(), this->prev_centroid_,
          eta / (phi * t_ + theta + eta)) * this->xSum_ / this->xSum_.norm();
    }

    void ReInstantiate(const Matrix<T, Dynamic, Dynamic>& x_i) {
      this->xSum_ = x_i;
      this->N_ = 1;
      ReInstantiate();
    }

    T MaxDist() const {
      return this->lambda_ + 1.0f;
    }

    T Dist(const Matrix<T, Dynamic, 1>& x_i) const {
      if (this->IsInstantiated()) {
        return Spherical::Dist(this->centroid_, x_i);
      } else {
        T phi, theta, eta;
        T zeta = acos(
            max(static_cast<T>(-1.), min(static_cast<T>(1.0),
            Spherical::Dist(x_i, this->prev_centroid_))));
        Spherical::SolveProblem2Approx(
            x_i, zeta, t_, w_, beta_, phi, theta, eta);
        return w_ * (cos(theta) - 1.0f) + t_ * beta_ * (cos(phi) - 1.0f)
            + Q_ * t_ + cos(eta); // no minus 1 here cancels with Z(beta) from the two other assignments
      }
    }

    T beta() const {
      return beta_;
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

  static T LambdaOffset() { // DJC added
    return 1.0f;
  }

  static T Dist(const Matrix<T, Dynamic, 1>& a,
      const Matrix<T, Dynamic, 1>& b) {
    return a.transpose() * b;
  }

  static T Dissimilarity(const Matrix<T, Dynamic, 1>& a,
      const Matrix<T, Dynamic, 1>& b) {
    return acos(
        min(static_cast<T>(1.0), max(static_cast<T>(-1.0),
        (a.transpose() * b)(0))));
  }

  static bool Closer(const T a, const T b) {
    return a > b;
  }

  template<int D>
  static void ComputeCenters(
      const std::vector<Eigen::Matrix<T, D, 1>,
          Eigen::aligned_allocator<Eigen::Matrix<T, D, 1> > >& xs,
      const std::vector<uint32_t> zs, uint32_t K,
      std::vector<Eigen::Matrix<T, D, 1>,
          Eigen::aligned_allocator<Eigen::Matrix<T, D, 1> > >& mus);
 private:
  static void SolveProblem1(T gamma, T age, const T beta, T& phi, T& theta);
  static void SolveProblem2(
      const Matrix<T, Dynamic, 1>& xSum, T zeta, T age, T w, const T beta,
      T& phi, T& theta, T& eta);
  static void SolveProblem1Approx(
      T gamma, T age, const T beta, T& phi, T& theta);
  static void SolveProblem2Approx(
      const Matrix<T, Dynamic, 1>& xSum, T zeta, T age, T w, const T beta,
      T& phi, T& theta, T& eta);
  static Matrix<T, Dynamic, 1> ComputeSum(
      const Matrix<T, Dynamic, Dynamic>& x, const VectorXu& z, const uint32_t k,
      uint32_t* N_k);
};

// ================================ impl ======================================

template<typename T> template<int D>
void Spherical<T>::ComputeCenters(
    const std::vector<Eigen::Matrix<T, D, 1>,
        Eigen::aligned_allocator<Eigen::Matrix<T, D, 1> > >& xs,
    const std::vector<uint32_t> zs, uint32_t K,
    std::vector<Eigen::Matrix<T, D, 1>,
        Eigen::aligned_allocator<Eigen::Matrix<T, D, 1> > >& mus) {
  for (uint32_t k = 0; k < K; ++k)
    mus[k].fill(0);
  for (uint32_t i = 0; i < xs.size(); ++i) {
    mus[zs[i]] += xs[i];
  }
  for (uint32_t k = 0; k < K; ++k) {
    mus[k] /= mus[k].norm();
  }
}

template<typename T>
Matrix<T, Dynamic, 1> Spherical<T>::ComputeSum(
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

template<class T>
void Spherical<T>::SolveProblem1(
    T gamma, T age, const T beta, T& phi, T& theta) {
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0;
  for (uint32_t i = 0; i < 10; ++i) {
    T sinPhi = sin(phi);
    T f = -gamma + age * phi + asin(beta * sinPhi);
    T df = age + (beta * cos(phi)) / sqrt(1.0f - beta * beta * sinPhi * sinPhi);
    T dPhi = f / df;
    phi = phi - dPhi;  // Newton iteration
    if (fabs(dPhi) < 1e-6)
      break;
  }
  theta = asin(beta * sin(phi));
}

template<class T>
void Spherical<T>::SolveProblem2(
    const Matrix<T, Dynamic, 1>& xSum, T zeta, T age, T w, const T beta, T& phi,
    T& theta, T& eta) {
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta)
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  phi = 0.0;
  T L2xSum = xSum.norm();
  for (uint32_t i = 0; i < 10; ++i) {
    T sinPhi = sin(phi);
    T cosPhi = cos(phi);
    T f = -zeta + asin(beta / L2xSum * sinPhi) + age * phi
        + asin(beta / w * sinPhi);
    T df = age + (beta * cosPhi)
        / sqrt(L2xSum * L2xSum - beta * beta * sinPhi * sinPhi)
        + (beta * cosPhi) / sqrt(w * w - beta * beta * sinPhi * sinPhi);
    T dPhi = f / df;
    phi = phi - dPhi;  // Newton iteration
    if (fabs(dPhi) < 1e-6)
      break;
  }
  theta = asin(beta / w * sin(phi));
  eta = asin(beta / L2xSum * sin(phi));
}

template<class T>
void Spherical<T>::SolveProblem1Approx(
    T gamma, T age, const T beta, T& phi, T& theta) {
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0;
  for (uint32_t i = 0; i < 10; ++i) {
    T sinPhi = phi;
    T cosPhi = 1.0f;
    T f = -gamma + age * phi + asin(beta * sinPhi);
    T df = age + (beta * cosPhi) / sqrt(1.0f - beta * beta * sinPhi * sinPhi);
    T dPhi = f / df;
    phi = phi - dPhi;  // Newton iteration
    if (fabs(dPhi) < 1e-6)
      break;
  }
  theta = asin(beta * sin(phi));
}

template<class T>
void Spherical<T>::SolveProblem2Approx(
    const Matrix<T, Dynamic, 1>& xSum, T zeta, T age, T w, const T beta, T& phi,
    T& theta, T& eta) {
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta)
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  phi = zeta / (beta * (1.0f + 1.0f / w) + age);
  theta = zeta / (1.0f + w * (1.0f + age / beta));
  eta = zeta / (1.0f + 1.0f / w + age / beta);
}

#endif /* SPHERICAL_HPP_ */
