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

#ifndef SPHERICAL_MM_L2_DISTANCE_HPP_
#define SPHERICAL_MM_L2_DISTANCE_HPP_

#include "CppNumericalSolvers/include/cppoptlib/meta.h"
#include "CppNumericalSolvers/include/cppoptlib/problem.h"
#include "CppNumericalSolvers/include/cppoptlib/solver/lbfgssolver.h"

template<typename T>
class SphericalMML2Distance : public cppoptlib::Problem<T, 6> {
 public:
  using Matrix3T = typename Eigen::Matrix<T, 3, 3>;
  using MatrixX3T = typename Eigen::Matrix<T, Eigen::Dynamic, 3>;
  using Vector6T = typename Eigen::Matrix<T, 6, 1>;
  using VectorXT = typename Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using RowVector3T = typename Eigen::Matrix<T, 1, 3>;

  SphericalMML2Distance() {
  }

  SphericalMML2Distance(const MatrixX3T& mus_2d, const MatrixX3T& mus_3d,
                        const VectorXT& kappas_2d, const VectorXT& variances_3d,
                        const VectorXT& log_phis_3d,
                        const VectorXT& log_phis_on_z_kappas_2d,
                        const std::vector<int>& class_num_components_2d,
                        const std::vector<int>& class_num_components_3d,
                        const std::vector<T>& log_class_weights)
      : mus_2d_(mus_2d),
        mus_3d_(mus_3d),
        kappas_2d_(kappas_2d),
        variances_3d_(variances_3d),
        log_phis_3d_(log_phis_3d),
        log_phis_on_z_kappas_2d_(log_phis_on_z_kappas_2d),
        class_num_components_2d_(class_num_components_2d),
        class_num_components_3d_(class_num_components_3d),
        log_class_weights_(log_class_weights) {
  }

  T value(const Vector6T &x) {
    RowVector3T translation_vector = x.head(3).transpose();
    RowVector3T rotation_vector = x.tail(3).transpose();

    // Transform mus
    MatrixX3T mus_3d_translated = mus_3d_.rowwise() - translation_vector;
    VectorXT mus_3d_translated_norm2 =
        mus_3d_translated.rowwise().squaredNorm();
    MatrixX3T mus_3d_translated_normalised = mus_3d_translated.array().colwise()
        / mus_3d_translated_norm2.array().sqrt();
    MatrixX3T mus_2d_rotated = mus_2d_
        * RotationVectorToMatrix(rotation_vector); // Not transpose - (R^-1)mu2d

    VectorXT kappas_3d = (mus_3d_translated_norm2.array()
        / variances_3d_.array() + 1).matrix();
    VectorXT log_phis_on_z_kappas_3d = log_phis_3d_;
    for (int i = 0; i < log_phis_on_z_kappas_3d.rows(); ++i) {
      log_phis_on_z_kappas_3d[i] -= LogZFunction(kappas_3d[i]);
    }
    MatrixX3T kappas_mus_2d_rotated = mus_2d_rotated.array().colwise()
        * kappas_2d_.array();
    MatrixX3T kappas_mus_3d_translated_normalised =
        mus_3d_translated_normalised.array().colwise() * kappas_3d.array();

    int start_index_2d = 0;
    int start_index_3d = 0;
    int stop_index_2d = 0;
    int stop_index_3d = 0;
    T function_value_11 = 0.0f;
    T function_value_12 = 0.0f;
    for (int c = 0; c < class_num_components_2d_.size(); ++c) {
      stop_index_2d += class_num_components_2d_[c];
      stop_index_3d += class_num_components_3d_[c];
      for (int i = start_index_3d; i < stop_index_3d; ++i) {
        for (int j = start_index_3d; j < stop_index_3d; ++j) {
          T K1i1j = (kappas_mus_3d_translated_normalised.row(i)
              + kappas_mus_3d_translated_normalised.row(j)).norm();
          function_value_11 += exp(
              log_class_weights_[c] + log_phis_on_z_kappas_3d[i]
                  + log_phis_on_z_kappas_3d[j] + LogZFunction(K1i1j));
        }
        for (int j = start_index_2d; j < stop_index_2d; ++j) {
          T K1i2j = (kappas_mus_3d_translated_normalised.row(i)
              + kappas_mus_2d_rotated.row(j)).norm();
          function_value_12 += exp(
              log_class_weights_[c] + log_phis_on_z_kappas_3d[i]
                  + log_phis_on_z_kappas_2d_[j] + LogZFunction(K1i2j));
        }
      }
      start_index_2d = stop_index_2d;
      start_index_3d = stop_index_3d;
    }
    return function_value_11 - 2.0f * function_value_12;
  }

  void gradient(const Vector6T &x, Vector6T &grad) {
    RowVector3T translation_vector = x.head(3).transpose();
    RowVector3T rotation_vector = x.tail(3).transpose();

    // Transform mus
    MatrixX3T mus_3d_translated = mus_3d_.rowwise() - translation_vector;
    VectorXT mus_3d_translated_norm2 =
        mus_3d_translated.rowwise().squaredNorm();
    VectorXT mus_3d_translated_norm =
        mus_3d_translated_norm2.array().sqrt().matrix();
    MatrixX3T mus_3d_translated_normalised = mus_3d_translated.array().colwise()
        / mus_3d_translated_norm.array();
    Matrix3T rotation_matrix = RotationVectorToMatrix(rotation_vector);
    MatrixX3T mus_2d_rotated = mus_2d_ * rotation_matrix; // Not transpose

    VectorXT kappas_3d = (mus_3d_translated_norm2.array()
        / variances_3d_.array() + 1.0f).matrix();
    VectorXT log_phis_on_z_kappas_3d = log_phis_3d_;
    for (int i = 0; i < log_phis_on_z_kappas_3d.rows(); ++i) {
      log_phis_on_z_kappas_3d[i] -= LogZFunction(kappas_3d[i]);
    }
    MatrixX3T kappas_mus_2d_rotated = mus_2d_rotated.array().colwise()
        * kappas_2d_.array();
    MatrixX3T kappas_mus_3d_translated_normalised =
        mus_3d_translated_normalised.array().colwise() * kappas_3d.array();

    MatrixX3T dkappas1dt = mus_3d_translated_normalised.array().colwise()
        * (-2.0f * (kappas_3d.array() - 1.0f) / mus_3d_translated_norm.array());
    Matrix3T rotation_component;
    if (rotation_vector.norm() <= std::numeric_limits<float>::epsilon()) {
      rotation_component = Matrix3T::Identity();
    } else {
      rotation_component = (rotation_vector.transpose() * rotation_vector
          + (rotation_matrix.transpose() - Matrix3T::Identity())
          * ToSkewSymmetric(rotation_vector))
          / rotation_vector.squaredNorm();
    }

    int start_index_2d = 0;
    int start_index_3d = 0;
    int stop_index_2d = 0;
    int stop_index_3d = 0;
    RowVector3T df11dt = RowVector3T::Zero();
    RowVector3T df12dt = RowVector3T::Zero();
    RowVector3T df12dr = RowVector3T::Zero();
    for (int c = 0; c < class_num_components_2d_.size(); ++c) {
      stop_index_2d += class_num_components_2d_[c];
      stop_index_3d += class_num_components_3d_[c];
      for (int i = start_index_3d; i < stop_index_3d; ++i) {
        for (int j = start_index_3d; j < stop_index_3d; ++j) {
          RowVector3T k1i1j = kappas_mus_3d_translated_normalised.row(i)
              + kappas_mus_3d_translated_normalised.row(j);
          T K1i1j = k1i1j.norm();
          T f1i1j = exp(
              log_class_weights_[c] + log_phis_on_z_kappas_3d[i]
                  + log_phis_on_z_kappas_3d[j] + LogZFunction(K1i1j));

          RowVector3T dK1i1jdt = k1i1j / K1i1j
              * (((2.0f - kappas_3d[i])
                  * mus_3d_translated_normalised.row(i).transpose()
                  * mus_3d_translated_normalised.row(i)
                  - kappas_3d[i] * Matrix3T::Identity())
                  / mus_3d_translated_norm[i]
                  + ((2.0f - kappas_3d[j])
                      * mus_3d_translated_normalised.row(j).transpose()
                      * mus_3d_translated_normalised.row(j)
                      - kappas_3d[j] * Matrix3T::Identity())
                      / mus_3d_translated_norm[j]);
          df11dt += f1i1j
              * (dZOnZ(K1i1j) * dK1i1jdt
                  - dZOnZ(kappas_3d[i]) * dkappas1dt.row(i)
                  - dZOnZ(kappas_3d[j]) * dkappas1dt.row(j));
        }
        for (int j = start_index_2d; j < stop_index_2d; ++j) {
          RowVector3T k1i2j = kappas_mus_3d_translated_normalised.row(i)
              + kappas_mus_2d_rotated.row(j);
          T K1i2j = k1i2j.norm();
          T f1i2j = exp(
              log_class_weights_[c] + log_phis_on_z_kappas_3d[i]
                  + log_phis_on_z_kappas_2d_[j] + LogZFunction(K1i2j));

          RowVector3T dK1i2jdt = k1i2j / K1i2j
              * (((2.0f - kappas_3d[i])
                  * mus_3d_translated_normalised.row(i).transpose()
                  * mus_3d_translated_normalised.row(i)
                  - kappas_3d[i] * Matrix3T::Identity())
                  / mus_3d_translated_norm[i]);
          RowVector3T dK1i2jdr = -kappas_3d[i] * k1i2j / K1i2j
              * ToSkewSymmetric(mus_3d_translated_normalised.row(i))
              * rotation_component;
          df12dt += f1i2j
              * (dZOnZ(K1i2j) * dK1i2jdt
                  - dZOnZ(kappas_3d[i]) * dkappas1dt.row(i));
          df12dr += f1i2j * (dZOnZ(K1i2j) * dK1i2jdr);
        }
      }
      start_index_2d = stop_index_2d;
      start_index_3d = stop_index_3d;
    }
    grad.head(3) = (df11dt - 2.0f * df12dt).transpose();
    grad.tail(3) = (-2.0f * df12dr).transpose();
  }

 private:
  Matrix3T RotationVectorToMatrix(const RowVector3T& rotation_vector) {
    T rotation_angle = rotation_vector.norm();
    if (rotation_angle <= std::numeric_limits<float>::epsilon()) {
      return Matrix3T::Identity();
    } else {
      T v[3] = { rotation_vector[0] / rotation_angle, rotation_vector[1]
          / rotation_angle, rotation_vector[2] / rotation_angle };
      T ca = std::cos(rotation_angle);
      T ca2 = 1 - ca;
      T sa = std::sin(rotation_angle);
      T v0sa = v[0] * sa;
      T v1sa = v[1] * sa;
      T v2sa = v[2] * sa;
      T v0v1ca2 = v[0] * v[1] * ca2;
      T v0v2ca2 = v[0] * v[2] * ca2;
      T v1v2ca2 = v[1] * v[2] * ca2;
      Matrix3T rotation_matrix;
      rotation_matrix(0, 0) = ca + v[0] * v[0] * ca2;
      rotation_matrix(0, 1) = v0v1ca2 - v2sa;
      rotation_matrix(0, 2) = v0v2ca2 + v1sa;
      rotation_matrix(1, 0) = v0v1ca2 + v2sa;
      rotation_matrix(1, 1) = ca + v[1] * v[1] * ca2;
      rotation_matrix(1, 2) = v1v2ca2 - v0sa;
      rotation_matrix(2, 0) = v0v2ca2 - v1sa;
      rotation_matrix(2, 1) = v1v2ca2 + v0sa;
      rotation_matrix(2, 2) = ca + v[2] * v[2] * ca2;
      return rotation_matrix;
    }
  }

  T LogZFunction(T x) {
    if (fabs(x) < 1.0e-3f) {
      return log(2.0f);
    } else if (x < 10.0f) {
      return log(exp(x) - exp(-x)) - log(x);
    } else {  // equivalent for x >= 8 (float) or x >= 17 (double)
      return x - log(x);
    }
  }

  T dZOnZ(T x) {
    return 2.0f / (1 - exp(-2.0f * x)) - 1.0f / x - 1.0f;
  }

  Matrix3T ToSkewSymmetric(const RowVector3T& x) {
    return (Matrix3T() << 0, -x[2], x[1], x[2], 0, -x[0], -x[1], x[0], 0)
        .finished();
  }

  std::vector<int> class_num_components_2d_;
  std::vector<int> class_num_components_3d_;
  std::vector<T> log_class_weights_;
  MatrixX3T mus_2d_;
  MatrixX3T mus_3d_;
  VectorXT kappas_2d_;
  VectorXT variances_3d_;
  VectorXT log_phis_3d_;
  VectorXT log_phis_on_z_kappas_2d_;
};

#endif /* SPHERICAL_MM_L2_DISTANCE_HPP_ */
