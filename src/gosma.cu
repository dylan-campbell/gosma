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

#include "gosma.h"

//#define USE_DEBUG_OUTPUT // Uncomment if debug output desired

#ifdef USE_DEBUG_OUTPUT
  #define DEBUG_OUTPUT(x) do {std::cerr << x << std::endl;} while (0)
#else
  #define DEBUG_OUTPUT(x) do {} while (0)
#endif

#define CudaErrorCheck(ans) {__CudaErrorCheck((ans), __FILE__, __LINE__);}

void __CudaErrorCheck(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    std::cout << "CUDA Error (" << file << ":" << line << "): "
        << cudaGetErrorString(code) << std::endl;
    exit(code);
  }
}

const int GOSMA::kMaxDuration = 600;
const int GOSMA::kMaxQueueSize = 2e8;  // ~10GB
const int GOSMA::kNumDevices = 1;
const int GOSMA::kNumConcurrentNodes = 6144;
const int GOSMA::kNumThreadsPerBlock = 32;
const int GOSMA::kNumChildrenPerNode = 8;
const int GOSMA::kNumConcurrentThreads = kNumConcurrentNodes
    * kNumChildrenPerNode;
const int GOSMA::kNumBlocks = (kNumConcurrentThreads + kNumThreadsPerBlock - 1)
    / kNumThreadsPerBlock;  // Rounded up
const int GOSMA::kMaxNumComponents = 1024;
const int GOSMA::kMaxNumClasses = 32;
const float GOSMA::kPi = 3.14159265358979323846f;
const float GOSMA::kSquareRootThree = 1.732050808f;
const float GOSMA::kInvertedGoldenRatio = 0.618033988749895f;  // 0.5f * (sqrt(5) - 1)
const float GOSMA::kOneMinusInvertedGoldenRatio = 0.381966011250105f;  // 0.5 * (3 - sqrt(5))
const float GOSMA::kToleranceSquared = 0.000002353097057602252f;  // (pi / 2048)^2 [at most 0.088 degrees incorrect]

// Declare constant memory (64KB maximum)
// Size: 2 + 2*2*kMaxNumClasses + 4*4 + 4*kMaxNumComponents(2*3 + 4)
// = 18 + 4*kMaxNumClasses + 40*kMaxNumComponents = 41858B / 41.858KB
__constant__ int c_num_components_2d;
__constant__ int c_num_components_3d;
__constant__ int c_num_classes;
__constant__ int c_class_num_components_2d[GOSMA::kMaxNumClasses];
__constant__ int c_class_num_components_3d[GOSMA::kMaxNumClasses];
__constant__ float c_log_class_weights[GOSMA::kMaxNumClasses];
__constant__ float c_zeta2;
__constant__ float c_l2_normaliser;
__constant__ float c_min_linear_resolution;
__constant__ float c_min_angular_resolution;
__constant__ float c_mus_2d[GOSMA::kMaxNumComponents][3];
__constant__ float c_mus_3d[GOSMA::kMaxNumComponents][3];
__constant__ float c_kappas_2d[GOSMA::kMaxNumComponents];
__constant__ float c_variances_3d[GOSMA::kMaxNumComponents];
__constant__ float c_log_phis_on_z_kappas_2d[GOSMA::kMaxNumComponents];
__constant__ float c_log_phis_3d[GOSMA::kMaxNumComponents];

__device__ void RotationVectorToMatrix(
    const float (&rotation_vector)[3], float (&rotation_matrix)[3][3]);
__device__ float ComputeTranslationUncertaintyAngle(
    const float (&mu_3d_translated_normalised)[3],
    const float (&min_mu_3d_translated)[3],
    const float (&max_mu_3d_translated)[3], float &max_mu_3d_translated_norm2);
__device__ float ComputeRotationUncertaintyAngle(
    const float (&mu_2d)[3], const float (&mu_2d_rotated)[3],
    const float (&min_rotation_vector)[3],
    const float (&max_rotation_vector)[3]);
__device__ float RotateAndDot(
    const float (&mu_2d_centre_rotated)[3], const float (&mu_2d)[3],
    const float (&rotation_vector)[3]);
__device__ bool InvalidTranslationSubcube(
    const float (&min_mu_3d_translated)[3],
    const float (&max_mu_3d_translated)[3], const float zeta2);
__device__ bool InvalidRotationSubcube(
    const float (&min_rotation_vector)[3],
    const float (&max_rotation_vector)[3]);
__device__ bool CuboidInsideSphere(
    const float (&cuboid_min)[3], const float (&cuboid_max)[3],
    const float radius_squared);
__device__ bool CuboidOutsideSphere(
    const float (&cuboid_min)[3], const float (&cuboid_max)[3],
    const float radius_squared);
__device__ float MinSquaredDistanceToCuboid(
    const float (&cuboid_min)[3], const float (&cuboid_max)[3]);
__device__ float LogZFunction(const float x);
__device__ bool ComputeGradientSign(
    const float kappa_3d, const float K1i2j, const float kappa_2d_cos_theta);
__device__ float ComputeGradient(
    const float kappa_3d, const float K1i2j, const float kappa_2d_cos_theta);
__device__ float ComputeKstar(
    const float kappa_lb_3d, const float kappa_ub_3d, const float kappa2_2d,
    const float kappa_2d_cos_theta, float& kappa_3d_star);

/*
 * GetBounds
 * Branch parent node and calculate the lower and upper bounds for the child node
 * Inputs:
 * - d_parent_nodes_: array of parent nodes
 * Outputs:
 * - d_nodes_: array of child nodes
 */
__global__ void GetBounds(Node* d_parent_nodes_, Node* d_nodes_) {

  // Calculate the thread global_index: [node_index: 9 bits, child_index: 3 bits]
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  int child_index = global_index & 7;
  int node_index = global_index >> 3;

  float zeta2 = c_zeta2;

  /*
   * Branching
   */
  Node parent_node = d_parent_nodes_[node_index];
  Node node;
  if (parent_node.branch == 1) {  // Branch over translation
    node.twx = 0.5f * parent_node.twx;
    node.twy = 0.5f * parent_node.twy;
    node.twz = 0.5f * parent_node.twz;
    node.tx = parent_node.tx + (child_index & 1) * node.twx;
    node.ty = parent_node.ty + ((child_index >> 1) & 1) * node.twy;
    node.tz = parent_node.tz + ((child_index >> 2) & 1) * node.twz;
    node.rw = parent_node.rw;
    node.rx = parent_node.rx;
    node.ry = parent_node.ry;
    node.rz = parent_node.rz;
    node.lb = 0.0f;
    node.ub = 0.0f;
    node.branch = 2;
  } else if (parent_node.branch == 2) {  // Branch over rotation
    node.twx = parent_node.twx;
    node.twy = parent_node.twy;
    node.twz = parent_node.twz;
    node.tx = parent_node.tx;
    node.ty = parent_node.ty;
    node.tz = parent_node.tz;
    node.rw = 0.5f * parent_node.rw;
    node.rx = parent_node.rx + (child_index & 1) * node.rw;
    node.ry = parent_node.ry + ((child_index >> 1) & 1) * node.rw;
    node.rz = parent_node.rz + ((child_index >> 2) & 1) * node.rw;
    node.lb = 0.0f;
    node.ub = 0.0f;
    node.branch = 1;
  } else {  // branch == 0
    node.lb = 0.0f;
    node.ub = 0.0f;
    node.branch = 0;  // Set branch flag to discard
    d_nodes_[global_index] = node;
    return;
  }

  float tx0 = node.tx + 0.5f * node.twx;
  float ty0 = node.ty + 0.5f * node.twy;
  float tz0 = node.tz + 0.5f * node.twz;
  float rx0 = node.rx + 0.5f * node.rw;
  float ry0 = node.ry + 0.5f * node.rw;
  float rz0 = node.rz + 0.5f * node.rw;

  float translation_uncertainty_radius = 0.5f
      * std::sqrt(
          node.twx * node.twx + node.twy * node.twy + node.twz * node.twz);
  float rotation_uncertainty_radius = 0.5f * GOSMA::kSquareRootThree * node.rw;

  // If translation cuboid or rotation cube is less than the minimum size, discard subcube
  if (translation_uncertainty_radius <= c_min_linear_resolution
      || rotation_uncertainty_radius <= c_min_angular_resolution) {
    node.branch = 0;  // Set branch flag to discard
    d_nodes_[global_index] = node;
    return;
  }

  // Skip if rotation subcube is entirely outside the rotation pi-ball
  float min_rotation_vector[3] = { node.rx, node.ry, node.rz };
  float max_rotation_vector[3] = { node.rx + node.rw, node.ry + node.rw, node.rz
      + node.rw };
  if (InvalidRotationSubcube(min_rotation_vector, max_rotation_vector)) {
    node.branch = 0;  // Set branch flag to discard
    d_nodes_[global_index] = node;
    return;
  }

  /*
   * Bounding
   */
  /*
   * Translation Uncertainty Angles
   */
  float kappas_3d[GOSMA::kMaxNumComponents];
  float kappas_lb_3d[GOSMA::kMaxNumComponents];
  float kappas_ub_3d[GOSMA::kMaxNumComponents];
  float log_phis_on_z_kappas_3d[GOSMA::kMaxNumComponents];
  float log_phis_on_z_kappas_lb_3d[GOSMA::kMaxNumComponents];
  float log_phis_on_z_kappas_ub_3d[GOSMA::kMaxNumComponents];
  float mus_3d_translated_normalised[GOSMA::kMaxNumComponents][3];
  float max_translation_uncertainty_angle = 0.0f;
  float translation_uncertainty_angles[GOSMA::kMaxNumComponents];
  for (int i = 0; i < c_num_components_3d; ++i) {  // For every mu3d
    float mu_3d[3] = { c_mus_3d[i][0], c_mus_3d[i][1], c_mus_3d[i][2] };
    float min_mu_3d_translated[3] = { mu_3d[0] - node.tx - node.twx,
        mu_3d[1] - node.ty - node.twy, mu_3d[2] - node.tz - node.twz };
    float max_mu_3d_translated[3] = { mu_3d[0] - node.tx, mu_3d[1] - node.ty,
        mu_3d[2] - node.tz };

    // Skip if a translated mu3d cuboid is entirely less than zeta from the origin
    if (InvalidTranslationSubcube(
        min_mu_3d_translated, max_mu_3d_translated, zeta2)) {
      node.branch = 0;  // Set branch flag to discard
      d_nodes_[global_index] = node;
      return;
    }

    // Translate mus3d and normalise
    mus_3d_translated_normalised[i][0] = mu_3d[0] - tx0;
    mus_3d_translated_normalised[i][1] = mu_3d[1] - ty0;
    mus_3d_translated_normalised[i][2] = mu_3d[2] - tz0;
    float norm2 = mus_3d_translated_normalised[i][0]
        * mus_3d_translated_normalised[i][0]
        + mus_3d_translated_normalised[i][1]
        * mus_3d_translated_normalised[i][1]
        + mus_3d_translated_normalised[i][2]
        * mus_3d_translated_normalised[i][2];
    float norm = std::sqrt(norm2);
    for (int j = 0; j < 3; ++j) {
      mus_3d_translated_normalised[i][j] /= norm;
    }

    // Compute translation uncertainty angles
    float max_mu_3d_translated_norm2 = 0.0f;
    float cos_translation_uncertainty_angle =
        ComputeTranslationUncertaintyAngle(
            mus_3d_translated_normalised[i], min_mu_3d_translated,
            max_mu_3d_translated, max_mu_3d_translated_norm2);
    if (cos_translation_uncertainty_angle <= -1.0f)
      cos_translation_uncertainty_angle = -1.0f;
    translation_uncertainty_angles[i] = std::acos(
        cos_translation_uncertainty_angle);
    if (translation_uncertainty_angles[i] > max_translation_uncertainty_angle)
      max_translation_uncertainty_angle = translation_uncertainty_angles[i];

    // Compute and store components needed to compute lower and upper function bound
    kappas_3d[i] = norm2 / c_variances_3d[i] + 1.0f;
    kappas_lb_3d[i] = MinSquaredDistanceToCuboid(
        min_mu_3d_translated, max_mu_3d_translated)
        / c_variances_3d[i] + 1.0f;
    kappas_ub_3d[i] = max_mu_3d_translated_norm2 / c_variances_3d[i] + 1.0f;
    float log_phi_3d = c_log_phis_3d[i];
    log_phis_on_z_kappas_3d[i] = log_phi_3d - LogZFunction(kappas_3d[i]);
    log_phis_on_z_kappas_lb_3d[i] = log_phi_3d - LogZFunction(kappas_lb_3d[i]);
    log_phis_on_z_kappas_ub_3d[i] = log_phi_3d - LogZFunction(kappas_ub_3d[i]);
  }

  /*
   * Rotation Uncertainty Angles
   */
  // Convert angle-axis rotation into a rotation matrix
  float rotation_vector[3] = { rx0, ry0, rz0 };
  float R[3][3];
  RotationVectorToMatrix(rotation_vector, R);
  float kappas_2d[GOSMA::kMaxNumComponents];
  float log_phis_on_z_kappas_2d[GOSMA::kMaxNumComponents];
  float mus_2d_rotated[GOSMA::kMaxNumComponents][3];
  float rotation_uncertainty_angles[GOSMA::kMaxNumComponents];
  float weak_rotation_uncertainty_angle = rotation_uncertainty_radius;
  float max_rotation_uncertainty_angle = 0.0f;
  for (int i = 0; i < c_num_components_2d; ++i) {  // For every mu2d
    // Rotate mu2d by R^-1
    float mu_2d[3] = { c_mus_2d[i][0], c_mus_2d[i][1], c_mus_2d[i][2] };
    mus_2d_rotated[i][0] = R[0][0] * mu_2d[0] + R[1][0] * mu_2d[1]
        + R[2][0] * mu_2d[2];
    mus_2d_rotated[i][1] = R[0][1] * mu_2d[0] + R[1][1] * mu_2d[1]
        + R[2][1] * mu_2d[2];
    mus_2d_rotated[i][2] = R[0][2] * mu_2d[0] + R[1][2] * mu_2d[1]
        + R[2][2] * mu_2d[2];

    // Calculate rotation uncertainty angle for this bearing vector
    float cos_rotation_uncertainty_angle = ComputeRotationUncertaintyAngle(
        mu_2d, mus_2d_rotated[i], min_rotation_vector, max_rotation_vector);
    if (cos_rotation_uncertainty_angle <= -1.0f)
      cos_rotation_uncertainty_angle = -1.0f;
    rotation_uncertainty_angles[i] = std::acos(cos_rotation_uncertainty_angle);
    if (rotation_uncertainty_angles[i] > weak_rotation_uncertainty_angle)
      rotation_uncertainty_angles[i] = weak_rotation_uncertainty_angle;
    if (rotation_uncertainty_angles[i] > max_rotation_uncertainty_angle)
      max_rotation_uncertainty_angle = rotation_uncertainty_angles[i];

    // Compute and store components needed to compute lower and upper function bound
    kappas_2d[i] = c_kappas_2d[i];
    log_phis_on_z_kappas_2d[i] = c_log_phis_on_z_kappas_2d[i];
  }

  /*
   * Compute Bounds
   */
  int start_index_2d = 0;
  int start_index_3d = 0;
  int stop_index_2d = 0;
  int stop_index_3d = 0;
  float upper_bound_11 = 0.0f;
  float upper_bound_12 = 0.0f;
  float lower_bound_11 = 0.0f;
  float lower_bound_12 = 0.0f;
  for (int c = 0; c < c_num_classes; ++c) {
    stop_index_2d += c_class_num_components_2d[c];
    stop_index_3d += c_class_num_components_3d[c];
    for (int i = start_index_3d; i < stop_index_3d; ++i) {
      for (int j = start_index_3d; j < stop_index_3d; ++j) {
        // Upper Bound Calculations:
        float cos_angle = mus_3d_translated_normalised[i][0]
            * mus_3d_translated_normalised[j][0]
            + mus_3d_translated_normalised[i][1]
            * mus_3d_translated_normalised[j][1]
            + mus_3d_translated_normalised[i][2]
            * mus_3d_translated_normalised[j][2];
        if (cos_angle < -1.0f)
          cos_angle = -1.0f;
        if (cos_angle > 1.0f)
          cos_angle = 1.0f;
        float K1i1j = std::sqrt(
            kappas_3d[i] * kappas_3d[i] + kappas_3d[j] * kappas_3d[j]
                + 2.0f * kappas_3d[i] * kappas_3d[j] * cos_angle);
        upper_bound_11 += exp(
            c_log_class_weights[c] + log_phis_on_z_kappas_3d[i]
                + log_phis_on_z_kappas_3d[j] + LogZFunction(K1i1j));
        // Lower Bound Calculations:
        float K1i1j_star;
        float log_phi_on_z_kappa_3d_i_star;
        float log_phi_on_z_kappa_3d_j_star;
        float angle_plus_sum_translation_uncertainty_angles = std::acos(
            cos_angle) + translation_uncertainty_angles[i]
            + translation_uncertainty_angles[j];
        // If angle sum is >= 180 degrees
        if (angle_plus_sum_translation_uncertainty_angles >= GOSMA::kPi) {
          // Z(K)/Z(kappa_i)/Z(kappa_j) is minimised by kappa_ub for angle = 180 degrees
          K1i1j_star = abs(kappas_ub_3d[i] - kappas_ub_3d[j]);
          log_phi_on_z_kappa_3d_i_star = log_phis_on_z_kappas_ub_3d[i];
          log_phi_on_z_kappa_3d_j_star = log_phis_on_z_kappas_ub_3d[j];
        } else {
          float cos_theta = std::cos(
              angle_plus_sum_translation_uncertainty_angles);
          if (angle_plus_sum_translation_uncertainty_angles
              >= 0.5f * GOSMA::kPi) {  // If angle is [90, 180) degrees
            // Z(K)/Z(kappa_i)/Z(kappa_j) is minimised by kappa_ub for angle in [90, 180) degrees
            K1i1j_star = std::sqrt(
                kappas_ub_3d[i] * kappas_ub_3d[i]
                    + kappas_ub_3d[j] * kappas_ub_3d[j]
                    + 2.0f * kappas_ub_3d[i] * kappas_ub_3d[j] * cos_theta);
            log_phi_on_z_kappa_3d_i_star = log_phis_on_z_kappas_ub_3d[i];
            log_phi_on_z_kappa_3d_j_star = log_phis_on_z_kappas_ub_3d[j];
          } else {  // If angle is [0, 90) degrees
            // Z(K)/Z(kappa_i)/Z(kappa_j) is minimised by kappa_lb or kappa_ub
            // (concave minimisation) for angle in (0, 90) degrees
            float K1i1j_lb_lb = std::sqrt(
                kappas_lb_3d[i] * kappas_lb_3d[i]
                    + kappas_lb_3d[j] * kappas_lb_3d[j]
                    + 2.0f * kappas_lb_3d[i] * kappas_lb_3d[j] * cos_theta);
            // Investigate kappa_i:
            float kappa_i_star;
            // If gradient is negative at kappa_lb, use kappa_ub
            if (!ComputeGradientSign(
                kappas_lb_3d[i], K1i1j_lb_lb, kappas_lb_3d[j] * cos_theta)) {
              kappa_i_star = kappas_ub_3d[i];
              log_phi_on_z_kappa_3d_i_star = log_phis_on_z_kappas_ub_3d[i];
            } else {
              // Gradient positive at kappa_lb, investigate kappa_i further
              float K1i1j_ub_lb = std::sqrt(
                  kappas_ub_3d[i] * kappas_ub_3d[i]
                      + kappas_lb_3d[j] * kappas_lb_3d[j]
                      + 2.0f * kappas_ub_3d[i] * kappas_lb_3d[j] * cos_theta);
              // If gradient is positive at kappa_ub, use kappa_lb
              if (ComputeGradientSign(
                  kappas_ub_3d[i], K1i1j_ub_lb, kappas_lb_3d[j] * cos_theta)) {
                kappa_i_star = kappas_lb_3d[i];
                log_phi_on_z_kappa_3d_i_star = log_phis_on_z_kappas_lb_3d[i];
              } else {
                // If gradient is positive at kappa_lb and negative at kappa_ub,
                // minimiser is either kappa_lb or kappa_ub
                if (log_phis_on_z_kappas_ub_3d[i] + LogZFunction(K1i1j_ub_lb)
                    <= log_phis_on_z_kappas_lb_3d[i]
                        + LogZFunction(K1i1j_lb_lb)) {
                  kappa_i_star = kappas_ub_3d[i];
                  log_phi_on_z_kappa_3d_i_star = log_phis_on_z_kappas_ub_3d[i];
                } else {
                  kappa_i_star = kappas_lb_3d[i];
                  log_phi_on_z_kappa_3d_i_star = log_phis_on_z_kappas_lb_3d[i];
                }
              }
            }
            // Investigate kappa_j:
            float kappa_j_star;
            float K1i1j_star_lb = std::sqrt(
                kappa_i_star * kappa_i_star + kappas_lb_3d[j] * kappas_lb_3d[j]
                    + 2.0f * kappa_i_star * kappas_lb_3d[j] * cos_theta);
            // If gradient is negative at kappa_lb, use kappa_ub
            if (!ComputeGradientSign(
                kappas_lb_3d[j], K1i1j_star_lb, kappa_i_star * cos_theta)) {
              kappa_j_star = kappas_ub_3d[j];
              log_phi_on_z_kappa_3d_j_star = log_phis_on_z_kappas_ub_3d[j];
            } else {
              // Gradient positive at kappa_lb, investigate kappa_j further
              float K1i1j_star_ub = std::sqrt(
                  kappa_i_star * kappa_i_star
                      + kappas_ub_3d[j] * kappas_ub_3d[j]
                      + 2.0f * kappa_i_star * kappas_ub_3d[j] * cos_theta);
              // If gradient is positive at kappa_ub, use kappa_lb
              if (ComputeGradientSign(
                  kappas_ub_3d[j], K1i1j_star_ub, kappa_i_star * cos_theta)) {
                kappa_j_star = kappas_lb_3d[j];
                log_phi_on_z_kappa_3d_j_star = log_phis_on_z_kappas_lb_3d[j];
              } else {
                // If gradient is positive at kappa_lb and negative at kappa_ub, minimiser is either kappa_lb or kappa_ub
                if (log_phis_on_z_kappas_ub_3d[j] + LogZFunction(K1i1j_star_ub)
                    <= log_phis_on_z_kappas_lb_3d[j]
                        + LogZFunction(K1i1j_star_lb)) {
                  kappa_j_star = kappas_ub_3d[j];
                  log_phi_on_z_kappa_3d_j_star = log_phis_on_z_kappas_ub_3d[j];
                } else {
                  kappa_j_star = kappas_lb_3d[j];
                  log_phi_on_z_kappa_3d_j_star = log_phis_on_z_kappas_lb_3d[j];
                }
              }
            }
            K1i1j_star = std::sqrt(
                kappa_i_star * kappa_i_star + kappa_j_star * kappa_j_star
                    + 2.0f * kappa_i_star * kappa_j_star * cos_theta);
          }
        }
        lower_bound_11 += exp(
            c_log_class_weights[c] + log_phi_on_z_kappa_3d_i_star
                + log_phi_on_z_kappa_3d_j_star + LogZFunction(K1i1j_star));
      }
      for (int j = start_index_2d; j < stop_index_2d; ++j) {
        // Upper Bound Calculations:
        float cos_angle = mus_3d_translated_normalised[i][0]
            * mus_2d_rotated[j][0]
            + mus_3d_translated_normalised[i][1] * mus_2d_rotated[j][1]
            + mus_3d_translated_normalised[i][2] * mus_2d_rotated[j][2];
        if (cos_angle <= -1.0f)
          cos_angle = -1.0f;
        float kappa2_2d = kappas_2d[j] * kappas_2d[j];
        float K1i2j = std::sqrt(
            kappas_3d[i] * kappas_3d[i] + kappa2_2d
                + 2.0f * kappas_3d[i] * kappas_2d[j] * cos_angle);
        upper_bound_12 += exp(
            c_log_class_weights[c] + log_phis_on_z_kappas_3d[i]
                + log_phis_on_z_kappas_2d[j] + LogZFunction(K1i2j));
        // Lower Bound Calculations:
        float K1i2j_star;
        float log_phi_on_z_kappa_3d_star;
        float angle_minus_sum_translation_rotation_uncertainty_angles =
            std::acos(cos_angle) - translation_uncertainty_angles[i]
                - rotation_uncertainty_angles[j];
        // If angle sum has been clipped to 0 degrees
        if (angle_minus_sum_translation_rotation_uncertainty_angles <= 0.0f) {
          // Z(K)/Z(kappa) is maximised by kappa_ub for angle = 0 degrees
          K1i2j_star = kappas_ub_3d[i] + kappas_2d[j];
          log_phi_on_z_kappa_3d_star = log_phis_on_z_kappas_ub_3d[i];
        } else {
          float kappa_2d_cos_theta = kappas_2d[j]
              * std::cos(
                  angle_minus_sum_translation_rotation_uncertainty_angles);
          if (angle_minus_sum_translation_rotation_uncertainty_angles
              < 0.5f * GOSMA::kPi) {  // If angle is (0, 90) degrees
            // Z(K)/Z(kappa) is maximised by kappa = [kappa_lb, kappa_ub] for
            // angle in (0, 90) degrees (kappa_lb is most common for large kappa2d)
            float K1i2j_lb = std::sqrt(
                kappas_lb_3d[i] * kappas_lb_3d[i] + kappa2_2d
                    + 2.0f * kappas_lb_3d[i] * kappa_2d_cos_theta);
            // If gradient is negative at kappa_lb
            if (!ComputeGradientSign(
                kappas_lb_3d[i], K1i2j_lb, kappa_2d_cos_theta)) {
              K1i2j_star = K1i2j_lb;
              log_phi_on_z_kappa_3d_star = log_phis_on_z_kappas_lb_3d[i];
            } else {
              float K1i2j_ub = std::sqrt(
                  kappas_ub_3d[i] * kappas_ub_3d[i] + kappa2_2d
                      + 2.0f * kappas_ub_3d[i] * kappa_2d_cos_theta);
              // If gradient is positive at kappa_lb and kappa_ub
              if (ComputeGradientSign(
                  kappas_ub_3d[i], K1i2j_ub, kappa_2d_cos_theta)) {
                K1i2j_star = K1i2j_ub;
                log_phi_on_z_kappa_3d_star = log_phis_on_z_kappas_ub_3d[i];
              } else {
                // If gradient is positive at kappa_lb and negative at kappa_ub
                float kappa_3d_star;
                K1i2j_star = ComputeKstar(
                    kappas_lb_3d[i], kappas_ub_3d[i], kappa2_2d,
                    kappa_2d_cos_theta, kappa_3d_star);
                log_phi_on_z_kappa_3d_star = c_log_phis_3d[i]
                    - LogZFunction(kappa_3d_star);
              }
            }
          } else {
            // Z(K)/Z(kappa) is maximised by kappa_lb for angle >= 90 degrees
            float K1i2j_star2 = kappas_lb_3d[i] * kappas_lb_3d[i] + kappa2_2d
                + 2.0f * kappas_lb_3d[i] * kappa_2d_cos_theta;
            if (K1i2j_star2 > 0.0f) {
              K1i2j_star = std::sqrt(K1i2j_star2);
            } else {
              K1i2j_star = 0.0f;
            }
            log_phi_on_z_kappa_3d_star = log_phis_on_z_kappas_lb_3d[i];
          }
        }
        lower_bound_12 += exp(
            c_log_class_weights[c] + log_phi_on_z_kappa_3d_star
                + log_phis_on_z_kappas_2d[j] + LogZFunction(K1i2j_star));
      }
    }
    start_index_2d = stop_index_2d;
    start_index_3d = stop_index_3d;
  }
  node.ub = (upper_bound_11 - 2.0f * upper_bound_12) / c_l2_normaliser;
  node.lb = (lower_bound_11 - 2.0f * lower_bound_12) / c_l2_normaliser;

  // Select split dimension
  if (max_translation_uncertainty_angle >= max_rotation_uncertainty_angle) {
    node.branch = 1;
  } else {
    node.branch = 2;
  }

  d_nodes_[global_index] = node;
}

/*
 * RotationVectorToMatrix
 * Inputs:
 * - rotation_vector: rotation vector
 * Outputs:
 * - return_value: rotation matrix corresponding to the vector
 */
__device__ void RotationVectorToMatrix(
    const float (&rotation_vector)[3], float (&rotation_matrix)[3][3]) {
  float rotation_angle = std::sqrt(
      rotation_vector[0] * rotation_vector[0]
          + rotation_vector[1] * rotation_vector[1]
          + rotation_vector[2] * rotation_vector[2]);
  float v[3] = { rotation_vector[0] / rotation_angle,
      rotation_vector[1] / rotation_angle, rotation_vector[2] / rotation_angle};
  float ca = std::cos(rotation_angle);
  float ca2 = 1 - ca;
  float sa = std::sin(rotation_angle);
  float v0sa = v[0] * sa;
  float v1sa = v[1] * sa;
  float v2sa = v[2] * sa;
  float v0v1ca2 = v[0] * v[1] * ca2;
  float v0v2ca2 = v[0] * v[2] * ca2;
  float v1v2ca2 = v[1] * v[2] * ca2;
  rotation_matrix[0][0] = ca + v[0] * v[0] * ca2;
  rotation_matrix[0][1] = v0v1ca2 - v2sa;
  rotation_matrix[0][2] = v0v2ca2 + v1sa;
  rotation_matrix[1][0] = v0v1ca2 + v2sa;
  rotation_matrix[1][1] = ca + v[1] * v[1] * ca2;
  rotation_matrix[1][2] = v1v2ca2 - v0sa;
  rotation_matrix[2][0] = v0v2ca2 - v1sa;
  rotation_matrix[2][1] = v1v2ca2 + v0sa;
  rotation_matrix[2][2] = ca + v[2] * v[2] * ca2;
}

/*
 * ComputeTranslationUncertaintyAngle
 * Inputs:
 * - mu_3d_translated_normalised: normalised centre point of the cuboid formed by subtracting the translation subcuboid from mu_3d
 * - min_mu_3d_translated: minimum point of the cuboid formed by subtracting the translation subcuboid from mu_3d
 * - max_mu_3d_translated: maximum point of the cuboid formed by subtracting the translation subcuboid from mu_3d
 * Outputs:
 * - return_value: cosine of the translation uncertainty angle
 */
__device__ float ComputeTranslationUncertaintyAngle(
    const float (&mu_3d_translated_normalised)[3],
    const float (&min_mu_3d_translated)[3],
    const float (&max_mu_3d_translated)[3], float &max_mu_3d_translated_norm2) {
  // Check if origin is inside cube of translations (mu3d - t)
  bool origin_inside_cube = true;
  for (int i = 0; i < 3; ++i) {
    if (min_mu_3d_translated[i] > 0.0f || max_mu_3d_translated[i] < 0.0f) {
      origin_inside_cube = false;
    }
  }
  if (origin_inside_cube) {
    float vertex[3];
    for (int i = 0; i < 2; ++i) {
      vertex[0] = (i == 0) ? min_mu_3d_translated[0] : max_mu_3d_translated[0];
      for (int j = 0; j < 2; ++j) {
        vertex[1] =
            (j == 0) ? min_mu_3d_translated[1] : max_mu_3d_translated[1];
        for (int k = 0; k < 2; ++k) {
          vertex[2] =
              (k == 0) ? min_mu_3d_translated[2] : max_mu_3d_translated[2];
          float vertex_norm2 = vertex[0] * vertex[0] + vertex[1] * vertex[1]
              + vertex[2] * vertex[2];
          if (vertex_norm2 > max_mu_3d_translated_norm2)
            max_mu_3d_translated_norm2 = vertex_norm2;
        }
      }
    }
    return -1.0f;
  } else {
    float min_cos_angle = 1.0f;
    float vertex[3];
    for (int i = 0; i < 2; ++i) {
      vertex[0] = (i == 0) ? min_mu_3d_translated[0] : max_mu_3d_translated[0];
      for (int j = 0; j < 2; ++j) {
        vertex[1] =
            (j == 0) ? min_mu_3d_translated[1] : max_mu_3d_translated[1];
        for (int k = 0; k < 2; ++k) {
          vertex[2] =
              (k == 0) ? min_mu_3d_translated[2] : max_mu_3d_translated[2];
          float vertex_norm2 = vertex[0] * vertex[0] + vertex[1] * vertex[1]
              + vertex[2] * vertex[2];
          if (vertex_norm2 > max_mu_3d_translated_norm2)
            max_mu_3d_translated_norm2 = vertex_norm2;
          float vertex_norm = std::sqrt(vertex_norm2);
          if (vertex_norm <= FLT_EPSILON) {
            min_cos_angle = -1.0f;
            break;
          } else {
            float cos_angle = (mu_3d_translated_normalised[0] * vertex[0]
                + mu_3d_translated_normalised[1] * vertex[1]
                + mu_3d_translated_normalised[2] * vertex[2]) / vertex_norm;
            if (cos_angle < min_cos_angle) {
              min_cos_angle = cos_angle;
            }
          }
        }
      }
    }
    return min_cos_angle;
  }
}

/*
 * ComputeRotationUncertaintyAngle
 * Inputs:
 * - mu_2d: bearing vector for which to calculate the uncertainty angle
 * - mu_2d_rotated: bearing vector rotated by the inverse of the rotation vector at the centre of the rotation subcube
 * - min_rotation_vector: minimum vertex of the rotation subcube
 * - max_rotation_vector: maximum vertex of the rotation subcube
 * Outputs:
 * - return_value: cosine of the rotation uncertainty angle
 */
__device__ float ComputeRotationUncertaintyAngle(
    const float (&mu_2d)[3], const float (&mu_2d_rotated)[3],
    const float (&min_rotation_vector)[3],
    const float (&max_rotation_vector)[3]) {
  // Method: Test all vertices and use derivative to check whether edges need inspection
  // 1. Rotate by all vertices: bearing_vector_rotated_vertices_
  // 2. Evaluate angles between vertex-rotated and centre-rotated bearing vector
  // 3. Evaluate s1 = sign(dcosA/dlambda at lambda=0) and s2 = sign(dcosA/dlambda at lambda=1)

  // Get vertex rotation vectors, vertex-rotated bearing vectors and derivative components
  int linear_index = 0;
  float r[3];
  float vertex_rotation_vectors[8][3];
  float vertex_derivative_components[8][3][3];
  float vertex_rotated_mu_2d[8][3];
  for (int i = 0; i < 2; ++i) {
    r[0] = (i == 0) ? min_rotation_vector[0] : max_rotation_vector[0];
    for (int j = 0; j < 2; ++j) {
      r[1] = (j == 0) ? min_rotation_vector[1] : max_rotation_vector[1];
      for (int k = 0; k < 2; ++k) {
        r[2] = (k == 0) ? min_rotation_vector[2] : max_rotation_vector[2];
        for (int l = 0; l < 3; ++l)
          vertex_rotation_vectors[linear_index][l] = r[l];
        float vertex_rotation_angle2 = vertex_rotation_vectors[linear_index][0]
            * vertex_rotation_vectors[linear_index][0]
            + vertex_rotation_vectors[linear_index][1]
                * vertex_rotation_vectors[linear_index][1]
            + vertex_rotation_vectors[linear_index][2]
                * vertex_rotation_vectors[linear_index][2];
        if (vertex_rotation_angle2 <= FLT_EPSILON) {  // Check for very small rotations
          for (int l = 0; l < 3; ++l) {
            vertex_rotated_mu_2d[linear_index][l] = mu_2d_rotated[l];
            for (int m = 0; m < 3; ++m) {
              vertex_derivative_components[linear_index][l][m] = 0.0f;
            }
          }
        } else {
          float R[3][3];
          RotationVectorToMatrix(r, R);
          // Rotate by R^-1
          vertex_rotated_mu_2d[linear_index][0] = R[0][0] * mu_2d[0]
              + R[1][0] * mu_2d[1] + R[2][0] * mu_2d[2];
          vertex_rotated_mu_2d[linear_index][1] = R[0][1] * mu_2d[0]
              + R[1][1] * mu_2d[1] + R[2][1] * mu_2d[2];
          vertex_rotated_mu_2d[linear_index][2] = R[0][2] * mu_2d[0]
              + R[1][2] * mu_2d[1] + R[2][2] * mu_2d[2];
          // vertex_derivative_components[linear_index] = Ri' * (ri * ri' - (Ri - I) * [ri]x);
          vertex_derivative_components[linear_index][0][0] = R[0][0]
              * (r[0] * r[0] - R[0][1] * r[2] + R[0][2] * r[1])
              + R[1][0] * (R[1][2] * r[1] + r[0] * r[1] - r[2] * (R[1][1] - 1))
              + R[2][0] * (r[0] * r[2] - R[2][1] * r[2] + r[1] * (R[2][2] - 1));
          vertex_derivative_components[linear_index][0][1] = R[1][0]
              * (r[1] * r[1] + R[1][0] * r[2] - R[1][2] * r[0])
              + R[0][0] * (r[0] * r[1] - R[0][2] * r[0] + r[2] * (R[0][0] - 1))
              + R[2][0] * (R[2][0] * r[2] + r[1] * r[2] - r[0] * (R[2][2] - 1));
          vertex_derivative_components[linear_index][0][2] = R[2][0]
              * (r[2] * r[2] - R[2][0] * r[1] + R[2][1] * r[0])
              + R[0][0] * (R[0][1] * r[0] + r[0] * r[2] - r[1] * (R[0][0] - 1))
              + R[1][0] * (r[1] * r[2] - R[1][0] * r[1] + r[0] * (R[1][1] - 1));
          vertex_derivative_components[linear_index][1][0] = R[0][1]
              * (r[0] * r[0] - R[0][1] * r[2] + R[0][2] * r[1])
              + R[1][1] * (R[1][2] * r[1] + r[0] * r[1] - r[2] * (R[1][1] - 1))
              + R[2][1] * (r[0] * r[2] - R[2][1] * r[2] + r[1] * (R[2][2] - 1));
          vertex_derivative_components[linear_index][1][1] = R[1][1]
              * (r[1] * r[1] + R[1][0] * r[2] - R[1][2] * r[0])
              + R[0][1] * (r[0] * r[1] - R[0][2] * r[0] + r[2] * (R[0][0] - 1))
              + R[2][1] * (R[2][0] * r[2] + r[1] * r[2] - r[0] * (R[2][2] - 1));
          vertex_derivative_components[linear_index][1][2] = R[2][1]
              * (r[2] * r[2] - R[2][0] * r[1] + R[2][1] * r[0])
              + R[0][1] * (R[0][1] * r[0] + r[0] * r[2] - r[1] * (R[0][0] - 1))
              + R[1][1] * (r[1] * r[2] - R[1][0] * r[1] + r[0] * (R[1][1] - 1));
          vertex_derivative_components[linear_index][2][0] = R[0][2]
              * (r[0] * r[0] - R[0][1] * r[2] + R[0][2] * r[1])
              + R[1][2] * (R[1][2] * r[1] + r[0] * r[1] - r[2] * (R[1][1] - 1))
              + R[2][2] * (r[0] * r[2] - R[2][1] * r[2] + r[1] * (R[2][2] - 1));
          vertex_derivative_components[linear_index][2][1] = R[1][2]
              * (r[1] * r[1] + R[1][0] * r[2] - R[1][2] * r[0])
              + R[0][2] * (r[0] * r[1] - R[0][2] * r[0] + r[2] * (R[0][0] - 1))
              + R[2][2] * (R[2][0] * r[2] + r[1] * r[2] - r[0] * (R[2][2] - 1));
          vertex_derivative_components[linear_index][2][2] = R[2][2]
              * (r[2] * r[2] - R[2][0] * r[1] + R[2][1] * r[0])
              + R[0][2] * (R[0][1] * r[0] + r[0] * r[2] - r[1] * (R[0][0] - 1))
              + R[1][2] * (r[1] * r[2] - R[1][0] * r[1] + r[0] * (R[1][1] - 1));
        }
        linear_index++;
      }
    }
  }

  // Loop over vertices to find largest angle between the centre and a vertex
  float min_cos_centre_to_vertex_angle = 1.0f;
  for (int i = 0; i < 8; ++i) {  // Loop over the vertices to find maximum angle
    float cos_centre_to_vertex_angle = mu_2d_rotated[0]
        * vertex_rotated_mu_2d[i][0]
        + mu_2d_rotated[1] * vertex_rotated_mu_2d[i][1]
        + mu_2d_rotated[2] * vertex_rotated_mu_2d[i][2];
    if (cos_centre_to_vertex_angle < min_cos_centre_to_vertex_angle)
      min_cos_centre_to_vertex_angle = cos_centre_to_vertex_angle;
  }

  // Loop over edges to determine whether additional searching is required
  // (~0.64% of edges, ~0.55% of bearing vectors)
  static const int edge_indices[12][2] = { { 0, 1 }, { 0, 2 }, { 0, 4 },
      { 1, 3 }, { 1, 5 }, { 2, 3 }, { 2, 6 }, { 3, 7 }, { 4, 5 }, { 4, 6 },
      { 5, 7 }, { 6, 7 } };
  for (int i = 0; i < 12; ++i) {  // Loop over the edges
    int v1 = edge_indices[i][0];
    int v2 = edge_indices[i][1];
    float vector_difference[3] = {
        vertex_rotation_vectors[v2][0] - vertex_rotation_vectors[v1][0],
        vertex_rotation_vectors[v2][1] - vertex_rotation_vectors[v1][1],
        vertex_rotation_vectors[v2][2] - vertex_rotation_vectors[v1][2] };
    float cross_product[3] = {
        vertex_rotated_mu_2d[v1][1]
            * (vertex_derivative_components[v1][2][0] * vector_difference[0]
                + vertex_derivative_components[v1][2][1] * vector_difference[1]
                + vertex_derivative_components[v1][2][2] * vector_difference[2])
            - vertex_rotated_mu_2d[v1][2]
                * (vertex_derivative_components[v1][1][0] * vector_difference[0]
                    + vertex_derivative_components[v1][1][1]
                        * vector_difference[1]
                    + vertex_derivative_components[v1][1][2]
                        * vector_difference[2]), vertex_rotated_mu_2d[v1][2]
            * (vertex_derivative_components[v1][0][0] * vector_difference[0]
                + vertex_derivative_components[v1][0][1] * vector_difference[1]
                + vertex_derivative_components[v1][0][2] * vector_difference[2])
            - vertex_rotated_mu_2d[v1][0]
                * (vertex_derivative_components[v1][2][0] * vector_difference[0]
                    + vertex_derivative_components[v1][2][1]
                        * vector_difference[1]
                    + vertex_derivative_components[v1][2][2]
                        * vector_difference[2]), vertex_rotated_mu_2d[v1][0]
            * (vertex_derivative_components[v1][1][0] * vector_difference[0]
                + vertex_derivative_components[v1][1][1] * vector_difference[1]
                + vertex_derivative_components[v1][1][2] * vector_difference[2])
            - vertex_rotated_mu_2d[v1][1]
                * (vertex_derivative_components[v1][0][0] * vector_difference[0]
                    + vertex_derivative_components[v1][0][1]
                        * vector_difference[1]
                    + vertex_derivative_components[v1][0][2]
                        * vector_difference[2]) };
    // angle_derivative_at_0 = -(R0'f)' * (Ri'f)x(vertex_derivative_components_i * (rj - ri));
    float angle_derivative_at_0 = -(mu_2d_rotated[0] * cross_product[0]
        + mu_2d_rotated[1] * cross_product[1]
        + mu_2d_rotated[2] * cross_product[2]);
    if (angle_derivative_at_0 >= 0.0f) {
      cross_product[0] = vertex_rotated_mu_2d[v2][1]
          * (vertex_derivative_components[v2][2][0] * vector_difference[0]
              + vertex_derivative_components[v2][2][1] * vector_difference[1]
              + vertex_derivative_components[v2][2][2] * vector_difference[2])
          - vertex_rotated_mu_2d[v2][2]
              * (vertex_derivative_components[v2][1][0] * vector_difference[0]
                  + vertex_derivative_components[v2][1][1]
                      * vector_difference[1]
                  + vertex_derivative_components[v2][1][2]
                      * vector_difference[2]);
      cross_product[1] = vertex_rotated_mu_2d[v2][2]
          * (vertex_derivative_components[v2][0][0] * vector_difference[0]
              + vertex_derivative_components[v2][0][1] * vector_difference[1]
              + vertex_derivative_components[v2][0][2] * vector_difference[2])
          - vertex_rotated_mu_2d[v2][0]
              * (vertex_derivative_components[v2][2][0] * vector_difference[0]
                  + vertex_derivative_components[v2][2][1]
                      * vector_difference[1]
                  + vertex_derivative_components[v2][2][2]
                      * vector_difference[2]);
      cross_product[2] = vertex_rotated_mu_2d[v2][0]
          * (vertex_derivative_components[v2][1][0] * vector_difference[0]
              + vertex_derivative_components[v2][1][1] * vector_difference[1]
              + vertex_derivative_components[v2][1][2] * vector_difference[2])
          - vertex_rotated_mu_2d[v2][1]
              * (vertex_derivative_components[v2][0][0] * vector_difference[0]
                  + vertex_derivative_components[v2][0][1]
                      * vector_difference[1]
                  + vertex_derivative_components[v2][0][2]
                      * vector_difference[2]);
      // angle_derivative_at_1 = -(R0'f)' * (Rj'f)x(vertex_derivative_components_j * (rj - ri));
      float angle_derivative_at_1 = -(mu_2d_rotated[0] * cross_product[0]
          + mu_2d_rotated[1] * cross_product[1]
          + mu_2d_rotated[2] * cross_product[2]);
      if (angle_derivative_at_1 <= 0.0f) {
        // Golden-Section Search (assumption: unimodal on interval)
        // Search along edge for angle maximiser (dot_product minimiser)
        float a[3] = { vertex_rotation_vectors[v1][0],
            vertex_rotation_vectors[v1][1], vertex_rotation_vectors[v1][2] };
        float b[3] = { vertex_rotation_vectors[v2][0],
            vertex_rotation_vectors[v2][1], vertex_rotation_vectors[v2][2] };
        float c[3] = { b[0] - GOSMA::kInvertedGoldenRatio * (b[0] - a[0]), b[1]
            - GOSMA::kInvertedGoldenRatio * (b[1] - a[1]), b[2]
            - GOSMA::kInvertedGoldenRatio * (b[2] - a[2]) };
        float d[3] = { a[0] + GOSMA::kInvertedGoldenRatio * (b[0] - a[0]), a[1]
            + GOSMA::kInvertedGoldenRatio * (b[1] - a[1]), a[2]
            + GOSMA::kInvertedGoldenRatio * (b[2] - a[2]) };
        float fc = RotateAndDot(mu_2d_rotated, mu_2d, c);
        float fd = RotateAndDot(mu_2d_rotated, mu_2d, d);
        while (pow(c[0] - d[0], 2) + pow(c[1] - d[1], 2) + pow(c[2] - d[2], 2)
            > GOSMA::kToleranceSquared) {  // norm > tol
          if (fc < fd) {
            for (int j = 0; j < 3; ++j) {
              b[j] = d[j];
              d[j] = c[j];
              c[j] = a[j] + GOSMA::kOneMinusInvertedGoldenRatio * (b[j] - a[j]);
            }
            fd = fc;
            fc = RotateAndDot(mu_2d_rotated, mu_2d, c);
          } else {
            for (int j = 0; j < 3; ++j) {
              a[j] = c[j];
              c[j] = d[j];
              d[j] = b[j] - GOSMA::kOneMinusInvertedGoldenRatio * (b[j] - a[j]);
            }
            fc = fd;
            fd = RotateAndDot(mu_2d_rotated, mu_2d, d);
          }
        }
        for (int j = 0; j < 3; ++j)
          c[j] = 0.5f * (b[j] + a[j]);
        fc = RotateAndDot(mu_2d_rotated, mu_2d, c);
        if (fc < min_cos_centre_to_vertex_angle)
          min_cos_centre_to_vertex_angle = fc;
      }
    }
  }
  return min_cos_centre_to_vertex_angle;
}

/*
 * RotateAndDot
 * dot(mu_2d_centre_rotated, Rr' * mu_2d)
 * Inputs:
 * - mu_2d_centre_rotated: mu2d rotated by the inverse of the rotation vector at the centre of the rotation subcube
 * - mu_2d: mu2d with which to dot product
 * - rotation_vector: rotation vector by which to rotate mu2d
 * Outputs:
 * - return_value: dot product of centre-rotated mu2d and rotation-vector-rotated mu2d
 */
__device__ float RotateAndDot(
    const float (&mu_2d_centre_rotated)[3], const float (&mu_2d)[3],
    const float (&rotation_vector)[3]) {
  float rotation_angle = std::sqrt(
      rotation_vector[0] * rotation_vector[0]
          + rotation_vector[1] * rotation_vector[1]
          + rotation_vector[2] * rotation_vector[2]);
  float v[3] = { rotation_vector[0] / rotation_angle, rotation_vector[1]
      / rotation_angle, rotation_vector[2] / rotation_angle };
  float ca = std::cos(rotation_angle);
  float ca2 = 1 - ca;
  float sa = std::sin(rotation_angle);
  float v0sa = v[0] * sa;
  float v1sa = v[1] * sa;
  float v2sa = v[2] * sa;
  float v0v1ca2 = v[0] * v[1] * ca2;
  float v0v2ca2 = v[0] * v[2] * ca2;
  float v1v2ca2 = v[1] * v[2] * ca2;
  float R[3][3];
  R[0][0] = ca + v[0] * v[0] * ca2;
  R[0][1] = v0v1ca2 - v2sa;
  R[0][2] = v0v2ca2 + v1sa;
  R[1][0] = v0v1ca2 + v2sa;
  R[1][1] = ca + v[1] * v[1] * ca2;
  R[1][2] = v1v2ca2 - v0sa;
  R[2][0] = v0v2ca2 - v1sa;
  R[2][1] = v1v2ca2 + v0sa;
  R[2][2] = ca + v[2] * v[2] * ca2;

  // Rotate by R^-1 and dot with other bearing vector
  return (R[0][0] * mu_2d[0] + R[1][0] * mu_2d[1] + R[2][0] * mu_2d[2])
      * mu_2d_centre_rotated[0]
      + (R[0][1] * mu_2d[0] + R[1][1] * mu_2d[1] + R[2][1] * mu_2d[2])
          * mu_2d_centre_rotated[1]
      + (R[0][2] * mu_2d[0] + R[1][2] * mu_2d[1] + R[2][2] * mu_2d[2])
          * mu_2d_centre_rotated[2];
}

/*
 * InvalidTranslationSubcube
 * Inputs:
 * - min_mu_3d_translated: mu3d translated by the translation subcuboid vertex with the smallest coordinates
 * - max_mu_3d_translated: mu3d translated by the translation subcuboid vertex with the largest coordinates
 * Outputs:
 * - return_value: true if translation subcuboid is entirely within the minimum distance to the camera
 */
__device__ bool InvalidTranslationSubcube(
    const float (&min_mu_3d_translated)[3],
    const float (&max_mu_3d_translated)[3], const float zeta2) {
  return CuboidInsideSphere(min_mu_3d_translated, max_mu_3d_translated, zeta2);
}

/*
 * InvalidRotationSubcube
 * Inputs:
 * - min_rotation_vector: rotation subcube vertex with the smallest coordinates
 * - max_rotation_vector: rotation subcube vertex with the largest coordinates
 * Outputs:
 * - return_value: true if subcube is entirely outside the rotation pi-ball
 */
__device__ bool InvalidRotationSubcube(
    const float (&min_rotation_vector)[3],
    const float (&max_rotation_vector)[3]) {
  return CuboidOutsideSphere(
      min_rotation_vector, max_rotation_vector, GOSMA::kPi * GOSMA::kPi);
}

/*
 * CuboidInsideSphere
 * Test whether cuboid is entirely inside a sphere
 * Inputs:
 * - cuboid_min: cuboid vertex with smallest coordinates
 * - cuboid_max: cuboid vertex with largest coordinates
 * - radius_squared: square of sphere's radius
 * Outputs:
 * - return value: true if the cuboid is entirely inside the sphere
 */
__device__ bool CuboidInsideSphere(
    const float (&cuboid_min)[3], const float (&cuboid_max)[3],
    const float radius_squared) {
  float max_distance_squared = 0.0f;
  for (int i = 0; i < 3; ++i) {
    if (cuboid_min[i] > 0) {
      max_distance_squared += std::pow(cuboid_max[i], 2);
    } else if (cuboid_max[i] < 0) {
      max_distance_squared += std::pow(cuboid_min[i], 2);
    } else {
      max_distance_squared += std::pow(fmax(-cuboid_min[i], cuboid_max[i]), 2);
    }
  }
  return max_distance_squared <= radius_squared;
}

/*
 * CuboidOutsideSphere
 * Test whether cuboid is entirely outside a sphere
 * Inputs:
 * - cuboid_min: cuboid vertex with smallest coordinates
 * - cuboid_max: cuboid vertex with largest coordinates
 * - radius_squared: square of sphere's radius
 * Outputs:
 * - return value: true if the cuboid is entirely outside the sphere
 */
__device__ bool CuboidOutsideSphere(
    const float (&cuboid_min)[3], const float (&cuboid_max)[3],
    const float radius_squared) {
  return MinSquaredDistanceToCuboid(cuboid_min, cuboid_max) > radius_squared;
}

__device__ float MinSquaredDistanceToCuboid(
    const float (&cuboid_min)[3], const float (&cuboid_max)[3]) {
  float min_distance_squared = 0.0f;
  for (int i = 0; i < 3; ++i) {
    if (cuboid_min[i] > 0) {
      min_distance_squared += std::pow(cuboid_min[i], 2);
    } else if (cuboid_max[i] < 0) {
      min_distance_squared += std::pow(cuboid_max[i], 2);
    }
  }
  return min_distance_squared;
}

/*
 * LogZFunction
 * Inputs:
 * - x: scalar value
 * Outputs:
 * - return value: log((exp(x) - exp(-x)) / x)
 */
__device__ float LogZFunction(const float x) {
  if (fabs(x) < 1.0e-3f) {
    return log(2.0f);
  } else if (x < 10.0f) {
    return log(exp(x) - exp(-x)) - log(x);
  } else {  // equivalent for x >= 8 (float) or x >= 17 (double)
    return x - log(x);
  }
}

/*
 * ComputeGradientSign
 * Inputs:
 * - kappa_3d:
 * - K1i2j: function of kappa_3d
 * - kappa_2d_cos_theta:
 * Outputs:
 * - return value: true if gradient is positive at kappa_3d
 */
__device__ bool ComputeGradientSign(
    const float kappa_3d, const float K1i2j, const float kappa_2d_cos_theta) {
  if (ComputeGradient(kappa_3d, K1i2j, kappa_2d_cos_theta) >= 0.0f) {
    return true;
  } else {
    return false;
  }
}

/*
 * ComputeGradient
 * Inputs:
 * - kappa_3d:
 * - K1i2j: function of kappa_3d
 * - kappa_2d_cos_theta:
 * Outputs:
 * - return value: true if gradient is positive at kappa_3d
 */
__device__ float ComputeGradient(
    const float kappa_3d, const float K1i2j, const float kappa_2d_cos_theta) {
  if (kappa_3d >= 10.0f) {
    return kappa_3d * (kappa_3d + kappa_2d_cos_theta) * (K1i2j - 1.0f)
        - K1i2j * K1i2j * (kappa_3d - 1.0f);
  } else {
    float exp_minus_2_kappa = exp(-2.0f * kappa_3d);
    float exp_minus_2_K = exp(-2.0f * K1i2j);
    return kappa_3d * (kappa_3d + kappa_2d_cos_theta)
        * (1.0f - exp_minus_2_kappa)
        * (K1i2j * (1.0f + exp_minus_2_K) - (1.0f - exp_minus_2_K))
        - K1i2j * K1i2j * (1.0f - exp_minus_2_K)
            * (kappa_3d * (1.0f + exp_minus_2_kappa)
                - (1.0f - exp_minus_2_kappa));
  }
}

/*
 * ComputeKstar
 * Inputs:
 * - kappa_3d:
 * - K1i2j: function of kappa_3d
 * - kappa_2d_cos_theta:
 * Outputs:
 * - return value:
 */
__device__ float ComputeKstar(
    const float kappa_lb_3d, const float kappa_ub_3d, const float kappa2_2d,
    const float kappa_2d_cos_theta, float& kappa_3d_star) {
  // Only get here if gradient(lb) >= 0 and gradient(ub) < 0

  // Bisection Search
  int iteration_count = 0;  // Can oscillate, so need exit condition
  float a = kappa_lb_3d;
  float b = kappa_ub_3d;
  float c, Kc, gc;
  while (b - a > 0.1f && iteration_count < 100) {
    iteration_count++;
    c = 0.5f * (a + b);
    Kc = std::sqrt(c * c + kappa2_2d + 2.0f * c * kappa_2d_cos_theta);
    gc = ComputeGradient(c, Kc, kappa_2d_cos_theta);
    if (gc >= 0.0f) {
      a = c;
    } else {
      b = c;
    }
  }
  c = 0.5f * (a + b);
  Kc = std::sqrt(c * c + kappa2_2d + 2.0f * c * kappa_2d_cos_theta);
  kappa_3d_star = c;
  return Kc;
}

GOSMA::GOSMA() {
  num_components_2d_ = -1;
  num_components_3d_ = -1;
  num_classes_ = 1;
  optimality_certificate_ = -1;
  epsilon_ = 0.0001f;
  zeta_ = 0.1f;
  min_linear_resolution_ = 0.001f;
  min_angular_resolution_ = 0.001f;
  translation_domain_expansion_factor_ = 1.0f;
  min_l2_distance_ = std::numeric_limits<float>::max();
  min_fvalue_ = std::numeric_limits<float>::max();
  l2_normaliser_ = 1.0f;
  l2_constant_ = 0.0f;
  initial_node_.tx = -0.5;
  initial_node_.ty = -0.5;
  initial_node_.tz = -0.5;
  initial_node_.twx = 1.0;
  initial_node_.twy = 1.0;
  initial_node_.twz = 1.0;
  initial_node_.rx = -kPi;
  initial_node_.ry = -kPi;
  initial_node_.rz = -kPi;
  initial_node_.rw = 2.0 * kPi;
  initial_node_.lb = -std::numeric_limits<float>::infinity();
  initial_node_.ub = std::numeric_limits<float>::infinity();
  initial_node_.branch = 1;
}

GOSMA::~GOSMA() {
}

/*
 * Public Class Functions
 */

/*
 * Run
 * Run the GOSMA algorithm
 */
void GOSMA::Run() {
  if (CheckInputs()) {
    Initialise();
    BranchAndBound();
    Clear();
  }
}

/*
 * RunRefinementOnly
 * Run the GOSMA local optimisation algorithm
 */
void GOSMA::RunRefinementOnly() {
  if (CheckInputs()) {
    // Compute constant values
    ComputeLogClassWeights();
    ComputeLogPhis2d();
    ComputeLogPhis3d();
    ComputeLogPhisOnZKappas2d();
    ComputeL2Normaliser();
    ComputeL2Constant();

    // Set up local optimiser
    f_ = SphericalMML2Distance<double>(
        mus_2d_.cast<double>(), mus_3d_.cast<double>(),
        kappas_2d_.cast<double>(), variances_3d_.cast<double>(),
        log_phis_3d_.cast<double>(), log_phis_on_z_kappas_2d_.cast<double>(),
        class_num_components_2d_, class_num_components_3d_,
        std::vector<double>(log_class_weights_.begin(),
                            log_class_weights_.end()));
    // Create a Criteria class to set the solver's stop conditions
    cppoptlib::Criteria<double> stopping_criteria =
        cppoptlib::Criteria<double>::defaults();
    stopping_criteria.iterations = 200;  //!< Maximum number of iterations (10000)
//    stopping_criteria.xDelta = 1e-4;        //!< Minimum change in parameter vector (0)
//    stopping_criteria.fDelta = 0;        //!< Minimum change in cost function (0)
//    stopping_criteria.gradNorm = 1e-4;   //!< Minimum norm of gradient vector (1e-4)
//    stopping_criteria.gradNorm = 0;   //!< Minimum norm of gradient vector (1e-4)
    solver_.setStopCriteria(stopping_criteria);

    // Initialise bearing vector and point variables
    mus_2d_rotated_.resize(num_components_2d_, Eigen::NoChange);
    mus_3d_translated_.resize(num_components_3d_, Eigen::NoChange);
    mus_3d_translated_norm2_.resize(num_components_3d_);
    mus_3d_translated_normalised_.resize(num_components_3d_, Eigen::NoChange);

    // Compute L2 distance for node centre
    Node node = initial_queue_.top();
    initial_queue_.pop();
    ApplyTransformation(node);
    min_fvalue_ = ComputeFunctionValue();
    min_l2_distance_ = ComputeL2Distance(min_fvalue_);
    DEBUG_OUTPUT("L2 Distance: " << std::setprecision(6) << min_l2_distance_
                 << " (INITIAL)");

    optimal_translation_ = Eigen::RowVector3f(node.tx + 0.5f * node.twx,
                                              node.ty + 0.5f * node.twy,
                                              node.tz + 0.5f * node.twz);
    optimal_rotation_vector_ = Eigen::RowVector3f(node.rx + 0.5f * node.rw,
                                                  node.ry + 0.5f * node.rw,
                                                  node.rz + 0.5f * node.rw);
    optimal_rotation_ = RotationVectorToMatrix(optimal_rotation_vector_);

    // Run local optimisation to (potentially) find a better value
    Eigen::Matrix<double, 6, 1> x;
    x << optimal_translation_.transpose().cast<double>(),
        optimal_rotation_vector_.transpose().cast<double>();
    solver_.minimize(f_, x);
    float fvalue = static_cast<float>(f_(x)) / l2_normaliser_;
    if (fvalue < min_fvalue_) {
      min_fvalue_ = fvalue;
      Eigen::Matrix<float, 6, 1> x_float = x.cast<float>();
      optimal_translation_ = x_float.head(3).transpose();
      optimal_rotation_vector_ = x_float.tail(3).transpose();
      optimal_rotation_ = RotationVectorToMatrix(optimal_rotation_vector_);
      optimality_certificate_ = 0;  // If upper bound has been reduced, optimality is possible
      min_l2_distance_ = ComputeL2Distance(min_fvalue_);
      DEBUG_OUTPUT("L2 Distance: " << std::setprecision(6) << min_l2_distance_
                   << " (REFINEMENT)");
    }
  }
}

/*
 * Private Class Functions
 */

/*
 * CheckInputs
 * Outputs:
 * - bool: false if the 2D or 3D mixtures have not been loaded or the config parameters are invalid
 */
bool GOSMA::CheckInputs() {
  // Check if mixture parameters have been loaded
  if (mus_2d_.rows() < 1 || kappas_2d_.rows() < 1 || phis_2d_.rows() < 1) {
    std::cout << "2D mixture has not been loaded" << std::endl;
    return false;
  }
  if (mus_3d_.rows() < 1 || variances_3d_.rows() < 1 || phis_3d_.rows() < 1) {
    std::cout << "3D mixture has not been loaded" << std::endl;
    return false;
  }
  // Check whether mixtures exceed the maximum number permitted
  if (mus_2d_.rows() > GOSMA::kMaxNumComponents) {
    std::cout << "2D mixture has too many components (" << mus_2d_.rows() << " > " << GOSMA::kMaxNumComponents << ")" << std::endl;
    return false;
  }
  if (mus_3d_.rows() > GOSMA::kMaxNumComponents) {
    std::cout << "3D mixture has too many components (" << mus_3d_.rows() << " > " << GOSMA::kMaxNumComponents << ")" << std::endl;
    return false;
  }
  // Check validity of config options
  if (epsilon_ <= 0) {
    std::cout << "A valid epsilon value has not been provided" << std::endl;
    return false;
  }
  if (zeta_ < 0) {
    std::cout << "A valid zeta value has not been provided" << std::endl;
    return false;
  }
  return true;
}

/*
 * Initialise
 * Sets up variables used in the algorithm, initial translation domain, initial rotation domain
 */
void GOSMA::Initialise() {
  // Compute constant values
  optimality_certificate_ = -1;  // Reset optimality certificate
  epsilon_ *= 2.0f * kPi;  // Scale epsilon to function_value units (L2 distance: fvalue / 2pi + constant)
  ComputeLogClassWeights();
  ComputeLogPhis2d();
  ComputeLogPhis3d();
  ComputeLogPhisOnZKappas2d();
  float l2_normaliser = ComputeL2Normaliser();
  DEBUG_OUTPUT("L2 Normaliser: " << std::setprecision(6)
      << l2_normaliser / (2.0f * kPi));
  float l2_constant = ComputeL2Constant();
  DEBUG_OUTPUT("L2 Constant: " << std::setprecision(6)
      << l2_constant / (2.0f * kPi));

  // Set up local optimiser
  f_ = SphericalMML2Distance<double>(
      mus_2d_.cast<double>(), mus_3d_.cast<double>(),
      kappas_2d_.cast<double>(), variances_3d_.cast<double>(),
      log_phis_3d_.cast<double>(), log_phis_on_z_kappas_2d_.cast<double>(),
      class_num_components_2d_, class_num_components_3d_,
      std::vector<double>(log_class_weights_.begin(),
                          log_class_weights_.end()));
  f_float_ = SphericalMML2Distance<float>(
      mus_2d_, mus_3d_, kappas_2d_, variances_3d_, log_phis_3d_,
      log_phis_on_z_kappas_2d_, class_num_components_2d_,
      class_num_components_3d_, log_class_weights_);
  cppoptlib::Criteria<double> stopping_criteria =
      cppoptlib::Criteria<double>::defaults();  // Create a Criteria class to set the solver's stop conditions
  stopping_criteria.iterations = 200;  //!< Maximum number of iterations (10000)
//  stopping_criteria.xDelta = 1e-4;        //!< Minimum change in parameter vector (0)
//  stopping_criteria.fDelta = 0;        //!< Minimum change in cost function (0)
//  stopping_criteria.gradNorm = 1e-4;   //!< Minimum norm of gradient vector (1e-4)
//  stopping_criteria.gradNorm = 0;   //!< Minimum norm of gradient vector (1e-4)
  solver_.setStopCriteria(stopping_criteria);

  // Initialise vector of array of nodes
  d_nodes_.resize(kNumDevices);
  d_parent_nodes_.resize(kNumDevices);

  // Initialise bearing vector and point variables
  mus_2d_rotated_.resize(num_components_2d_, Eigen::NoChange);
  mus_3d_translated_.resize(num_components_3d_, Eigen::NoChange);
  mus_3d_translated_norm2_.resize(num_components_3d_);
  mus_3d_translated_normalised_.resize(num_components_3d_, Eigen::NoChange);

  // Initialise translation domain using a bounding box if not supplied by the user
  if (initial_queue_.empty()) {
    // Equivalent to the padded axis-aligned bounding box of the 3D Gaussian centres
    // Set expansion factor to 1.0 if camera is known to be within the bounding box)
    Eigen::Vector3f mus_3d_min = mus_3d_.colwise().minCoeff().transpose();
    Eigen::Vector3f mus_3d_max = mus_3d_.colwise().maxCoeff().transpose();
    Eigen::Vector3f centre = 0.5f * (mus_3d_max + mus_3d_min);
    Eigen::Vector3f widths = mus_3d_max - mus_3d_min;

    // Pad each dimension by max_width * (expansion_factor - 1)
    float pad = (translation_domain_expansion_factor_ - 1) * widths.maxCoeff();
    initial_node_.twx = widths(0) + pad;
    initial_node_.twy = widths(1) + pad;
    initial_node_.twz = widths(2) + pad;
    initial_node_.tx = centre(0, 0) - 0.5f * initial_node_.twx;
    initial_node_.ty = centre(1, 0) - 0.5f * initial_node_.twy;
    initial_node_.tz = centre(2, 0) - 0.5f * initial_node_.twz;
    initial_node_.lb = -std::numeric_limits<float>::infinity();
    initial_node_.ub = std::numeric_limits<float>::infinity();
    initial_queue_.push(initial_node_);  // Push node to queue
  }

  // Initialise optimal transformations
  optimal_translation_ = Eigen::RowVector3f::Zero();
  optimal_rotation_vector_ = Eigen::RowVector3f::Zero();
  optimal_rotation_ = Eigen::Matrix3f::Identity();

  // Initialise the optimal value
  InitialValue();

  // Subdivide transformation domain into 4096 nodes: 64 (4^3) translation subcubes and 64 (4^3) rotation subcubes
  if (initial_queue_.size() == 1) {
    Node initial_node = initial_queue_.top();  // initial_node unless user supplied domain
    initial_queue_.pop();
    Node node;
    node.lb = -std::numeric_limits<float>::infinity();
    node.ub = min_fvalue_;
    node.branch = 2;  // Branch over rotation first
    for (int i = 0; i < 64; ++i) {
      node.twx = initial_node.twx * 0.25f;
      node.twy = initial_node.twy * 0.25f;
      node.twz = initial_node.twz * 0.25f;
      node.tx = initial_node.tx + node.twx * static_cast<float>(i & 3);  // 0b11
      node.ty = initial_node.ty + node.twy * static_cast<float>((i >> 2) & 3);
      node.tz = initial_node.tz + node.twz * static_cast<float>((i >> 4) & 3);
      for (int j = 0; j < 64; ++j) {
        node.rw = initial_node.rw * 0.25f;
        node.rx = initial_node.rx + node.rw * static_cast<float>(j & 3);  // 0b11
        node.ry = initial_node.ry + node.rw * static_cast<float>((j >> 2) & 3);
        node.rz = initial_node.rz + node.rw * static_cast<float>((j >> 4) & 3);
        initial_queue_.push(node);  // Push node to queue
      }
    }
  } else {  // Multiple input domains
    while (initial_queue_.size() < kNumConcurrentNodes) {
      std::priority_queue<Node> queue_temp;
      while (!initial_queue_.empty()) {
        Node parent_node = initial_queue_.top();
        initial_queue_.pop();
        Node node;
        node.lb = -std::numeric_limits<float>::infinity();
        node.ub = min_fvalue_;
        node.branch = 2;  // Branch over rotation first
        if (parent_node.rw > 0.5f * kPi) {  // Split over rotation until sufficiently small
          node.tx = parent_node.tx;
          node.ty = parent_node.ty;
          node.tz = parent_node.tz;
          node.twx = parent_node.twx;
          node.twy = parent_node.twy;
          node.twz = parent_node.twz;
          for (int i = 0; i < 8; ++i) {
            node.rw = parent_node.rw * 0.5f;
            node.rx = parent_node.rx + node.rw * static_cast<float>(i & 1);  // 0b1
            node.ry = parent_node.ry
                + node.rw * static_cast<float>((i >> 1) & 1);
            node.rz = parent_node.rz
                + node.rw * static_cast<float>((i >> 2) & 1);
            queue_temp.push(node);  // Push node to queue
          }
        } else {
          node.rx = parent_node.rx;
          node.ry = parent_node.ry;
          node.rz = parent_node.rz;
          node.rw = parent_node.rw;
          for (int i = 0; i < 8; ++i) {  // Split over translation
            node.twx = parent_node.twx * 0.5f;
            node.twy = parent_node.twy * 0.5f;
            node.twz = parent_node.twz * 0.5f;
            node.tx = parent_node.tx + node.twx * static_cast<float>(i & 1);  // 0b1
            node.ty = parent_node.ty
                + node.twy * static_cast<float>((i >> 1) & 1);
            node.tz = parent_node.tz
                + node.twz * static_cast<float>((i >> 2) & 1);
            queue_temp.push(node);  // Push node to queue
          }
        }
      }
      initial_queue_ = queue_temp;
    }
  }
}

/*
 * BranchAndBound
 * Branch-and-bound over the transformation domain
 */
void GOSMA::BranchAndBound() {
  // Setup BB variables
  int num_iterations = 0;
  int refinement_ctr = 0;
  float zeta2 = zeta_ * zeta_;
  double iteration_duration = 0.0;
  std::priority_queue<Node> queue = initial_queue_;

  int class_num_components_2d_serialised[kMaxNumClasses];
  int class_num_components_3d_serialised[kMaxNumClasses];
  float log_class_weights_serialised[kMaxNumClasses];
  for (int i = 0; i < num_classes_; ++i) {
    class_num_components_2d_serialised[i] = class_num_components_2d_[i];
    class_num_components_3d_serialised[i] = class_num_components_3d_[i];
    log_class_weights_serialised[i] = log_class_weights_[i];
  }
  float mus_2d_serialised[kMaxNumComponents][3];
  float kappas_2d_serialised[kMaxNumComponents];
  float log_phis_on_z_kappas_2d_serialised[kMaxNumComponents];
  for (int i = 0; i < num_components_2d_; ++i) {
    mus_2d_serialised[i][0] = mus_2d_(i, 0);
    mus_2d_serialised[i][1] = mus_2d_(i, 1);
    mus_2d_serialised[i][2] = mus_2d_(i, 2);
    kappas_2d_serialised[i] = kappas_2d_(i);
    log_phis_on_z_kappas_2d_serialised[i] = log_phis_on_z_kappas_2d_(i);
  }
  float mus_3d_serialised[kMaxNumComponents][3];
  float variances_3d_serialised[kMaxNumComponents];
  float log_phis_3d_serialised[kMaxNumComponents];
  for (int i = 0; i < num_components_3d_; ++i) {
    mus_3d_serialised[i][0] = mus_3d_(i, 0);
    mus_3d_serialised[i][1] = mus_3d_(i, 1);
    mus_3d_serialised[i][2] = mus_3d_(i, 2);
    variances_3d_serialised[i] = variances_3d_(i);
    log_phis_3d_serialised[i] = log_phis_3d_(i);
  }

  // Allocate input array in host memory
  Node parent_nodes[kNumDevices][kNumConcurrentNodes];
  // Allocate output array in host memory
  Node nodes[kNumDevices * kNumConcurrentThreads];
  for (int device = 0; device < kNumDevices; ++device) {
    // Set GPU to use
    CudaErrorCheck(cudaSetDevice(device));
    // Allocate input arrays in device memory
    CudaErrorCheck(cudaMalloc(
        &d_parent_nodes_[device], sizeof(Node) * kNumConcurrentNodes));
    // Allocate output arrays in device memory
    CudaErrorCheck(cudaMalloc(
        &d_nodes_[device], sizeof(Node) * kNumConcurrentThreads));
    // Copy constants to device constant memory
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_num_components_2d, &num_components_2d_,sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_num_components_3d, &num_components_3d_, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_num_classes, &num_classes_, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_zeta2, &zeta2, sizeof(float)));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_l2_normaliser, &l2_normaliser_, sizeof(float)));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_min_linear_resolution, &min_linear_resolution_, sizeof(float)));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_min_angular_resolution, &min_angular_resolution_, sizeof(float)));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_class_num_components_2d, &class_num_components_2d_serialised,
        sizeof(int) * kMaxNumClasses));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_class_num_components_3d, &class_num_components_3d_serialised,
        sizeof(int) * kMaxNumClasses));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_log_class_weights, &log_class_weights_serialised,
        sizeof(float) * kMaxNumClasses));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_mus_2d, &mus_2d_serialised, sizeof(float) * kMaxNumComponents * 3));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_mus_3d, &mus_3d_serialised, sizeof(float) * kMaxNumComponents * 3));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_kappas_2d, &kappas_2d_serialised, sizeof(float) * kMaxNumComponents));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_variances_3d, &variances_3d_serialised,
        sizeof(float) * kMaxNumComponents));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_log_phis_on_z_kappas_2d, &log_phis_on_z_kappas_2d_serialised,
        sizeof(float) * kMaxNumComponents));
    CudaErrorCheck(cudaMemcpyToSymbol(
        c_log_phis_3d, &log_phis_3d_serialised,
        sizeof(float) * kMaxNumComponents));
  }

  // Explore the transformation domain until convergence
  float min_lb = min_fvalue_;
  auto clock_begin = std::chrono::steady_clock::now();
  while (1) {
    auto clock_begin_iteration = std::chrono::steady_clock::now();
    // If the queue is empty, all potentially optimal regions have been explored and discarded
    if (queue.empty()) {
      optimality_certificate_++;
      DEBUG_OUTPUT(std::fixed << "#" << std::setw(6) << num_iterations
          << ": Lower Bound: " << std::setprecision(6) << min_lb
          << ", Upper Bound: " << std::setprecision(6) << min_fvalue_
          << ", Queue Size: " << std::setw(10) << queue.size() << ", Duration: "
          << std::setprecision(6) << iteration_duration << "s");
      min_l2_distance_ = ComputeL2Distance(min_fvalue_);
      DEBUG_OUTPUT("L2 Distance: " << std::setprecision(6) << min_l2_distance_
          << ", Lower Bound: " << std::setprecision(6)
          << ComputeL2Distance(min_lb));
      break;
    }
    // If maximum duration exceeded, terminate with sub-optimality
    if (std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_begin).count()
        > kMaxDuration) {
      optimality_certificate_ = -3;
      min_l2_distance_ = ComputeL2Distance(min_fvalue_);
      std::cout << "Exceeds kMaxDuration, terminating with sub-optimality"
                << std::endl;
      break;
    }
    // If maximum queue size exceeded, terminate with sub-optimality
    if (queue.size() > kMaxQueueSize) {
      optimality_certificate_ = -3;
      min_l2_distance_ = ComputeL2Distance(min_fvalue_);
      std::cout << "Exceeds kMaxQueueSize, terminating with sub-optimality"
                << std::endl;
      break;
    }

    // Access kNumConcurrentNodes nodes with lowest lower bound and remove them from the queue
    // If fewer are available, fill with null nodes (branch=0)
    for (int device = 0; device < kNumDevices; ++device) {
      for (int i = 0; i < kNumConcurrentNodes; ++i) {
        if (queue.empty()) {
          Node discard_node;
          discard_node.lb = 0.0f;
          discard_node.ub = 0.0f;
          discard_node.branch = 0;  // Set branch flag to discard
          parent_nodes[device][i] = discard_node;
        } else {
          parent_nodes[device][i] = queue.top();
          queue.pop();
        }
      }
    }
    min_lb = parent_nodes[0][0].lb;

    DEBUG_OUTPUT(std::fixed << "#" << std::setw(6) << num_iterations
        << ": Lower Bound: " << std::setprecision(6) << min_lb
        << ", Upper Bound: " << std::setprecision(6) << min_fvalue_
        << ", Queue Size: " << std::setw(10) << queue.size()
        << ", Cuboid Widths: [" << std::setprecision(6)
        << parent_nodes[0][0].twx << ", " << parent_nodes[0][0].rw
        << "], Duration: " << std::setprecision(6) << iteration_duration
        << "s, #Refinements: " << refinement_ctr);

    // Exit if the function value is less than or equal to the lower bound plus epsilon (convergence)
    if (min_lb + epsilon_ >= min_fvalue_) {
      min_l2_distance_ = ComputeL2Distance(min_fvalue_);
      DEBUG_OUTPUT("L2 Distance: " << std::setprecision(6) << min_l2_distance_
          << ", Lower Bound: " << std::setprecision(6)
          << ComputeL2Distance(min_lb));
      optimality_certificate_++;
      break;
    }

    /*
     * On the GPU:
     * Branch: subdivide the parent nodes
     * Bound: compute the lower and upper bounds for each child node
     */
    for (int device = 0; device < kNumDevices; ++device) {
      CudaErrorCheck(cudaSetDevice(device));  // Set GPU to use
      CudaErrorCheck(cudaMemcpy(
          d_parent_nodes_[device], parent_nodes[device],
          sizeof(Node) * kNumConcurrentNodes, cudaMemcpyHostToDevice));  // Copy parent nodes to GPU
      GetBounds<<<kNumBlocks, kNumThreadsPerBlock>>>(
          d_parent_nodes_[device], d_nodes_[device]);
    }
    for (int device = 0; device < kNumDevices; ++device) {
      CudaErrorCheck(cudaSetDevice(device));  // Set GPU to use
      CudaErrorCheck(cudaPeekAtLastError());  // Check for kernel launch error
      CudaErrorCheck(cudaDeviceSynchronize());  // Check for kernel execution error

      CudaErrorCheck(cudaMemcpy(
          nodes + device * kNumConcurrentThreads, d_nodes_[device],
          sizeof(Node) * kNumConcurrentThreads, cudaMemcpyDeviceToHost));  // Copy results to host
    }
    // Find the new best upper bound
    int best_index = -1;
    float previous_min_fvalue = min_fvalue_;
    std::vector<int> node_indices;
    for (int i = 0; i < kNumDevices * kNumConcurrentThreads; ++i) {
      if (nodes[i].branch > 0 && nodes[i].ub < previous_min_fvalue) {
        node_indices.push_back(i);  // Store set of nodes that improved on the previous global upper bound
        if (nodes[i].ub < min_fvalue_) {
          min_fvalue_ = nodes[i].ub;
          best_index = i;
        }
      }
    }
    if (!node_indices.empty()) {
      // Update optimal rotation and translation with the best parameters
      optimal_translation_ = Eigen::RowVector3f(
          nodes[best_index].tx + 0.5f * nodes[best_index].twx,
          nodes[best_index].ty + 0.5f * nodes[best_index].twy,
          nodes[best_index].tz + 0.5f * nodes[best_index].twz);
      optimal_rotation_vector_ = Eigen::RowVector3f(
          nodes[best_index].rx + 0.5f * nodes[best_index].rw,
          nodes[best_index].ry + 0.5f * nodes[best_index].rw,
          nodes[best_index].rz + 0.5f * nodes[best_index].rw);
      optimal_rotation_ = RotationVectorToMatrix(optimal_rotation_vector_);
      DEBUG_OUTPUT("New Best Lower Bound: " << std::setprecision(6) << min_lb
          << ", Upper Bound: " << std::setprecision(6) << min_fvalue_);
      optimality_certificate_ = 0;  // If upper bound has been reduced, optimality is possible

      // Run refinement for all nodes that reduced the upper bound, capped at max_num_refinements
      int max_num_refinements = 100;
      if (node_indices.size() > max_num_refinements) {
        std::partial_sort(
            node_indices.begin(), node_indices.begin() + max_num_refinements,
            node_indices.end(),
            [&](int i, int j) {return nodes[i].ub < nodes[j].ub;});
        node_indices.erase(
            node_indices.begin() + max_num_refinements, node_indices.end());
      }
      for (std::vector<int>::iterator it = node_indices.begin();
          it != node_indices.end(); ++it) {
        refinement_ctr++;
        Eigen::RowVector3f initial_translation = Eigen::RowVector3f(
            nodes[*it].tx + 0.5f * nodes[*it].twx,
            nodes[*it].ty + 0.5f * nodes[*it].twy,
            nodes[*it].tz + 0.5f * nodes[*it].twz);
        Eigen::RowVector3f initial_rotation_vector = Eigen::RowVector3f(
            nodes[*it].rx + 0.5f * nodes[*it].rw,
            nodes[*it].ry + 0.5f * nodes[*it].rw,
            nodes[*it].rz + 0.5f * nodes[*it].rw);
        if (RunRefinement(initial_translation, initial_rotation_vector)) {  // May update min_fvalue_, optimal_translation_, optimal_rotation_
          DEBUG_OUTPUT("New Best Lower Bound: " << std::setprecision(6)
              << min_lb << ", Upper Bound: " << std::setprecision(6)
              << min_fvalue_ << " (REFINEMENT)");
        }
      }
    }

    // Add feasible nodes to the queue
    for (int i = 0; i < kNumDevices * kNumConcurrentThreads; ++i) {
      // If the lower bound is less than the best value found so far, add the node to the queue
      if (nodes[i].branch > 0 && nodes[i].lb + epsilon_ < min_fvalue_) {  // Only store cubes that need further exploration
        queue.push(nodes[i]);
      }
    }

    ++num_iterations;
    iteration_duration = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_begin_iteration).count();
  }
}

/*
 * Routines for mixture manipulation
 */

/*
 * ApplyTransformation
 * Inputs:
 * - node: transformation subcuboid
 * Outputs:
 */
void GOSMA::ApplyTransformation(const Node& node) {
  TranslateAndNormaliseMus3dByNode(node);  // mu3d - t
  RotateMus2dByNode(node);  // (R^-1)mu2d
}

/*
 * TranslateAndNormaliseMus3dByNode
 * Translate 3D mixture into the (rotated) camera coordinate frame (mu3d - t)
 * Inputs:
 * - node: transformation subcuboid
 * Outputs:
 */
void GOSMA::TranslateAndNormaliseMus3dByNode(const Node& node) {
  TranslateAndNormaliseMus3dByVector(
      Eigen::RowVector3f(node.tx + 0.5f * node.twx,
                         node.ty + 0.5f * node.twy,
                         node.tz + 0.5f * node.twz));
}

/*
 * TranslateAndNormaliseMus3dsByVector
 * Inputs:
 * - translation_vector: translation 3-vector
 * Outputs:
 */
void GOSMA::TranslateAndNormaliseMus3dByVector(
    const Eigen::RowVector3f& translation_vector) {
  mus_3d_translated_ = mus_3d_.rowwise() - translation_vector;
  mus_3d_translated_norm2_ = mus_3d_translated_.rowwise().squaredNorm();
  mus_3d_translated_normalised_ = mus_3d_translated_.array().colwise()
      / mus_3d_translated_norm2_.array().sqrt();
}

/*
 * RotateMus2dByNode
 * Rotate 2D mixture into the camera coordinate frame (R^-1)*mu2d
 * Inputs:
 * - node: transformation subcuboid
 * Outputs:
 */
void GOSMA::RotateMus2dByNode(const Node& node) {
  RotateMus2dByVector(
      Eigen::RowVector3f(node.rx + 0.5f * node.rw,
                         node.ry + 0.5f * node.rw,
                         node.rz + 0.5f * node.rw));
}

/*
 * RotateMus2dByVector
 * Inputs:
 * - rotation_vector: rotation angle-axis 3-vector
 * Outputs:
 */
void GOSMA::RotateMus2dByVector(const Eigen::RowVector3f& rotation_vector) {
  float rotation_angle = rotation_vector.norm();
  if (rotation_angle <= std::numeric_limits<float>::epsilon()) {
    mus_2d_rotated_ = mus_2d_;
  } else {
    RotateMus2dByMatrix(RotationVectorToMatrix(rotation_vector));
  }
}

/*
 * RotateMus2dByMatrix
 * Inputs:
 * - rotation_matrix: rotation matrix
 * Outputs:
 */
void GOSMA::RotateMus2dByMatrix(const Eigen::Matrix3f& rotation_matrix) {
  mus_2d_rotated_ = mus_2d_ * rotation_matrix;  // Not transpose - (R^-1)mu2d
}

/*
 * RotateMu2dByVector
 * Inputs:
 * - mu_2d: bearing 3-vector
 * - rotation_vector: rotation angle-axis 3-vector
 * Outputs:
 * - return_value: rotated bearing vector
 */
Eigen::RowVector3f GOSMA::RotateMu2dByVector(
    const Eigen::RowVector3f& mu_2d,
    const Eigen::RowVector3f& rotation_vector) {
  float angle = rotation_vector.norm();
  if (angle <= std::numeric_limits<float>::epsilon()) {  // Check for very small rotations
    return mu_2d;
  } else {
    return RotateMu2dByMatrix(mu_2d, RotationVectorToMatrix(rotation_vector));
  }
}

/*
 * RotateMu2dByMatrix
 * Inputs:
 * - mu_2d: bearing 3-vector
 * - rotation_matrix: rotation matrix
 * Outputs:
 * - return_value: rotated bearing vector
 */
Eigen::RowVector3f GOSMA::RotateMu2dByMatrix(
    const Eigen::RowVector3f& mu_2d, const Eigen::Matrix3f& rotation_matrix) {
  return mu_2d * rotation_matrix;  // (R^T)mu2d
}

/*
 * NodeToRotationMatrix
 * Inputs:
 * - node: transformation subcuboid
 * Outputs:
 * - return_value: rotation matrix corresponding to the vector at the centre of the rotation subcuboid
 */
Eigen::Matrix3f GOSMA::NodeToRotationMatrix(const Node& node) {
  Eigen::RowVector3f rotation_vector = Eigen::RowVector3f(
      node.rx + 0.5f * node.rw,
      node.ry + 0.5f * node.rw,
      node.rz + 0.5f * node.rw);
  return RotationVectorToMatrix(rotation_vector);
}

/*
 * RotationVectorToMatrix
 * Inputs:
 * - rotation_vector: rotation vector
 * Outputs:
 * - return_value: rotation matrix corresponding to the vector
 */
Eigen::Matrix3f GOSMA::RotationVectorToMatrix(
    const Eigen::RowVector3f& rotation_vector) {
  float rotation_angle = std::sqrt(
      rotation_vector[0] * rotation_vector[0]
      + rotation_vector[1] * rotation_vector[1]
      + rotation_vector[2] * rotation_vector[2]);
  if (rotation_angle <= std::numeric_limits<float>::epsilon()) {
    return Eigen::Matrix3f::Identity();
  } else {
    float v[3] = {
        rotation_vector[0] / rotation_angle,
        rotation_vector[1] / rotation_angle,
        rotation_vector[2] / rotation_angle };
    float ca = std::cos(rotation_angle);
    float ca2 = 1 - ca;
    float sa = std::sin(rotation_angle);
    float v0sa = v[0] * sa;
    float v1sa = v[1] * sa;
    float v2sa = v[2] * sa;
    float v0v1ca2 = v[0] * v[1] * ca2;
    float v0v2ca2 = v[0] * v[2] * ca2;
    float v1v2ca2 = v[1] * v[2] * ca2;
    Eigen::Matrix3f rotation_matrix;
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

/*
 * Utility functions
 */

/*
 * InitialValue
 * Initialise the optimal function value based on [the user-supplied value, RANSAC and PnP]
 */
void GOSMA::InitialValue() {
  // Calculate function value for first element in initial queue and update upper bound
  Node node = initial_queue_.top();
  initial_queue_.pop();
  ApplyTransformation(node);
  min_fvalue_ = ComputeFunctionValue();
  min_l2_distance_ = ComputeL2Distance(min_fvalue_);
  node.ub = min_fvalue_;
  optimal_translation_ = Eigen::RowVector3f(node.tx + 0.5f * node.twx,
                                            node.ty + 0.5f * node.twy,
                                            node.tz + 0.5f * node.twz);
  optimal_rotation_vector_ = Eigen::RowVector3f(node.rx + 0.5f * node.rw,
                                                node.ry + 0.5f * node.rw,
                                                node.rz + 0.5f * node.rw);
  optimal_rotation_ = RotationVectorToMatrix(optimal_rotation_vector_);
  initial_queue_.push(node);
  DEBUG_OUTPUT("L2 Distance: " << std::setprecision(6) << min_l2_distance_
      << " (INITIAL)");
  DEBUG_OUTPUT("#     0: Lower Bound: " << std::setprecision(6) << node.lb
      << ", Upper Bound: " << std::setprecision(6) << min_fvalue_
      << " (INITIAL)");
  // Run local optimisation to (potentially) find a better value
  if (RunRefinement(optimal_translation_, optimal_rotation_vector_)) {  // If improved, output
    DEBUG_OUTPUT("New Best Lower Bound: " << std::setprecision(6) << node.lb
        << ", Upper Bound: " << std::setprecision(6) << min_fvalue_
        << " (REFINEMENT)");
  }
}

/*
 * ComputeKappas3d
 * Computes kappa approximation for the 3D mixture
 * Assumes mus_3d_translated_norm2_ has already been updated
 */
void GOSMA::ComputeKappas3d() {
  kappas_3d_ = (mus_3d_translated_norm2_.array() / variances_3d_.array() + 1)
      .matrix();
}

void GOSMA::ComputeLogClassWeights() {
  for (auto w : class_weights_) {
    log_class_weights_.push_back(log(w));
  }
}

void GOSMA::ComputeLogPhis2d() {
  log_phis_2d_ = phis_2d_.array().log();
}

void GOSMA::ComputeLogPhis3d() {
  log_phis_3d_ = phis_3d_.array().log();
}

void GOSMA::ComputeLogPhisOnZKappas2d() {
  log_phis_on_z_kappas_2d_.resize(num_components_2d_, Eigen::NoChange);
  for (int i = 0; i < num_components_2d_; ++i) {
    log_phis_on_z_kappas_2d_[i] = log_phis_2d_[i] - LogZFunction(kappas_2d_[i]);
  }
}

void GOSMA::ComputeLogPhisOnZKappas3d() {
  log_phis_on_z_kappas_3d_.resize(num_components_3d_, Eigen::NoChange);
  for (int i = 0; i < num_components_3d_; ++i) {
    log_phis_on_z_kappas_3d_[i] = log_phis_3d_[i] - LogZFunction(kappas_3d_[i]);
  }
}

float GOSMA::ComputeL2Normaliser() {
  float max_norm_3d_2 = pow(10.0f, 2);
  int start_index_2d = 0;
  int stop_index_2d = 0;
  int start_index_3d = 0;
  int stop_index_3d = 0;
  float l2_normaliser = 0.0f;
  for (int c = 0; c < num_classes_; ++c) {
    stop_index_2d += class_num_components_2d_[c];
    stop_index_3d += class_num_components_3d_[c];
    for (int i = start_index_2d; i < stop_index_2d; ++i) {
      l2_normaliser += class_weights_[c] * 0.5f * phis_2d_(i) * phis_2d_(i)
          * kappas_2d_(i);
    }
    for (int i = start_index_3d; i < stop_index_3d; ++i) {
      l2_normaliser += class_weights_[c] * 0.5f * phis_3d_(i) * phis_3d_(i)
          * (max_norm_3d_2 / variances_3d_(i) + 1.0f);
    }
    start_index_2d = stop_index_2d;
    start_index_3d = stop_index_3d;
  }
  l2_normaliser_ = l2_normaliser;
  return l2_normaliser;
}

/*
 * ComputeL2Constant
 * Outputs:
 * - return value: constant component of the L2 distance between the transformed mixture models (f22)
 */
float GOSMA::ComputeL2Constant() {
  Eigen::MatrixX3f kappas_mus_2d = mus_2d_.array().colwise()
      * kappas_2d_.array();
  int start_index_2d = 0;
  int stop_index_2d = 0;
  float function_value = 0.0f;
  for (int c = 0; c < num_classes_; ++c) {
    stop_index_2d += class_num_components_2d_[c];
    for (int i = start_index_2d; i < stop_index_2d; ++i) {
      for (int j = start_index_2d; j < stop_index_2d; ++j) {
        float K2i2j = (kappas_mus_2d.row(i) + kappas_mus_2d.row(j)).norm();
        function_value += exp(
            log_class_weights_[c] + log_phis_on_z_kappas_2d_[i]
                + log_phis_on_z_kappas_2d_[j] + LogZFunction(K2i2j));
      }
    }
    start_index_2d = stop_index_2d;
  }
  l2_constant_ = function_value / l2_normaliser_;
  return l2_constant_;
}

/*
 * ComputeL2Distance
 * Outputs:
 * - return value: L2 distance between the transformed mixture models (f11 - 2*f12 + f22) / (2*pi)
 */
float GOSMA::ComputeL2Distance() {
  return ComputeL2Distance(ComputeFunctionValue());
}

/*
 * ComputeL2Distance
 * Inputs:
 * - fvalue: function value to convert to L2 distance
 * Outputs:
 * - return value: L2 distance between the transformed mixture models (f11 - 2*f12 + f22) / (2*pi)
 */
float GOSMA::ComputeL2Distance(const float fvalue) {
  return (fvalue + l2_constant_) / (2.0f * kPi);
}

/*
 * ComputeFunctionValue
 * Outputs:
 * - return value: variable component of the L2 distance between the transformed mixture models (f11 - 2*f12)
 */
float GOSMA::ComputeFunctionValue() {
  ComputeKappas3d();
  ComputeLogPhisOnZKappas3d();
  Eigen::MatrixX3f kappas_mus_2d_rotated =
      mus_2d_rotated_.array().colwise() * kappas_2d_.array();
  Eigen::MatrixX3f kappas_mus_3d_translated_normalised =
      mus_3d_translated_normalised_.array().colwise() * kappas_3d_.array();
  int start_index_2d = 0;
  int start_index_3d = 0;
  int stop_index_2d = 0;
  int stop_index_3d = 0;
  float function_value_11 = 0.0f;
  float function_value_12 = 0.0f;
  for (int c = 0; c < num_classes_; ++c) {
    stop_index_2d += class_num_components_2d_[c];
    stop_index_3d += class_num_components_3d_[c];
    for (int i = start_index_3d; i < stop_index_3d; ++i) {
      for (int j = start_index_3d; j < stop_index_3d; ++j) {
        float K1i1j = (kappas_mus_3d_translated_normalised.row(i)
            + kappas_mus_3d_translated_normalised.row(j)).norm();
        function_value_11 += exp(
            log_class_weights_[c] + log_phis_on_z_kappas_3d_[i]
                + log_phis_on_z_kappas_3d_[j] + LogZFunction(K1i1j));
      }
      for (int j = start_index_2d; j < stop_index_2d; ++j) {
        float K1i2j = (kappas_mus_3d_translated_normalised.row(i)
            + kappas_mus_2d_rotated.row(j)).norm();
        function_value_12 += exp(
            log_class_weights_[c] + log_phis_on_z_kappas_3d_[i]
                + log_phis_on_z_kappas_2d_[j] + LogZFunction(K1i2j));
      }
    }
    start_index_2d = stop_index_2d;
    start_index_3d = stop_index_3d;
  }
  return (function_value_11 - 2.0f * function_value_12) / l2_normaliser_;
}

/*
 * LogZFunction
 * Inputs:
 * - x: scalar value
 * Outputs:
 * - return value: log((exp(x) - exp(-x)) / x)
 */
float GOSMA::LogZFunction(float x) {
  if (fabs(x) < 1.0e-3f) {
    return log(2.0f);
  } else if (x < 10.0f) {
    return log(exp(x) - exp(-x)) - log(x);
  } else {  // equivalent for x >= 8 (float) or x >= 17 (double)
    return x - log(x);
  }
}

/*
 * RunRefinement
 * Run local optimisation to (potentially) find a better function value
 * Inputs:
 * - initial_translation:
 * - initial_rotation:
 * Outputs:
 * - return value: true if min_fvalue_ has been reduced
 */
bool GOSMA::RunRefinement(
    const Eigen::RowVector3f& initial_translation,
    const Eigen::RowVector3f& initial_rotation_vector) {
  Eigen::Matrix<double, 6, 1> x;
  x << initial_translation.transpose().cast<double>(),
      initial_rotation_vector.transpose().cast<double>();
  solver_.minimize(f_, x);

  // Compute function value using float type (for consistency)
  Eigen::Matrix<float, 6, 1> x_float = x.cast<float>();
  float fvalue = f_float_(x_float) / l2_normaliser_;
  if (fvalue < min_fvalue_) {
    min_fvalue_ = fvalue;
    optimal_translation_ = x_float.head(3).transpose();
    optimal_rotation_vector_ = x_float.tail(3).transpose();
    optimal_rotation_ = RotationVectorToMatrix(optimal_rotation_vector_);
    optimality_certificate_ = 0;  // If upper bound has been reduced, optimality is possible
    return true;
  } else {
    return false;
  }
}

void GOSMA::Clear() {
  // Free device memory
  for (int device = 0; device < kNumDevices; ++device) {
    CudaErrorCheck(cudaFree(d_nodes_[device]));
    CudaErrorCheck(cudaFree(d_parent_nodes_[device]));
  }
}
