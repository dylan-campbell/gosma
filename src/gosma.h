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

#ifndef GOSMA_H_
#define GOSMA_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <queue>
#include <vector>
#include <limits> // for std::numeric_limits<float>::epsilon()
#include <cfloat> // for FLT_EPSILON
#define EIGEN_NO_CUDA
#undef __CUDACC__
#include <Eigen/Dense>
#define __CUDACC__ 1

#include "spherical_mm_l2_distance.hpp"

// 6D Transformation Space Node (50 bytes)
struct Node {
  float tx, ty, tz;  // Minimum translation values of node
  float rx, ry, rz;  // Minimum rotation values of node
  float twx, twy, twz;  // Width of along each translation dimension
  float rw;  // Width of along each rotation dimension
  float ub, lb;  // Upper and lower bound of node (32 bits)
  unsigned short int branch;  // Flag indicating whether to discard node (0), or branch over translation (1) or rotation (2)
  friend bool operator <(const Node& node1, const Node& node2) {  // if node1 < node2, node 2 comes first in queue
    if (node1.lb != node2.lb) {
      return node1.lb > node2.lb;  // Search lowest lower bound first
    } else if (node1.ub != node2.ub) {
      return node1.ub > node2.ub;  // Search lowest upper bound
    } else {
      return node1.twx < node2.twx;  // Search greatest translation width
    }
  }
};

class GOSMA {
 public:
  static const int kMaxDuration;  // Maximum runtime duration (in seconds)
  static const int kMaxQueueSize;  // Maximum priority queue size
  static const int kNumDevices;
  static const int kNumConcurrentNodes;
  static const int kNumChildrenPerNode;
  static const int kNumConcurrentThreads;
  static const int kNumThreadsPerBlock;
  static const int kNumBlocks;
  static const int kMaxNumComponents;
  static const int kMaxNumClasses;
  static const float kPi;
  static const float kSquareRootThree;
  static const float kInvertedGoldenRatio;
  static const float kOneMinusInvertedGoldenRatio;
  static const float kToleranceSquared;

  // Constructors
  GOSMA();

  // Destructors
  ~GOSMA();

  // Accessors
  int num_components_2d() const {
    return num_components_2d_;
  }
  int num_components_3d() const {
    return num_components_3d_;
  }
  int num_classes() const {
    return num_classes_;
  }
  int optimality_certificate() const {
    return optimality_certificate_;
  }
  float epsilon() const {
    return epsilon_;
  }
  float zeta() const {
    return zeta_;
  }
  float min_linear_resolution() const {
    return min_linear_resolution_;
  }
  float min_angular_resolution() const {
    return min_angular_resolution_;
  }
  float translation_domain_expansion_factor() const {
    return translation_domain_expansion_factor_;
  }
  float min_l2_distance() const {
    return min_l2_distance_;
  }
  float min_fvalue() const {
    return min_fvalue_;
  }
  float l2_constant() const {
    return l2_constant_;
  }
  std::vector<int> class_num_components_2d() const {
    return class_num_components_2d_;
  }
  std::vector<int> class_num_components_3d() const {
    return class_num_components_3d_;
  }
  std::vector<float> class_weights() const {
    return class_weights_;
  }
  Eigen::MatrixX3f mus_2d() const {
    return mus_2d_;
  }
  Eigen::MatrixX3f mus_3d() const {
    return mus_3d_;
  }
  Eigen::VectorXf kappas_2d() const {
    return kappas_2d_;
  }
  Eigen::VectorXf kappas_3d() const {
    return kappas_2d_;
  }
  Eigen::VectorXf variances_3d() const {
    return variances_3d_;
  }
  Eigen::VectorXf phis_2d() const {
    return phis_2d_;
  }
  Eigen::VectorXf phis_3d() const {
    return phis_3d_;
  }
  Eigen::RowVector3f optimal_translation() const {
    return optimal_translation_;
  }
  Eigen::RowVector3f optimal_rotation_vector() const {
    return optimal_rotation_vector_;
  }
  Eigen::Matrix3f optimal_rotation() const {
    return optimal_rotation_;
  }
  Node initial_node() const {
    return initial_node_;
  }
  std::priority_queue<Node> initial_queue() const {
    return initial_queue_;
  }

  // Mutators
  void set_num_components_2d(int num_components_2d) {
    num_components_2d_ = num_components_2d;
  }
  void set_num_components_3d(int num_components_3d) {
    num_components_3d_ = num_components_3d;
  }
  void set_num_classes(int num_classes) {
    num_classes_ = num_classes;
  }
  void set_optimality_certificate(int optimality_certificate) {
    optimality_certificate_ = optimality_certificate;
  }
  void set_epsilon(float epsilon) {
    epsilon_ = epsilon;
  }
  void set_zeta(float zeta) {
    zeta_ = zeta;
  }
  void set_min_linear_resolution(float min_linear_resolution) {
    min_linear_resolution_ = min_linear_resolution;
  }
  void set_min_angular_resolution(float min_angular_resolution) {
    min_angular_resolution_ = min_angular_resolution;
  }
  void set_translation_domain_expansion_factor(
      float translation_domain_expansion_factor) {
    translation_domain_expansion_factor_ = translation_domain_expansion_factor;
  }
  void set_min_l2_distance(float min_l2_distance) {
    min_l2_distance_ = min_l2_distance;
  }
  void set_min_fvalue(float min_fvalue) {
    min_fvalue_ = min_fvalue;
  }
  void set_l2_constant(float l2_constant) {
    l2_constant_ = l2_constant;
  }
  void set_class_num_components_2d(std::vector<int> class_num_components_2d) {
    class_num_components_2d_ = class_num_components_2d;
  }
  void set_class_num_components_3d(std::vector<int> class_num_components_3d) {
    class_num_components_3d_ = class_num_components_3d;
  }
  void set_class_weights(std::vector<float> class_weights) {
    class_weights_ = class_weights;
  }
  void set_mus_2d(Eigen::MatrixX3f mus_2d) {
    mus_2d_ = mus_2d;
  }
  void set_mus_3d(Eigen::MatrixX3f mus_3d) {
    mus_3d_ = mus_3d;
  }
  void set_kappas_2d(Eigen::VectorXf kappas_2d) {
    kappas_2d_ = kappas_2d;
  }
  void set_kappas_3d(Eigen::VectorXf kappas_3d) {
    kappas_3d_ = kappas_3d;
  }
  void set_variances_3d(Eigen::VectorXf variances_3d) {
    variances_3d_ = variances_3d;
  }
  void set_phis_2d(Eigen::VectorXf phis_2d) {
    phis_2d_ = phis_2d;
  }
  void set_phis_3d(Eigen::VectorXf phis_3d) {
    phis_3d_ = phis_3d;
  }
  void set_initial_queue(std::priority_queue<Node>& initial_queue) {
    initial_queue_ = initial_queue;
  }
  void set_optimal_translation(Eigen::RowVector3f& optimal_translation) {
    optimal_translation_ = optimal_translation;
  }
  void set_optimal_rotation_vector(
      Eigen::RowVector3f& optimal_rotation_vector) {
    optimal_rotation_vector_ = optimal_rotation_vector;
  }
  void set_optimal_rotation(Eigen::Matrix3f& optimal_rotation) {
    optimal_rotation_ = optimal_rotation;
  }
  void set_initial_node(Node& initial_node) {
    initial_node_ = initial_node;
  }

  // Public Class Functions
  void Run();
  void RunRefinementOnly();

 private:
  // Private Class Functions
  bool CheckInputs();
  void Initialise();
  void BranchAndBound();
  void Clear();
  void ApplyTransformation(const Node& node);
  void TranslateAndNormaliseMus3dByNode(const Node& node);
  void TranslateAndNormaliseMus3dByVector(
      const Eigen::RowVector3f& translation_vector);
  void RotateMus2dByNode(const Node& node);
  void RotateMus2dByVector(const Eigen::RowVector3f& rotation_vector);
  void RotateMus2dByMatrix(const Eigen::Matrix3f& rotation_matrix);
  Eigen::RowVector3f RotateMu2dByVector(
      const Eigen::RowVector3f& mu_2d,
      const Eigen::RowVector3f& rotation_vector);
  Eigen::RowVector3f RotateMu2dByMatrix(
      const Eigen::RowVector3f& mu_2d, const Eigen::Matrix3f& rotation_matrix);
  Eigen::Matrix3f NodeToRotationMatrix(const Node& node);
  Eigen::Matrix3f RotationVectorToMatrix(
      const Eigen::RowVector3f& rotation_vector);
  void InitialValue();
  void ComputeKappas3d();
  void ComputeLogClassWeights();
  void ComputeLogPhis2d();
  void ComputeLogPhis3d();
  void ComputeLogPhisOnZKappas2d();
  void ComputeLogPhisOnZKappas3d();
  float ComputeL2Constant();
  float ComputeL2Normaliser();
  float ComputeL2Distance();
  float ComputeL2Distance(const float fvalue);
  float ComputeFunctionValue();
  float LogZFunction(float x);
  bool RunRefinement(
      const Eigen::RowVector3f& initial_translation,
      const Eigen::RowVector3f& initial_rotation_vector);

  int num_components_2d_;  // Total number of 2D components
  int num_components_3d_;  // Total number of 3D components
  int num_classes_;
  int optimality_certificate_;  // Flag set if optimality is guaranteed
  float epsilon_;  // Final L2 function value guaranteed to be within epsilon of the global minimum
  float zeta_;  // Minimum distance between the camera and the closest 3D Gaussian centre
  float min_linear_resolution_;  // Translation sub-cuboids that fit within a sphere of this radius are not branched
  float min_angular_resolution_;  // Rotation sub-cubes that fit within a sphere of this radius (in radians) are not branched
  float translation_domain_expansion_factor_;  // Factor by which the width of the translation domain is larger than than the max width of the axis-aligned bounding box
  float min_l2_distance_;  // Minimum L2 distance found so far
  float min_fvalue_;  // Minimum function value (~L2 distance) found so far
  float l2_normaliser_;
  float l2_constant_;
  std::vector<int> class_num_components_2d_;  // Vector of number of components in each class
  std::vector<int> class_num_components_3d_;  // Vector of number of components in each class
  std::vector<float> class_weights_;  // Vector of weights per class (proportional to the 2D area of the class)
  std::vector<float> log_class_weights_;
  Eigen::MatrixX3f mus_2d_;
  Eigen::MatrixX3f mus_2d_rotated_;
  Eigen::MatrixX3f mus_3d_;
  Eigen::MatrixX3f mus_3d_translated_;
  Eigen::MatrixX3f mus_3d_translated_normalised_;
  Eigen::VectorXf mus_3d_translated_norm2_;
  Eigen::VectorXf kappas_2d_;
  Eigen::VectorXf kappas_3d_;
  Eigen::VectorXf variances_3d_;
  Eigen::VectorXf phis_2d_;
  Eigen::VectorXf phis_3d_;
  Eigen::VectorXf log_phis_2d_;
  Eigen::VectorXf log_phis_3d_;
  Eigen::VectorXf log_phis_on_z_kappas_2d_;
  Eigen::VectorXf log_phis_on_z_kappas_3d_;
  Eigen::RowVector3f optimal_translation_;
  Eigen::RowVector3f optimal_rotation_vector_;
  Eigen::Matrix3f optimal_rotation_;
  Node initial_node_;
  std::vector<Node*> d_nodes_;
  std::vector<Node*> d_parent_nodes_;
  std::priority_queue<Node> initial_queue_;
  SphericalMML2Distance<double> f_;
  SphericalMML2Distance<float> f_float_;
  cppoptlib::LbfgsSolver<SphericalMML2Distance<double>> solver_;
};
#endif /* GOSMA_H_ */
