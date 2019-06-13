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

#include <iostream>
#include <fstream>
#include <chrono>
#include <exception>
#include <system_error>
#include "gosma.h"
#include "point_set.h"
#include "bearing_vector_set.h"
#include "config_map.h"
#include "dpgm.hpp"
#include "dpvmfm.hpp"

#define DEFAULT_POINT_SET_FNAME "point_set.txt"
#define DEFAULT_BEARING_VECTOR_SET_FNAME "bearing_vector_set.txt"
#define DEFAULT_CONFIG_FNAME "config.txt"
#define DEFAULT_OUTPUT_FNAME "output.txt"
#define DEFAULT_TRANSLATION_DOMAIN_FNAME "NONE"
#define DEFAULT_ROTATION_DOMAIN_FNAME "NONE"

using std::string;
using std::ofstream;

const bool kDoRefinement = false;
const bool kComputeLambda = false;
const bool kComputeLambdaPerClass = false; // Only used if kComputeLambda true
const float kDesiredNumComponents3d = 60; // Only used if kComputeLambda true
const float kDesiredNumComponents2d = 45; // Only used if kComputeLambda true
const float kDesiredNumComponentsPerClass3d = 11; // Only used if kComputeLambda true
const float kDesiredNumComponentsPerClass2d = 11; // Only used if kComputeLambda true

void ParseInput(
    int argc, char **argv, string& point_set_filename,
    string& bearing_vector_set_filename, string& config_filename,
    string& output_filename, string& translation_domain_filename,
    string& rotation_domain_filename);
void ReadConfig(string filename, GOSMA& gosma);
void ReadMMConfig(
    string filename, float& lambda_euclidean, float& lambda_spherical_degrees);
void ReadDomains(
    string translation_domain_filename, string rotation_domain_filename,
    std::priority_queue<Node>& initial_queue);
int LoadDomain(
    string filename, Eigen::Matrix<float, Eigen::Dynamic, 6>& domains);
int LoadClassCounts(
    string filename, std::vector<int>& class_labels,
    std::vector<int>& class_counts);
void ComputeLambdaEuclidean(
    int desired_num_components_3d, int num_classes,
    const std::vector<int>& class_num_points, const Eigen::MatrixX3f& points,
    float& lambda_euclidean);
void ComputeLambdaEuclideanPerClass(
    int desired_num_components, int num_classes,
    const std::vector<int>& class_num_points, const Eigen::MatrixX3f& points,
    const float& lambda_euclidean, std::vector<float>& class_lambdas);
void ComputeLambdaSphericalDegrees(
    int desired_num_components_2d, int num_classes,
    const std::vector<int>& class_num_bearing_vectors,
    const Eigen::MatrixX3f& bearing_vectors, float& lambda_spherical_degrees);
void ComputeLambdaSphericalDegreesPerClass(
    int desired_num_components, int num_classes,
    const std::vector<int>& class_num_bearing_vectors,
    const Eigen::MatrixX3f& bearing_vectors,
    const float& lambda_spherical_degrees, std::vector<float>& class_lambdas);
void PrintComponents(
    int num_components_3d, const Eigen::VectorXf& variances_3d,
    const Eigen::VectorXf& phis_3d, const Eigen::MatrixX3f& mus_3d,
    int num_components_2d, const Eigen::VectorXf& kappas_2d,
    const Eigen::VectorXf& phis_2d, const Eigen::MatrixX3f& mus_2d,
    const std::vector<float>& class_weights);

int main(int argc, char* argv[]) {
  // Parse user input
  string point_set_filename, bearing_vector_set_filename, config_filename;
  string output_filename, translation_domain_filename, rotation_domain_filename;
  ParseInput(
      argc, argv, point_set_filename, bearing_vector_set_filename,
      config_filename, output_filename, translation_domain_filename,
      rotation_domain_filename);

  // Load point-set and bearing vector set
  PointSet point_set;
  BearingVectorSet bearing_vector_set;
  if (point_set.Load(point_set_filename) <= 0) {
    std::cout << "Point-set file is empty or corrupted" << std::endl;
    return -1;
  }
  if (bearing_vector_set.Load(bearing_vector_set_filename) <= 0) {
    std::cout << "Bearing vector set file is empty or corrupted" << std::endl;
    return -1;
  }
  // Ensure bearing vectors are normalised
  //bearing_vector_set.NormaliseBearings();

  // Load class counts for the point-set and bearing vector set
  // If no files are present, a single class is assumed
  std::vector<int> class_labels_3d;
  std::vector<int> class_num_points;
  int num_classes_3d = LoadClassCounts(
      point_set_filename, class_labels_3d, class_num_points);
  if (num_classes_3d <= 0) {
    num_classes_3d = 1;
    class_labels_3d.push_back(0);
    class_num_points.push_back(point_set.num_points());
  }
  std::vector<int> class_labels_2d;
  std::vector<int> class_num_bearing_vectors;
  int num_classes_2d = LoadClassCounts(
      bearing_vector_set_filename, class_labels_2d, class_num_bearing_vectors);
  if (num_classes_2d <= 0) {
    num_classes_2d = 1;
    class_labels_2d.push_back(0);
    class_num_bearing_vectors.push_back(
        bearing_vector_set.num_bearing_vectors());
  }
  // Select classes for which are present in 2D and 3D
  // Check if already ordered
  std::vector<int> class_labels;
  bool is_ordered = true;
  if (num_classes_3d == num_classes_2d) {
    for (int i = 0; i < num_classes_2d; ++i) {
      class_labels.push_back(class_labels_2d[i]);
      if (class_labels_2d[i] != class_labels_3d[i]
          || class_num_bearing_vectors[i] <= 0 || class_num_points[i] <= 0) {
        is_ordered = false;
      }
    }
  } else {
    is_ordered = false;
  }
  // If unordered, rearrange data or discard as needed
  if (!is_ordered) {
    Eigen::MatrixX3f aligned_points;
    Eigen::MatrixX3f aligned_bearing_vectors;
    std::vector<int> aligned_class_num_points;
    std::vector<int> aligned_class_num_bearing_vectors;
    std::vector<int> aligned_class_labels;
    int start_index_3d = 0;
    for (int i = 0; i < num_classes_3d; ++i) {
      int start_index_2d = 0;
      for (int j = 0; j < num_classes_2d; ++j) {
        if (class_labels_3d[i] == class_labels_2d[j] && class_num_points[i] > 0
            && class_num_bearing_vectors[j] > 0) {
          int num_points = aligned_points.rows();
          aligned_points.conservativeResize(
              num_points + class_num_points[i], Eigen::NoChange);
          aligned_points.bottomRows(class_num_points[i]) =
              point_set.points().middleRows(start_index_3d,class_num_points[i]);
          aligned_class_num_points.push_back(class_num_points[i]);
          int num_bearing_vectors = aligned_bearing_vectors.rows();
          aligned_bearing_vectors.conservativeResize(
              num_bearing_vectors + class_num_bearing_vectors[j],
              Eigen::NoChange);
          aligned_bearing_vectors.bottomRows(class_num_bearing_vectors[j]) =
              bearing_vector_set.bearing_vectors().middleRows(
                  start_index_2d, class_num_bearing_vectors[j]);
          aligned_class_num_bearing_vectors.push_back(
              class_num_bearing_vectors[j]);
          aligned_class_labels.push_back(class_labels_3d[i]);
          break;
        }
        start_index_2d += class_num_bearing_vectors[j];
      }
      start_index_3d += class_num_points[i];
    }
    num_classes_3d = aligned_class_num_points.size();
    num_classes_2d = aligned_class_num_bearing_vectors.size();
    class_num_points = aligned_class_num_points;
    class_num_bearing_vectors = aligned_class_num_bearing_vectors;
    class_labels = aligned_class_labels;
    point_set.set_num_points(aligned_points.rows());
    point_set.set_points(aligned_points);
    bearing_vector_set.set_num_bearing_vectors(aligned_bearing_vectors.rows());
    bearing_vector_set.set_bearing_vectors(aligned_bearing_vectors);
  }
  if (num_classes_3d != num_classes_2d) {
    std::cout << "The number of classes in 2D and 3D must be equal"
              << std::endl;
    return -1;
  }
  int num_classes = num_classes_3d;

  // Compute the vector of weights per class
  std::vector<float> class_weights;
  for (int n : class_num_bearing_vectors) {
    class_weights.push_back(1.0f / static_cast<float>(num_classes));  // Use equal weights
  }

  // Construct Gaussian and von Mises-Fisher mixtures for each class
  // Read lambda values from file
  float lambda_euclidean;  // Euclidean lambda parameter; new cluster created when a point is > lambda from every existing cluster centroid
  float lambda_spherical_degrees;  // spherical lambda parameter in degrees; new cluster created when a point is > lambda from every existing cluster centroid
  ReadMMConfig(config_filename, lambda_euclidean, lambda_spherical_degrees);  // Reads from <config_filename>_mm.txt if present, otherwise reads from <config_filename>

  std::vector<float> class_lambdas_3d;
  std::vector<float> class_lambdas_2d;

  if (!kComputeLambda) {
    // Use input lambda values
    for (int c = 0; c < num_classes; ++c) {
      class_lambdas_3d.push_back(lambda_euclidean);
      class_lambdas_2d.push_back(lambda_spherical_degrees);
    }
  } else {
    if (!kComputeLambdaPerClass) {
      // Use fixed lambda for all classes (desired_num_components per dataset):
      int desired_num_components_3d = kDesiredNumComponents3d;
      int desired_num_components_2d = kDesiredNumComponents2d;
      ComputeLambdaEuclidean(
          desired_num_components_3d, num_classes, class_num_points,
          point_set.points(), lambda_euclidean);
      ComputeLambdaSphericalDegrees(
          desired_num_components_2d, num_classes, class_num_bearing_vectors,
          bearing_vector_set.bearing_vectors(), lambda_spherical_degrees);
      for (int c = 0; c < num_classes; ++c) {
        class_lambdas_3d.push_back(lambda_euclidean);
        class_lambdas_2d.push_back(lambda_spherical_degrees);
      }
    } else {
      // Use variable lambda for each classes (desired_num_components per class):
      int desired_num_components_3d = kDesiredNumComponentsPerClass3d;
      int desired_num_components_2d = kDesiredNumComponentsPerClass2d;
      ComputeLambdaEuclideanPerClass(
          desired_num_components_3d, num_classes, class_num_points,
          point_set.points(), lambda_euclidean, class_lambdas_3d);
      ComputeLambdaSphericalDegreesPerClass(
          desired_num_components_2d, num_classes, class_num_bearing_vectors,
          bearing_vector_set.bearing_vectors(), lambda_spherical_degrees,
          class_lambdas_2d);
    }
  }

  // Cap scale parameters to ensure a reasonable floating point computation regime
  float min_variance = 0.01f;  // Prevents zero variance for single element clusters
  float max_kappa = 1000.0f;  // Prevents infinite kappa for single element clusters

  int num_components_3d = 0;
  int num_components_2d = 0;
  std::vector<int> class_num_components_3d;
  std::vector<int> class_num_components_2d;
  Eigen::MatrixX3f mus_3d;
  Eigen::MatrixX3f mus_2d;
  Eigen::VectorXf variances_3d;
  Eigen::VectorXf kappas_2d;
  Eigen::VectorXf phis_3d;
  Eigen::VectorXf phis_2d;

  auto start_mm = std::chrono::steady_clock::now();

  int start_index_2d = 0;
  int start_index_3d = 0;
  for (int c = 0; c < num_classes; ++c) {
    Eigen::MatrixX3f class_points = point_set.points().middleRows(
        start_index_3d, class_num_points[c]);
    Eigen::MatrixX3f class_bearing_vectors =
        bearing_vector_set.bearing_vectors().middleRows(
            start_index_2d, class_num_bearing_vectors[c]);

    float lambda_3d = class_lambdas_3d[c];
    float lambda_euclidean2 = lambda_3d * lambda_3d;

    DPGM gm(class_points, lambda_euclidean2, min_variance);
    gm.Construct();

    // Append matrices/vectors
    num_components_3d += gm.num_components();
    class_num_components_3d.push_back(gm.num_components());
    mus_3d.conservativeResize(num_components_3d, Eigen::NoChange);
    variances_3d.conservativeResize(num_components_3d, Eigen::NoChange);
    phis_3d.conservativeResize(num_components_3d, Eigen::NoChange);
    mus_3d.bottomRows(gm.num_components()) = gm.mus();
    variances_3d.tail(gm.num_components()) = gm.variances();
    phis_3d.tail(gm.num_components()) = gm.phis();

    float lambda_2d = class_lambdas_2d[c];
    float lambda_spherical = std::cos(lambda_2d * GOSMA::kPi / 180.0f) - 1.0f;

    DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
    vmfm.Construct();

    num_components_2d += vmfm.num_components();
    class_num_components_2d.push_back(vmfm.num_components());
    mus_2d.conservativeResize(num_components_2d, Eigen::NoChange);
    kappas_2d.conservativeResize(num_components_2d, Eigen::NoChange);
    phis_2d.conservativeResize(num_components_2d, Eigen::NoChange);
    mus_2d.bottomRows(vmfm.num_components()) = vmfm.mus();
    kappas_2d.tail(vmfm.num_components()) = vmfm.kappas();
    phis_2d.tail(vmfm.num_components()) = vmfm.phis();

    start_index_3d += class_num_points[c];
    start_index_2d += class_num_bearing_vectors[c];
  }

  auto end_mm = std::chrono::steady_clock::now();
  double duration_mm = std::chrono::duration<double>(end_mm - start_mm).count();

//  PrintComponents(
//      num_components_3d, variances_3d, phis_3d, mus_3d, num_components_2d,
//      kappas_2d, phis_2d, mus_2d, class_weights);

  // Setup and run GOSMA
  GOSMA gosma;
  ReadConfig(config_filename, gosma);

  // Read translation and rotation domains if available
  std::priority_queue<Node> initial_queue;
  ReadDomains(
      translation_domain_filename, rotation_domain_filename, initial_queue);
  gosma.set_initial_queue(initial_queue);

  // Pass mixture components to GOSMA
  gosma.set_num_components_2d(num_components_2d);
  gosma.set_num_components_3d(num_components_3d);
  gosma.set_num_classes(num_classes);
  gosma.set_class_num_components_2d(class_num_components_2d);
  gosma.set_class_num_components_3d(class_num_components_3d);
  gosma.set_class_weights(class_weights);
  gosma.set_mus_2d(mus_2d);
  gosma.set_mus_3d(mus_3d);
  gosma.set_kappas_2d(kappas_2d);
  gosma.set_variances_3d(variances_3d);
  gosma.set_phis_2d(phis_2d);
  gosma.set_phis_3d(phis_3d);

  // Remove class information
  gosma.set_num_classes(1);
  class_num_components_2d.clear();
  class_num_components_2d.push_back(num_components_2d);
  gosma.set_class_num_components_2d(class_num_components_2d);
  class_num_components_3d.clear();
  class_num_components_3d.push_back(num_components_3d);
  gosma.set_class_num_components_3d(class_num_components_3d);
  class_weights.clear();
  class_weights.push_back(1.0f);
  gosma.set_class_weights(class_weights);

  // Run GOSMA
  std::cout << "Point-Set: " << point_set_filename << " ("
            << point_set.num_points() << " points)" << std::endl;
  std::cout << "Bearing Vector Set: " << bearing_vector_set_filename << " ("
            << bearing_vector_set.num_bearing_vectors() << " bearing vectors)"
            << std::endl;
  std::cout << "GMM: " << num_components_3d << " components, vMFMM: "
            << num_components_2d << " components" << std::endl << std::endl;
  std::cout << "Running GOSMA" << std::endl;
  auto start = std::chrono::steady_clock::now();
  gosma.Run();
  auto end = std::chrono::steady_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();

  // Outputs
  int optimality_certificate = gosma.optimality_certificate();
  float l2_distance = gosma.min_l2_distance();
  Eigen::RowVector3f translation = gosma.optimal_translation();
  Eigen::Matrix3f rotation = gosma.optimal_rotation();
  Eigen::RowVector3f rotation_vector = gosma.optimal_rotation_vector();

  // Print results
  std::cout << std::endl;
  std::cout << std::setfill(' ');
  std::cout << "L2 Distance: " << std::fixed << std::setprecision(6)
            << l2_distance << std::endl;
  std::cout << "Rotation Matrix:" << std::endl;
  std::cout << rotation << std::endl;
  std::cout << "Rotation Vector:" << std::endl;
  std::cout << rotation_vector << std::endl;
  std::cout << "Translation Vector:" << std::endl;
  std::cout << translation << std::endl;
  std::cout << "Duration: " << std::setprecision(6) << duration << "s (GOSMA), "
            << duration_mm << "s (MM GENERATION)" << std::endl;
  std::cout << "Optimality Certificate: " << optimality_certificate
            << std::endl;

  // Save results
  Eigen::RowVectorXf rotation_vectorised(
      Eigen::Map<Eigen::RowVectorXf>(rotation.data(), rotation.size()));  // Column-major mapping
  ofstream ofile(output_filename.c_str(), ofstream::out | ofstream::app);
  if (ofile.is_open()) {
    ofile << std::scientific << std::setprecision(9) << translation << " "
        << rotation_vectorised << " " << l2_distance << " " << duration << " "
        << duration_mm << " " << optimality_certificate << std::endl;  // Column-major storage
    ofile.close();
  } else {
    std::cout << "Cannot open output file" << std::endl;
    return -1;
  }

  /*
   * REFINEMENT
   */
  if (kDoRefinement) {
    // Generate higher resolution MMs
    if (!kComputeLambda) {
      // Use a fixed fraction of the input lambda values
      float lambda_fraction = 0.1f; // Use fixed lambda fraction
      for (int c = 0; c < num_classes; ++c) {
        class_lambdas_3d[c] *= lambda_fraction;
        class_lambdas_2d[c] *= lambda_fraction;
      }
    } else {
      if (!kComputeLambdaPerClass) {
        // Use fixed lambda for all classes (desired_num_components per dataset):
        int desired_num_components_3d = 10 * kDesiredNumComponents3d;
        int desired_num_components_2d = 10 * kDesiredNumComponents2d;
        class_lambdas_3d.clear();
        class_lambdas_2d.clear();
        lambda_euclidean /= 2.0; // start search at smaller lambda
        lambda_spherical_degrees /= 2.0; // start search at smaller lambda
        ComputeLambdaEuclidean(
            desired_num_components_3d, num_classes, class_num_points,
            point_set.points(), lambda_euclidean);
        ComputeLambdaSphericalDegrees(
            desired_num_components_2d, num_classes, class_num_bearing_vectors,
            bearing_vector_set.bearing_vectors(), lambda_spherical_degrees);
        for (int c = 0; c < num_classes; ++c) {
          class_lambdas_3d.push_back(lambda_euclidean);
          class_lambdas_2d.push_back(lambda_spherical_degrees);
        }
      } else {
        // Use variable lambda for each classes (desired_num_components per class):
        int desired_num_components_3d = 10 * kDesiredNumComponentsPerClass3d;
        int desired_num_components_2d = 10 * kDesiredNumComponentsPerClass2d;
        class_lambdas_3d.clear();
        class_lambdas_2d.clear();
        lambda_euclidean /= 2.0;  // start search at smaller lambda
        lambda_spherical_degrees /= 2.0;  // start search at smaller lambda
        ComputeLambdaEuclideanPerClass(
            desired_num_components_3d, num_classes, class_num_points,
            point_set.points(), lambda_euclidean, class_lambdas_3d);
        ComputeLambdaSphericalDegreesPerClass(
            desired_num_components_2d, num_classes, class_num_bearing_vectors,
            bearing_vector_set.bearing_vectors(), lambda_spherical_degrees,
            class_lambdas_2d);
      }
    }

    num_components_3d = 0;
    num_components_2d = 0;
    class_num_components_3d.clear();
    class_num_components_2d.clear();
    start_index_2d = 0;
    start_index_3d = 0;
    start_mm = std::chrono::steady_clock::now();
    for (int c = 0; c < num_classes; ++c) {
      Eigen::MatrixX3f class_points = point_set.points().middleRows(
          start_index_3d, class_num_points[c]);
      Eigen::MatrixX3f class_bearing_vectors =
          bearing_vector_set.bearing_vectors().middleRows(
              start_index_2d, class_num_bearing_vectors[c]);

      float lambda_3d = class_lambdas_3d[c];
      float lambda_euclidean2 = lambda_3d * lambda_3d;

      DPGM gm(class_points, lambda_euclidean2, min_variance);
      gm.Construct();

      // Append matrices/vectors
      num_components_3d += gm.num_components();
      class_num_components_3d.push_back(gm.num_components());
      mus_3d.conservativeResize(num_components_3d, Eigen::NoChange);
      variances_3d.conservativeResize(num_components_3d, Eigen::NoChange);
      phis_3d.conservativeResize(num_components_3d, Eigen::NoChange);
      mus_3d.bottomRows(gm.num_components()) = gm.mus();
      variances_3d.tail(gm.num_components()) = gm.variances();
      phis_3d.tail(gm.num_components()) = gm.phis();

      float lambda_2d = class_lambdas_2d[c];
      float lambda_spherical = std::cos(lambda_2d * GOSMA::kPi / 180.0f) - 1.0f;

      DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
      vmfm.Construct();

      num_components_2d += vmfm.num_components();
      class_num_components_2d.push_back(vmfm.num_components());
      mus_2d.conservativeResize(num_components_2d, Eigen::NoChange);
      kappas_2d.conservativeResize(num_components_2d, Eigen::NoChange);
      phis_2d.conservativeResize(num_components_2d, Eigen::NoChange);
      mus_2d.bottomRows(vmfm.num_components()) = vmfm.mus();
      kappas_2d.tail(vmfm.num_components()) = vmfm.kappas();
      phis_2d.tail(vmfm.num_components()) = vmfm.phis();

      start_index_3d += class_num_points[c];
      start_index_2d += class_num_bearing_vectors[c];
    }
    end_mm = std::chrono::steady_clock::now();
    duration_mm = std::chrono::duration<double>(end_mm - start_mm).count();

    // Set initial queue to be node found by GOSMA
    Node node;
    node.tx = translation(0);
    node.ty = translation(1);
    node.tz = translation(2);
    node.twx = 0.0;
    node.twy = 0.0;
    node.twz = 0.0;
    node.rx = rotation_vector(0);
    node.ry = rotation_vector(1);
    node.rz = rotation_vector(2);
    node.rw = 0.0;
    node.lb = -std::numeric_limits<float>::infinity();
    node.ub = std::numeric_limits<float>::infinity();
    node.branch = 0;
    initial_queue = std::priority_queue<Node>();
    initial_queue.push(node);
    gosma.set_initial_queue(initial_queue);

    // Pass mixture components to GOSMA
    gosma.set_num_components_2d(num_components_2d);
    gosma.set_num_components_3d(num_components_3d);
    gosma.set_num_classes(num_classes);
    gosma.set_class_num_components_2d(class_num_components_2d);
    gosma.set_class_num_components_3d(class_num_components_3d);
    gosma.set_class_weights(class_weights);
    gosma.set_mus_2d(mus_2d);
    gosma.set_mus_3d(mus_3d);
    gosma.set_kappas_2d(kappas_2d);
    gosma.set_variances_3d(variances_3d);
    gosma.set_phis_2d(phis_2d);
    gosma.set_phis_3d(phis_3d);

    // Remove class information
    gosma.set_num_classes(1);
    class_num_components_2d.clear();
    class_num_components_2d.push_back(num_components_2d);
    gosma.set_class_num_components_2d(class_num_components_2d);
    class_num_components_3d.clear();
    class_num_components_3d.push_back(num_components_3d);
    gosma.set_class_num_components_3d(class_num_components_3d);
    class_weights.clear();
    class_weights.push_back(1.0f);
    gosma.set_class_weights(class_weights);

    // Run GOSMA
    std::cout << "Point-Set: " << point_set_filename << " ("
              << point_set.num_points() << " points)" << std::endl;
    std::cout << "Bearing Vector Set: " << bearing_vector_set_filename << " ("
              << bearing_vector_set.num_bearing_vectors() << " bearing vectors)"
              << std::endl;
    std::cout << "GMM: " << num_components_3d << " components, vMFMM: "
              << num_components_2d << " components" << std::endl << std::endl;
    std::cout << "Running GOSMA Refinement" << std::endl;
    start = std::chrono::steady_clock::now();
    gosma.RunRefinementOnly();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>(end - start).count();

    // Outputs
    optimality_certificate = 0;
    l2_distance = gosma.min_l2_distance();
    translation = gosma.optimal_translation();
    rotation = gosma.optimal_rotation();
    rotation_vector = gosma.optimal_rotation_vector();

    // Print results
    std::cout << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "L2 Distance: " << std::fixed << std::setprecision(6)
              << l2_distance << std::endl;
    std::cout << "Rotation Matrix:" << std::endl;
    std::cout << rotation << std::endl;
    std::cout << "Rotation Vector:" << std::endl;
    std::cout << rotation_vector << std::endl;
    std::cout << "Translation Vector:" << std::endl;
    std::cout << translation << std::endl;
    std::cout << "Duration: " << std::setprecision(6) << duration
              << "s (REFINEMENT), " << duration_mm << "s (MM GENERATION)"
              << std::endl;
    std::cout << "Optimality Certificate: " << optimality_certificate
              << std::endl;

    // Save results
    if (output_filename.substr(output_filename.length() - 4) == ".txt") {
      output_filename.erase(output_filename.length() - 4, 4);
    }
    output_filename += "_refinement.txt";
    rotation_vectorised = Eigen::Map<Eigen::RowVectorXf>(
        rotation.data(), rotation.size());  // Column-major mapping
    ofstream ofile_refinement(
        output_filename.c_str(), ofstream::out | ofstream::app);
    if (ofile_refinement.is_open()) {
      ofile_refinement << std::scientific << std::setprecision(9) << translation
          << " " << rotation_vectorised << " " << l2_distance << " " << duration
          << " " << duration_mm << " " << optimality_certificate << std::endl;  // Column-major storage
      ofile_refinement.close();
    } else {
      std::cout << "Cannot open output file" << std::endl;
      return -1;
    }
  }

  return 0;
}

void ParseInput(
    int argc, char **argv, string& point_set_filename,
    string& bearing_vector_set_filename, string& config_filename,
    string& output_filename, string& translation_domain_filename,
    string& rotation_domain_filename) {
  // Set default values
  point_set_filename = DEFAULT_POINT_SET_FNAME;
  bearing_vector_set_filename = DEFAULT_BEARING_VECTOR_SET_FNAME;
  config_filename = DEFAULT_CONFIG_FNAME;
  output_filename = DEFAULT_OUTPUT_FNAME;
  translation_domain_filename = DEFAULT_TRANSLATION_DOMAIN_FNAME;
  rotation_domain_filename = DEFAULT_ROTATION_DOMAIN_FNAME;

  // Parse input
  if (argc > 6)
    rotation_domain_filename = argv[6];
  if (argc > 5)
    translation_domain_filename = argv[5];
  if (argc > 4)
    output_filename = argv[4];
  if (argc > 3)
    config_filename = argv[3];
  if (argc > 2)
    bearing_vector_set_filename = argv[2];
  if (argc > 1)
    point_set_filename = argv[1];

  // Print to screen
  std::cout << std::endl;
  for (int i = 0; i < argc; ++i)
    std::cout << argv[i] << " ";
  std::cout << std::endl << std::endl;
  std::cout << "Globally-Optimal Spherical Mixture Alignment (GOSMA)";
  std::cout << std::endl;
  std::cout << "Copyright (c) 2019 Dylan John Campbell";
  std::cout << std::endl;
  std::cout << "USAGE: " << argv[0];
  std::cout << " POINT-SET_FILENAME BEARING-VECTOR-SET_FILENAME CONFIG_FILENAME";
  std::cout << std::endl;
  std::cout << "       OUTPUT_FILENAME [TRANSLATION_DOMAIN_FILENAME";
  std::cout << " [ROTATION_DOMAIN_FILENAME]]";
  std::cout << std::endl;
  std::cout << "OUTPUT:";
  std::cout << std::endl;
  std::cout << "[translation vector | rotation matrix | L2 distance |";
  std::cout << " duration | optimality certificate]";
  std::cout << std::endl;
  std::cout << "(transformation from bearing-vector set to point-set)";
  std::cout << std::endl << std::endl;
  std::cout << "INPUT:";
  std::cout << std::endl;
  std::cout << "(point_set_filename)->(" << point_set_filename << ")";
  std::cout << std::endl;
  std::cout << "(bearing_vector_set_filename)->(" << bearing_vector_set_filename
            << ")";
  std::cout << std::endl;
  std::cout << "(config_filename)->(" << config_filename << ")";
  std::cout << std::endl;
  std::cout << "(output_filename)->(" << output_filename << ")";
  std::cout << std::endl;
  std::cout << "(translation_domain_filename)->(" << translation_domain_filename
            << ")";
  std::cout << std::endl;
  std::cout << "(rotation_domain_filename)->(" << rotation_domain_filename
            << ")";
  std::cout << std::endl << std::endl;
}

void ReadConfig(string filename, GOSMA& gosma) {
  // Open and parse the associated config file
  ConfigMap config(filename.c_str());
  gosma.set_epsilon(config.GetF("epsilon"));
  gosma.set_zeta(config.GetF("zeta"));
  gosma.set_min_linear_resolution(config.GetF("min_linear_resolution"));
  gosma.set_min_angular_resolution(config.GetF("min_angular_resolution"));
  gosma.set_translation_domain_expansion_factor(
      config.GetF("translation_domain_expansion_factor"));
  // Print settings
  std::cout << "CONFIG:" << std::endl;
  config.Print();
  std::cout << std::endl;
}

/*
 * ReadMMConfig
 * Reads from <filename>_mm.txt if present, otherwise reads from <filename>
 */
void ReadMMConfig(
    string filename, float& lambda_euclidean, float& lambda_spherical_degrees) {
  // Open and parse the associated config file
  string mm_filename = filename;
  if (mm_filename.substr(mm_filename.length() - 4) == ".txt") {
    mm_filename.erase(mm_filename.length() - 4, 4);
  }
  mm_filename += "_mm.txt";
  std::ifstream fin(mm_filename);
  if (fin.is_open()) {
    fin.close();
    filename = mm_filename;
  }
  ConfigMap config(filename.c_str());
  lambda_euclidean = config.GetF("lambda_euclidean");
  lambda_spherical_degrees = config.GetF("lambda_spherical_degrees");
}

/*
 * Load whitespace-separated translation and rotation domains from files
 * Determines number of input cuboids automatically
 * Required format: x y z wx wy wz (coordinates of min vertex and side-widths)
 */
void ReadDomains(
    string translation_domain_filename, string rotation_domain_filename,
    std::priority_queue<Node>& initial_queue) {
  int num_translation_domains = 0;
  Eigen::Matrix<float, Eigen::Dynamic, 6> translation_domains;
  if (translation_domain_filename != DEFAULT_TRANSLATION_DOMAIN_FNAME) {  // Only run if file supplied
    // Read file and push initial translation nodes to queue
    num_translation_domains = LoadDomain(
        translation_domain_filename, translation_domains);
  }
  int num_rotation_domains = 0;
  Eigen::Matrix<float, Eigen::Dynamic, 6> rotation_domains;
  if (rotation_domain_filename != DEFAULT_ROTATION_DOMAIN_FNAME) {  // Only run if file supplied
    // Read file and push initial rotation nodes to queue
    num_rotation_domains = LoadDomain(
        rotation_domain_filename, rotation_domains);
  }
  // Setup default node
  Node node;
  node.tx = -0.5;
  node.ty = -0.5;
  node.tz = -0.5;
  node.twx = 1.0;
  node.twy = 1.0;
  node.twz = 1.0;
  node.rx = -GOSMA::kPi;
  node.ry = -GOSMA::kPi;
  node.rz = -GOSMA::kPi;
  node.rw = 2.0 * GOSMA::kPi;
  node.lb = -std::numeric_limits<float>::infinity();
  node.ub = std::numeric_limits<float>::infinity();
  node.branch = 1;
  if (num_translation_domains > 0) {
    for (int i = 0; i < num_translation_domains; ++i) {
      node.tx = translation_domains(i, 0);
      node.ty = translation_domains(i, 1);
      node.tz = translation_domains(i, 2);
      node.twx = translation_domains(i, 3);
      node.twy = translation_domains(i, 4);
      node.twz = translation_domains(i, 5);
      if (num_rotation_domains > 0) {  // Both a translation and a rotation domain were specified
        for (int j = 0; j < num_rotation_domains; ++j) {
          node.rx = rotation_domains(j, 0);
          node.ry = rotation_domains(j, 1);
          node.rz = rotation_domains(j, 2);
          node.rw = rotation_domains(j, 3);
          initial_queue.push(node);
        }
      } else {  // Only a translation domain was specified
        initial_queue.push(node);
      }
    }
  } else {
    if (num_rotation_domains > 0) {  // Only a rotation domain was specified
      for (int j = 0; j < num_rotation_domains; ++j) {
        node.rx = rotation_domains(j, 0);
        node.ry = rotation_domains(j, 1);
        node.rz = rotation_domains(j, 2);
        node.rw = rotation_domains(j, 3);
        initial_queue.push(node);
      }
    }
  }
}

/*
 * Load whitespace-separated translation or rotation domains from a file
 * Determines number of input cuboids automatically
 * Required format: x y z wx wy wz (coordinates of min vertex and side-widths)
 */
int LoadDomain(
    string filename, Eigen::Matrix<float, Eigen::Dynamic, 6>& domains) {
  int num_domains = 0;
  int num_parameters = 6;
  std::string line, s;
  std::istringstream ss;
  std::ifstream ifile(filename.c_str(), std::ios_base::in);
  if (ifile.is_open()) {
    // Ascertain the number of rows
    num_domains = std::count(
        std::istreambuf_iterator<char>(ifile), std::istreambuf_iterator<char>(),
        '\n');  // Includes newlines at end of document
    ifile.seekg(0);  // Return to beginning of file
    domains.resize(num_domains, num_parameters);
    for (int i = 0; i < num_domains; ++i) {
      for (int j = 0; j < num_parameters; ++j) {
        ifile >> domains(i, j);
      }
    }
    ifile.close();
  } else {
    std::cout << "Unable to open domain file '" << filename << "'" << std::endl;
  }
  return num_domains;
}

/*
 * Load class counts
 * Assumes that class files, if present, are labelled the same as the data files
 * except with "_class_counts.txt" appended
 * Format: <label number>
 */
int LoadClassCounts(
    string filename, std::vector<int>& class_labels,
    std::vector<int>& class_counts) {
  int num_classes = 0;
  int label, count;
  if (filename.substr(filename.length() - 4) == ".txt") {
    filename.erase(filename.length() - 4, 4);
  }
  filename += "_class_counts.txt";
  std::string line, s;
  std::istringstream ss;
  std::ifstream ifile(filename.c_str(), std::ios_base::in);
  if (ifile.is_open()) {
    // Ascertain the number of rows
    num_classes = std::count(
        std::istreambuf_iterator<char>(ifile),
        std::istreambuf_iterator<char>(), '\n');  // Includes newlines at end of document
    ifile.seekg(0);  // Return to beginning of file
    for (int i = 0; i < num_classes; ++i) {
      ifile >> label;
      class_labels.push_back(label);
      ifile >> count;
      class_counts.push_back(count);
    }
    ifile.close();
  }
  return num_classes;
}

/*
 * Compute Lambda Euclidean
 * Computes the value of lambda_euclidean required to generate approximately
 * a number of mixture components equal to desired_num_components_3d
 */
void ComputeLambdaEuclidean(
    int desired_num_components_3d, int num_classes,
    const std::vector<int>& class_num_points, const Eigen::MatrixX3f& points,
    float& lambda_euclidean) {
  float lambda_euclidean2 = lambda_euclidean * lambda_euclidean;  // squared Euclidean distance
  float min_variance = 0.01f;
  int num_components_3d = 0;
  int start_index_3d = 0;
  for (int c = 0; c < num_classes; ++c) {
    Eigen::MatrixX3f class_points = points.middleRows(
        start_index_3d, class_num_points[c]);
    DPGM gm(class_points, lambda_euclidean2, min_variance);
    gm.Construct();
    num_components_3d += gm.num_components();
    start_index_3d += class_num_points[c];
  }
  if (num_components_3d < desired_num_components_3d) {
    float lambda_euclidean_temp = lambda_euclidean;
    while (num_components_3d < desired_num_components_3d) {
      if (lambda_euclidean_temp <= 0.0f)
        break;
      lambda_euclidean_temp -= 0.05f;  // Reduce lambda
      lambda_euclidean2 = lambda_euclidean_temp * lambda_euclidean_temp;
      num_components_3d = 0;
      start_index_3d = 0;
      for (int c = 0; c < num_classes; ++c) {
        Eigen::MatrixX3f class_points = points.middleRows(start_index_3d,
                                                          class_num_points[c]);
        DPGM gm(class_points, lambda_euclidean2, min_variance);
        gm.Construct();
        num_components_3d += gm.num_components();
        start_index_3d += class_num_points[c];
      }
    }
    lambda_euclidean = lambda_euclidean_temp;
  } else if (num_components_3d > desired_num_components_3d) {
    float lambda_euclidean_temp = lambda_euclidean;
    while (num_components_3d > desired_num_components_3d) {
      if (lambda_euclidean_temp >= 100.0f)
        break;
      lambda_euclidean_temp += 0.05f;  // Increase lambda
      lambda_euclidean2 = lambda_euclidean_temp * lambda_euclidean_temp;
      num_components_3d = 0;
      start_index_3d = 0;
      for (int c = 0; c < num_classes; ++c) {
        Eigen::MatrixX3f class_points = points.middleRows(
            start_index_3d, class_num_points[c]);
        DPGM gm(class_points, lambda_euclidean2, min_variance);
        gm.Construct();
        num_components_3d += gm.num_components();
        start_index_3d += class_num_points[c];
      }
    }
    lambda_euclidean = lambda_euclidean_temp;
  }
//  std::cout << "lambda_euclidean: " << lambda_euclidean
//            << ", num_components_3d: " << num_components_3d << std::endl;
}

/*
 * Compute Lambda Euclidean Per Class
 * Computes the values of lambda_euclidean required to generate approximately
 * a number of mixture components equal to desired_num_components per class
 */
void ComputeLambdaEuclideanPerClass(
    int desired_num_components, int num_classes,
    const std::vector<int>& class_num_points, const Eigen::MatrixX3f& points,
    const float& lambda_euclidean, std::vector<float>& class_lambdas) {
  float min_variance = 0.01f;
  int num_components = 0;
  int prev_num_components = 0;
  int start_index = 0;
  for (int c = 0; c < num_classes; ++c) {
    Eigen::MatrixX3f class_points = points.middleRows(
        start_index, class_num_points[c]);
    float lambda_euclidean2 = lambda_euclidean * lambda_euclidean;  // squared Euclidean distance
    DPGM gm(class_points, lambda_euclidean2, min_variance);
    gm.Construct();
    num_components = gm.num_components();
    if (num_components < desired_num_components) {
      float lambda_euclidean_temp = lambda_euclidean;
      while (num_components < desired_num_components) {
        if (lambda_euclidean_temp <= 0.0f)
          break;
        prev_num_components = num_components;
        lambda_euclidean_temp -= 0.05f;  // Reduce lambda
        lambda_euclidean2 = lambda_euclidean_temp * lambda_euclidean_temp;
        DPGM gm(class_points, lambda_euclidean2, min_variance);
        gm.Construct();
        num_components = gm.num_components();
      }
      if (abs(prev_num_components - desired_num_components)
          < abs(num_components - desired_num_components)) {
        lambda_euclidean_temp += 0.05f;  // Increase lambda
        lambda_euclidean2 = lambda_euclidean_temp * lambda_euclidean_temp;
        DPGM gm(class_points, lambda_euclidean2, min_variance);
        gm.Construct();
        num_components = gm.num_components();
      }
      class_lambdas.push_back(lambda_euclidean_temp);
    } else if (num_components > desired_num_components) {
      float lambda_euclidean_temp = lambda_euclidean;
      while (num_components > desired_num_components) {
        if (lambda_euclidean_temp >= 100.0f)
          break;
        prev_num_components = num_components;
        lambda_euclidean_temp += 0.05f;  // Increase lambda
        lambda_euclidean2 = lambda_euclidean_temp * lambda_euclidean_temp;
        DPGM gm(class_points, lambda_euclidean2, min_variance);
        gm.Construct();
        num_components = gm.num_components();
      }
      if (abs(prev_num_components - desired_num_components)
          < abs(num_components - desired_num_components)) {
        lambda_euclidean_temp -= 0.05f;  // Reduce lambda
        lambda_euclidean2 = lambda_euclidean_temp * lambda_euclidean_temp;
        DPGM gm(class_points, lambda_euclidean2, min_variance);
        gm.Construct();
        num_components = gm.num_components();
      }
      class_lambdas.push_back(lambda_euclidean_temp);
    } else {
      class_lambdas.push_back(lambda_euclidean);
    }
    start_index += class_num_points[c];
//    std::cout << "lambda_euclidean: " << class_lambdas.back()
//              << ", num_components: " << num_components
//              << ", desired_num_components: " << desired_num_components
//              << std::endl;
  }
}

/*
 * Compute Lambda Spherical Degrees
 * Computes the value of lambda_spherical_degrees required to generate
 * approximately a number of mixture components equal to
 * desired_num_components_2d
 */
void ComputeLambdaSphericalDegrees(
    int desired_num_components_2d, int num_classes,
    const std::vector<int>& class_num_bearing_vectors,
    const Eigen::MatrixX3f& bearing_vectors, float& lambda_spherical_degrees) {
  float lambda_spherical = std::cos(
      lambda_spherical_degrees * GOSMA::kPi / 180.0f) - 1.0f;
  float max_kappa = 1000.0f;
  int num_components_2d = 0;
  int start_index_2d = 0;
  for (int c = 0; c < num_classes; ++c) {
    Eigen::MatrixX3f class_bearing_vectors = bearing_vectors.middleRows(
        start_index_2d, class_num_bearing_vectors[c]);
    DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
    vmfm.Construct();
    num_components_2d += vmfm.num_components();
    start_index_2d += class_num_bearing_vectors[c];
  }
  if (num_components_2d < desired_num_components_2d) {
    float lambda_spherical_degrees_temp = lambda_spherical_degrees;
    while (num_components_2d < desired_num_components_2d) {
      if (lambda_spherical_degrees_temp <= 0.0f)
        break;
      lambda_spherical_degrees_temp -= 0.5f;  // Reduce lambda
      lambda_spherical = std::cos(
          lambda_spherical_degrees_temp * GOSMA::kPi / 180.0f) - 1.0f;
      num_components_2d = 0;
      start_index_2d = 0;
      for (int c = 0; c < num_classes; ++c) {
        Eigen::MatrixX3f class_bearing_vectors = bearing_vectors.middleRows(
            start_index_2d, class_num_bearing_vectors[c]);
        DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
        vmfm.Construct();
        num_components_2d += vmfm.num_components();
        start_index_2d += class_num_bearing_vectors[c];
      }
    }
    lambda_spherical_degrees = lambda_spherical_degrees_temp;
  } else if (num_components_2d > desired_num_components_2d) {
    float lambda_spherical_degrees_temp = lambda_spherical_degrees;
    while (num_components_2d > desired_num_components_2d) {
      if (lambda_spherical_degrees_temp >= 180.0f)
        break;
      lambda_spherical_degrees_temp += 0.5f;  // Increase lambda
      lambda_spherical = std::cos(
          lambda_spherical_degrees_temp * GOSMA::kPi / 180.0f) - 1.0f;
      num_components_2d = 0;
      start_index_2d = 0;
      for (int c = 0; c < num_classes; ++c) {
        Eigen::MatrixX3f class_bearing_vectors = bearing_vectors.middleRows(
            start_index_2d, class_num_bearing_vectors[c]);
        DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
        vmfm.Construct();
        num_components_2d += vmfm.num_components();
        start_index_2d += class_num_bearing_vectors[c];
      }
    }
    lambda_spherical_degrees = lambda_spherical_degrees_temp;
  }
//  std::cout << "lambda_spherical_degrees: " << lambda_spherical_degrees
//            << ", num_components_2d: " << num_components_2d << std::endl
//            << std::endl;
}

/*
 * Compute Lambda Spherical Degrees Per Class
 * Computes the values of lambda_spherical_degrees required to generate
 * approximately a number of mixture components equal to desired_num_components
 * per class
 */
void ComputeLambdaSphericalDegreesPerClass(
    int desired_num_components, int num_classes,
    const std::vector<int>& class_num_bearing_vectors,
    const Eigen::MatrixX3f& bearing_vectors,
    const float& lambda_spherical_degrees, std::vector<float>& class_lambdas) {
  float max_kappa = 1000.0f;
  int num_components = 0;
  int prev_num_components = 0;
  int start_index = 0;
  for (int c = 0; c < num_classes; ++c) {
    Eigen::MatrixX3f class_bearing_vectors = bearing_vectors.middleRows(
        start_index, class_num_bearing_vectors[c]);
    float lambda_spherical = std::cos(
        lambda_spherical_degrees * GOSMA::kPi / 180.0f) - 1.0f;
    DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
    vmfm.Construct();
    num_components = vmfm.num_components();
    if (num_components < desired_num_components) {
      float lambda_spherical_degrees_temp = lambda_spherical_degrees;
      while (num_components < desired_num_components) {
        if (lambda_spherical_degrees_temp <= 0.0f)
          break;
        prev_num_components = num_components;
        lambda_spherical_degrees_temp -= 0.5f;  // Reduce lambda
        lambda_spherical = std::cos(
            lambda_spherical_degrees_temp * GOSMA::kPi / 180.0f) - 1.0f;
        DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
        vmfm.Construct();
        num_components = vmfm.num_components();
      }
      if (abs(prev_num_components - desired_num_components)
          < abs(num_components - desired_num_components)) {
        lambda_spherical_degrees_temp += 0.5f;  // Increase lambda
        lambda_spherical = std::cos(
            lambda_spherical_degrees_temp * GOSMA::kPi / 180.0f) - 1.0f;
        DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
        vmfm.Construct();
        num_components = vmfm.num_components();
      }
      class_lambdas.push_back(lambda_spherical_degrees_temp);
    } else if (num_components > desired_num_components) {
      float lambda_spherical_degrees_temp = lambda_spherical_degrees;
      while (num_components > desired_num_components) {
        if (lambda_spherical_degrees_temp >= 180.0f)
          break;
        prev_num_components = num_components;
        lambda_spherical_degrees_temp += 0.5f;  // Increase lambda
        lambda_spherical = std::cos(
            lambda_spherical_degrees_temp * GOSMA::kPi / 180.0f) - 1.0f;
        DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
        vmfm.Construct();
        num_components = vmfm.num_components();
      }
      if (abs(prev_num_components - desired_num_components)
          < abs(num_components - desired_num_components)) {
        lambda_spherical_degrees_temp -= 0.5f;  // Reduce lambda
        lambda_spherical = std::cos(
            lambda_spherical_degrees_temp * GOSMA::kPi / 180.0f) - 1.0f;
        DPVMFM vmfm(class_bearing_vectors, lambda_spherical, max_kappa);
        vmfm.Construct();
        num_components = vmfm.num_components();
      }
      class_lambdas.push_back(lambda_spherical_degrees_temp);
    } else {
      class_lambdas.push_back(lambda_spherical_degrees);
    }
    start_index += class_num_bearing_vectors[c];
//    std::cout << "lambda_spherical_degrees: " << class_lambdas.back()
//              << ", num_components: " << num_components
//              << ", desired_num_components: " << desired_num_components
//              << std::endl;
  }
}

/*
 * Print Components
 * Prints the mixture model parameters
 */
void PrintComponents(
    int num_components_3d, const Eigen::VectorXf& variances_3d,
    const Eigen::VectorXf& phis_3d, const Eigen::MatrixX3f& mus_3d,
    int num_components_2d, const Eigen::VectorXf& kappas_2d,
    const Eigen::VectorXf& phis_2d, const Eigen::MatrixX3f& mus_2d,
    const std::vector<float>& class_weights) {
  std::cout << "3D:" << std::endl;
  std::cout << "num_components_3d: " << num_components_3d << std::endl;
  std::cout << "variances_3d: " << variances_3d.transpose() << std::endl;
  std::cout << "phis_3d: " << phis_3d.transpose() << std::endl;
  std::cout << "mus_3d: " << std::endl << mus_3d.transpose() << std::endl;
  std::cout << std::endl;
  std::cout << "2D:" << std::endl;
  std::cout << "num_components_2d: " << num_components_2d << std::endl;
  std::cout << "kappas_2d: " << kappas_2d.transpose() << std::endl;
  std::cout << "phis_2d: " << phis_2d.transpose() << std::endl;
  std::cout << "mus_2d: " << std::endl << mus_2d.transpose() << std::endl;
  std::cout << std::endl;
  std::cout << "Class weights: ";
  for (int i = 0; i < class_weights.size(); ++i)
    std::cout << class_weights[i] << " ";
  std::cout << std::endl;
}
