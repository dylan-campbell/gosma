# Globally-Optimal Spherical Mixture Alignment (GOSMA)
# Copyright (c) 2019 Dylan John Campbell
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Contact Details:
# dylan.campbell@anu.edu.au
# Brian Anderson Building 115, North Road,
# Australian National University, Canberra ACT 2600, Australia
#
# Any publications resulting from the use of this code should cite the
# following paper:
#
# Dylan Campbell, Lars Petersson, Laurent Kneip, Hongdong Li, and Stephen Gould,
# "The Alignment of the Spheres: Globally-Optimal Spherical Mixture Alignment
# for Camera Pose Estimation", IEEE Conference on Computer Vision and Pattern
# Recognition (CVPR), Long Beach, USA, IEEE, Jun. 2019
#
# For the full license, see LICENSE.md in the root directory
#
# Author: Dylan Campbell
# Date: 20190612
# Revision: 2.0

PROJECT(GOSMA)

CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --std=c++11; -arch=sm_30)

SET(GOSMA_SRCS
	gosma_api.cpp
	gosma.cu
	point_set.cpp
	bearing_vector_set.cpp
	config_map.cpp
	string_tokeniser.cpp
)
CUDA_ADD_EXECUTABLE(gosma ${GOSMA_SRCS})
