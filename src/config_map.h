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

#ifndef CONFIGMAP_H
#define CONFIGMAP_H

#include <map>
#include <list>
#include <string>
#include <cstring> // for strchr
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "string_tokeniser.h"

class ConfigMap {
 private:
  std::map<std::string, std::string> mappings_;
  std::list<double*> double_allocated_memory_collector_;
  std::list<int*> int_allocated_memory_collector_;

 public:
  ConfigMap();
  ConfigMap(const char* config_file);
  ~ConfigMap();

  // Adds a config line to the map
  void AddLine(std::string line_);

  // Adds a key->value mapping
  void AddPair(std::string key_, std::string value_);

  // Gets the string value at a given key
  char* Get(char* key_);
  char* Get(const char* key_);

  // Gets the integer value at a given key
  // (uses the stdlib function 'atoi' to convert)
  int GetI(char* key_);
  int GetI(const char* key_);

  // Gets the double value at a given key
  // (uses the stdlib function 'atof' to convert)
  double GetF(char* key_);
  double GetF(const char* key_);

  // Gets an array of integers at the given key
  int* GetIArray(char* key_);
  int* GetIArray(const char* key_);
  int* GetIArray(char* key_, int& num_elements_);
  int* GetIArray(const char* key_, int& num_elements_);

  // Gets an array of floats at the given key
  double* GetFArray(char* key_);
  double* GetFArray(const char* key_);
  double* GetFArray(char* key_, int& num_elements_);
  double* GetFArray(const char* key_, int& num_elements_);

  // Prints the contents of the map list to stdout
  void Print();

};

#endif /* CONFIGMAP_H */
