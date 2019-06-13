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

#include "config_map.h"

ConfigMap::ConfigMap() {
}

ConfigMap::ConfigMap(const char * config_file) {
  std::ifstream fin(config_file);

  if (!fin.is_open()) {
    std::cout << "Unable to open config file '" << config_file << "'"
              << std::endl;
    exit(-2);
  } else {
    std::string buffer;
    while (std::getline(fin, buffer)) {
      // Ignore comments
      if (buffer.c_str()[0] == '#') {
      } else {
        this->AddLine(buffer);
      }
    }
    fin.close();
  }
}

ConfigMap::~ConfigMap() {
  for (std::list<double*>::iterator iter =
      double_allocated_memory_collector_.begin();
      iter != double_allocated_memory_collector_.end(); iter++) {
    delete[] (*iter);
  }
  for (std::list<int*>::iterator iter = int_allocated_memory_collector_.begin();
      iter != int_allocated_memory_collector_.end(); iter++) {
    delete[] (*iter);
  }
}

void ConfigMap::AddLine(std::string line_) {
  // Swallow last character (carriage return: ASCII 13)
  if (line_.size() > 0) {
    if ((int) line_.c_str()[line_.size() - 1] == 13) {
      line_.resize(line_.size() - 1);
    }
  }

  StringTokeniser st(line_, const_cast<char*>(" =;"));
  if (st.NumberOfTokens() != 2) {
    return;
  }

  std::string key = st.NextToken();
  std::string val = st.NextToken();
  AddPair(key, val);
}

void ConfigMap::AddPair(std::string key_, std::string value_) {
  mappings_[key_] = value_;
}

char * ConfigMap::Get(char * key_) {
  std::string key(key_);
#ifdef VERBOSE
  std::cout << "DEBUG::ConfigMap.get()::key is `" << key_ << "'" << std::endl;
  std::string val = mappings_[key];
  std::cout << "DEBUG::Requesting (" << key_ << ")->(" << val << ")" << std::endl;
  return const_cast<char*>(val.c_str());
#else
  return const_cast<char*>(mappings_[key].c_str());
#endif
}

char * ConfigMap::Get(const char * key_) {
  return Get(const_cast<char*>(key_));
}

int ConfigMap::GetI(char * key_) {
  char * str_val = Get(key_);

  if (str_val == NULL) {
    return 0;
  } else {
    bool has_exponent = false;
    if (strchr(str_val, 'e') != NULL)
      has_exponent = true;
    if (strchr(str_val, 'E') != NULL)
      has_exponent = true;
    if (has_exponent) {
      return static_cast<int>(atof(str_val));
    } else {
      return atoi(str_val);
    }
  }
}

int ConfigMap::GetI(const char * key_) {
  return GetI(const_cast<char*>(key_));
}

double ConfigMap::GetF(char *key_) {
  char * str_val = Get(key_);

  if (str_val == NULL) {
    return 0;
  } else {
    return atof(str_val);
  }
}

double ConfigMap::GetF(const char * key_) {
  return GetF(const_cast<char*>(key_));
}

int* ConfigMap::GetIArray(char * key_) {
  std::string key(key_);
  std::string val = mappings_[key];
  if (val.empty()) {
    return NULL;
  }
  StringTokeniser st(val, const_cast<char*>("(,)"));
  int* return_array = new int[st.NumberOfTokens()];
  int_allocated_memory_collector_.push_back(return_array);
  for (int i = 0; st.NumberOfTokens() > 0; i++) {
    return_array[i] = (int) atoi(st.NextToken().c_str());
  }
  return return_array;
}

int* ConfigMap::GetIArray(const char * key_) {
  return GetIArray(const_cast<char*>(key_));
}

int* ConfigMap::GetIArray(char * key_, int& num_elements_) {
  std::string key(key_);
  std::string val = mappings_[key];
  if (val.empty()) {
    return NULL;
  }
  StringTokeniser st(val, const_cast<char*>("(,)"));
  num_elements_ = st.NumberOfTokens();
  int* return_array = new int[st.NumberOfTokens()];
  int_allocated_memory_collector_.push_back(return_array);
  for (int i = 0; st.NumberOfTokens() > 0; i++) {
    return_array[i] = (int) atoi(st.NextToken().c_str());
  }
  return return_array;
}

int* ConfigMap::GetIArray(const char * key_, int& num_elements_) {
  return GetIArray(const_cast<char*>(key_), num_elements_);
}

double* ConfigMap::GetFArray(char * key_) {
  std::string key(key_);
  std::string val = mappings_[key];
  if (val.empty()) {
    return NULL;
  }
  StringTokeniser st(val, const_cast<char*>("(,)"));
  double* return_array = new double[st.NumberOfTokens()];
  double_allocated_memory_collector_.push_back(return_array);
  for (int i = 0; st.NumberOfTokens() > 0; i++) {
    return_array[i] = (double) atof(st.NextToken().c_str());
  }
  return return_array;
}

double* ConfigMap::GetFArray(const char * key_) {
  return GetFArray(const_cast<char*>(key_));
}

double* ConfigMap::GetFArray(char * key_, int& num_elements_) {
  std::string key(key_);
  std::string val = mappings_[key];
  if (val.empty()) {
    return NULL;
  }
  StringTokeniser st(val, const_cast<char*>("(,)"));
  num_elements_ = st.NumberOfTokens();
  double* return_array = new double[st.NumberOfTokens()];
  double_allocated_memory_collector_.push_back(return_array);
  for (int i = 0; st.NumberOfTokens() > 0; i++) {
    return_array[i] = (double) atof(st.NextToken().c_str());
  }
  return return_array;
}

double* ConfigMap::GetFArray(const char * key_, int& num_elements_) {
  return GetFArray(const_cast<char*>(key_), num_elements_);
}

void ConfigMap::Print() {
  for (std::map<std::string, std::string>::const_iterator iter =
      mappings_.begin(); iter != mappings_.end(); iter++) {
    std::cout << "(" << iter->first << ")->(" << iter->second << ")"
              << std::endl;
  }
}
