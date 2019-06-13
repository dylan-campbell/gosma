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
#include "string_tokeniser.h"

StringTokeniser::StringTokeniser() {
}

StringTokeniser::StringTokeniser(std::string str, char delim) {
  int delim_loc;
  while ((delim_loc = str.find_first_of(delim, 0)) != std::string::npos) {
    if (str.substr(0, delim_loc).length() > 0) {
      tokens_.push_back(str.substr(0, delim_loc));
    }
    str = str.substr(delim_loc + 1, str.length());
  }
  if (str.length() > 0) {
    tokens_.push_back(str);
  }
}

StringTokeniser::StringTokeniser(std::string str, char * delims) {
  int delim_loc;
  while ((delim_loc = str.find_first_of(delims, 0)) != std::string::npos) {
    if (str.substr(0, delim_loc).length() > 0) {
      tokens_.push_back(str.substr(0, delim_loc));
    }
    str = str.substr(delim_loc + 1, str.length());
  }
  if (str.length() > 0)
    tokens_.push_back(str);
}

StringTokeniser::~StringTokeniser() {
}

std::string StringTokeniser::NextToken() {
  if (!HasMoreTokens())
    return "";
  std::string return_str(tokens_.front());
  tokens_.pop_front();
  return return_str;
}

bool StringTokeniser::HasMoreTokens() {
  return !tokens_.empty();
}

int StringTokeniser::NumberOfTokens() {
  return tokens_.size();
}
