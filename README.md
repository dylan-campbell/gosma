# Globally-Optimal Spherical Mixture Alignment (GOSMA)

This code implements the GOSMA algorithm for estimating the position and orientation of a calibrated camera from a single image with respect to a 3D model.

Copyright (c) 2019 Dylan John Campbell

## Installing

### Dependencies
* Eigen: instructions for downloading and installing can be found at http://eigen.tuxfamily.org/.
* CppOptimizationLibrary: instructions for downloading can be found at https://github.com/PatWie/CppNumericalSolvers
* CUDA: instructions for downloading and installing can be found at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### Operating System
* Developed and tested on Ubuntu 18.04 LTS
* May be compiled on other systems, although no guarantee is given

### Installing on Linux
1. In the src folder, copy (or link to) the CppNumericalSolvers folder
2. In the release folder, open a terminal
3. Run: `cmake -D CMAKE_BUILD_TYPE=Release ../src`
4. Run: `make`

## Running

1. Run the following command from the terminal in the Release folder (arguments in square brackets are optional):
```
./gosma point-set_filename bearing-vector-set_filename config_filename output_filename [translation_domain_filename [rotation_domain_filename]]
```

### Inputs
* `point-set_filename`: filename of a plain text file with rows of whitespace-separated coordinates `[x y z]`
* `bearing-vector-set_filename`: filename of a plain text file with rows of whitespace-separated normalised coordinates `[x y z]`
* `config_filename`: filename of a configuration file, see ./release/config.txt
* `output_filename`: filename of the file in which to append the program output
* `translation_domain_filename` [optional]: filename of a plain text file with rows of whitespace-separated coordinates and side-widths `[x y z wx wy wz]` where `[x y z]` are the coordinates of minimum vertex and `[wx wy wz]` are the cuboid's side-widths
* `rotation_domain_filename` [optional]: filename of a plain text file with rows of whitespace-separated coordinates and side-widths `[x y z wx wy wz]` where `[x y z]` are the coordinates of minimum vertex and `[wx wy wz]` are the cuboid's side-widths
* If more than a single class is present, the user may optionally include class count files, to be located in the same directory as the data files (point-set and bearing vector set files).
  * Assumes that the file name is the same as the data file name with "_class_counts" appended
    * _e.g._ point_set_class_counts.txt
  * Format: `<label number>`
    * Class labels are assumed to correspond between 2D and 3D data

### Output
* The output is appended to the output file in the form `[tx ty tz R11 R12 R13 R21 R22 R23 R31 R32 R33 l d1 d2 c]`
* `t` is a 3x1 translation vector
* `R` is a 3x3 rotation matrix such that the transformed point-set `P' = R(P - t)` is aligned with the bearing vector set F
* `l` is L2 distance of the mixture models at the output transformation
* `d1` is the runtime duration (in seconds) of the GOSMA algorithm
* `d2` is the runtime duration (in seconds) of generating the mixture models
* `c` is the optimality certificate (1 if optimal)

### Settings
* See ./release/config.txt
* GPUs: if more than one GPU is available, adjust `kNumDevices` in gosma.cu accordingly

# Citation

Any publications resulting from the use of this code should cite the following paper:
> Dylan Campbell, Lars Petersson, Laurent Kneip, Hongdong Li, and Stephen Gould, "The Alignment of the Spheres: Globally-Optimal Spherical Mixture Alignment for Camera Pose Estimation", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, USA, IEEE, Jun. 2019

Please use the following bibtex entry:
```bibtex
@inproceedings{campbell2019globally,
  author = {Campbell, Dylan and Petersson, Lars and Kneip, Laurent and Li, Hongdong and Gould, Stephen},
  title = {The Alignment of the Spheres: Globally-Optimal Spherical Mixture Alignment for Camera Pose Estimation},
  booktitle = {Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={to appear},
  address={Long Beach, USA},
  month = {June},
  year = {2019},
  doi={},
  organization={IEEE}
}
```

## Licence
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details.


