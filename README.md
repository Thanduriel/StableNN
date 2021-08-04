# StableNN
A collection of experiments to build stable neural networks as explicit time-stepping schemes for physical simulations.
The two prototypical examples are the gravity pendulum and the 1D heat equation for non-homogeneous media.

## Build
The only necessary dependency is
* [libtorch](https://pytorch.org/get-started/locally/) installed anywhere,

but optionally you may also want
* [SFML](https://www.sfml-dev.org/) to visualize the simulations; either installed such that cmake can find it or to build from source clone it into dependencies/SFML,
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for performance measures although the related code is commented out and not faster than the manual implementation.

Then just run
```sh
$ git clone https://github.com/Thanduriel/StableNN.git
$ cd StableNN
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_PREFIX_PATH="<path to libtorch>/share/cmake/Torch" -DUSE_GRAPHICS=<ON/OFF> -DCMAKE_BUILD_TYPE=Release
```

Make sure to use a release build with `NDEBUG` defined for experiments, because otherwise very little data is generated to allow for debugging with reasonable speed.

The programs where tested with libtorch-1.7.0 and both g++-11.1 and msvc-19.29.
## Usage
Each simulation has its own executable, `pendulum` and `heateq`. All parameters are set in the code and require recompiling. General settings including the mode (training or evaluation) are set in `defs.hpp`. Parameters of the systems and networks can be found in `mainpendulum.cpp` + `pendulumeval.hpp` and `mainheateq.cpp` + `heateqeval.hpp`.

Most parameters are saved with the network during training in an .hparam file of the same name. However, to load a network a few compile time parameters need to set manually. These are the network type, `UseWrapper` and `NUM_INPUTS`.

For examples how the various evaluation methods are used take a look at the `experiments` branch.