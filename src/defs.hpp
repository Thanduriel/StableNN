#pragma once

#include <cinttypes>

// network parameters

// number of time-steps given as inputs
constexpr size_t NUM_INPUTS = 16;

// expect only the next time-step
constexpr bool USE_SINGLE_OUTPUT = true;

// size of the internal network state as individual values, not a multiple of the system state size!
constexpr int HIDDEN_SIZE = 4;// 2 * 2;

// use wrapper to increase the number of inputs or reduce the number of outputs
constexpr bool USE_WRAPPER = false;//USE_SINGLE_OUTPUT && (NUM_INPUTS > 1 || HIDDEN_SIZE > 2);

enum struct Mode {
	TRAIN,
	EVALUATE,
	TRAIN_MULTI,
	TRAIN_EVALUATE
};

constexpr Mode MODE = Mode::TRAIN_EVALUATE;

// If > 1, the network is applied NUM_FORWARDS before doing a backward pass. 
// The time-series data is adjusted accordingly to expect results further in the future.
constexpr int64_t NUM_FORWARDS = 1;
constexpr bool SAVE_NET = true;
static_assert(MODE != Mode::TRAIN_EVALUATE || SAVE_NET, "Network needs to be saved in order to be evaluated");

// Append final validation loss of a trained network to a persistent log.
constexpr bool LOG_LOSS = false;

enum struct Optimizer {
	ADAM,
	SGD,
	RMSPROP,
	LBFGS
};
constexpr Optimizer OPTIMIZER = Optimizer::LBFGS;
constexpr bool USE_LBFGS = OPTIMIZER == Optimizer::LBFGS;

// only relevant in TRAIN_MULTI to enforce same initial rng state for all networks
constexpr bool THREAD_FIXED_SEED = true;

// use sequential sampler instead of the default random sampler
constexpr bool USE_SEQ_SAMPLER = USE_LBFGS;

// The seed used to initialize torch when the training begins.
// In TRAIN_MULTI context can still be non-deterministic if the random sampler is used.
constexpr uint64_t TORCH_SEED = 9378341130ul; // 9378341130ul

// evaluation
constexpr bool SHOW_VISUAL = false;