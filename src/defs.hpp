#pragma once

#include <cinttypes>

// network parameters

// number of time-steps given as inputs
// This value only needs to be available at compile-time for evaluation.
constexpr size_t NUM_INPUTS = 8;

// expect only the next time-step
constexpr bool USE_SINGLE_OUTPUT = true;

enum struct Mode {
	TRAIN,
	EVALUATE,
	TRAIN_MULTI,
	TRAIN_EVALUATE
};

constexpr Mode MODE = Mode::TRAIN;

// If > 1, the network is applied NUM_FORWARDS before doing a backward pass. 
// The time-series data is adjusted accordingly to expect results further in the future.
constexpr int64_t NUM_FORWARDS = 1;
constexpr bool SAVE_NET = true || MODE == Mode::EVALUATE;
static_assert(MODE != Mode::TRAIN_EVALUATE || SAVE_NET, "Network needs to be saved in order to be evaluated.");

// Append final validation loss of a trained network to a persistent log.
constexpr bool LOG_LOSS = false;

// Write training and validation loss after each epoch into a file.
constexpr bool LOG_LEARNING_LOSS = false;

enum struct Optimizer {
	ADAM,
	SGD,
	RMSPROP,
	LBFGS
};
constexpr Optimizer OPTIMIZER = Optimizer::ADAM;
constexpr bool USE_LBFGS = OPTIMIZER == Optimizer::LBFGS;

// only relevant in TRAIN_MULTI to enforce same initial rng state for all networks
constexpr bool THREAD_FIXED_SEED = true;

// use sequential sampler instead of the default random sampler
constexpr bool USE_SEQ_SAMPLER = USE_LBFGS;

// The seed used to initialize torch when the training begins.
// In TRAIN_MULTI context can still be non-deterministic if the random sampler is used.
constexpr uint64_t TORCH_SEED = 9378341130ul; // 9378341130ul

// evaluation
// only set to true if build option USE_GRAPHICS=true
constexpr bool SHOW_VISUAL = false;