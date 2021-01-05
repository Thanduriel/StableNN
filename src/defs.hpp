#pragma once

#include <cinttypes>

// simulation related
constexpr int64_t HYPER_SAMPLE_RATE = 100;

// network parameters
constexpr size_t NUM_INPUTS = 1;
constexpr bool USE_SINGLE_OUTPUT = true;
constexpr int HIDDEN_SIZE = 2 * 2;
constexpr bool USE_WRAPPER = USE_SINGLE_OUTPUT && (NUM_INPUTS > 1 || HIDDEN_SIZE > 2);

// training
enum struct Mode {
	TRAIN,
	EVALUATE,
	TRAIN_MULTI,
	TRAIN_EVALUATE
};
constexpr Mode MODE = Mode::TRAIN_EVALUATE;
constexpr int64_t NUM_FORWARDS = 1;
constexpr bool SAVE_NET = true;
constexpr bool LOG_LOSS = true;
constexpr bool USE_LBFGS = true;
// only relevant in TRAIN_MULTI to enforce same initial rng state for all networks
constexpr bool THREAD_FIXED_SEED = true;
constexpr bool USE_SEQ_SAMPLER = USE_LBFGS;
constexpr uint64_t TORCH_SEED = 9378341130ul;

// evaluation
constexpr bool SHOW_VISUAL = false;