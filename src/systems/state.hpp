#pragma once

#include <array>

namespace systems {

	template<typename T, std::size_t Size>
	using State = std::array<T, Size>;
}