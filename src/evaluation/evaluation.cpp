#include "evaluation.hpp"

namespace eval {

	double g_sideEffect = 0.0;

	class NullBuffer : public std::streambuf
	{
	public:
		int overflow(int c) { return c; }
	};

	NullBuffer nullBuffer;
	std::ostream g_nullStream(&nullBuffer);
}