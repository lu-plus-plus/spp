#ifndef SPP_LOG_HPP
#define SPP_LOG_HPP

#include <cstdio>
#include <cstdlib>

#define cudaCheck(stat) do {		\
	cudaError_t result = stat;		\
	if (cudaSuccess != result) {	\
		std::fprintf(stderr, "[error] %s: %s\n" "\tat line %d, file '%s'\n", cudaGetErrorName(result), cudaGetErrorString(result), __LINE__, __FILE__); \
		std::exit(EXIT_FAILURE);	\
	}								\
} while (false)

#endif // SPP_LOG_HPP