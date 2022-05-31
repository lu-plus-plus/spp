#ifndef SPP_KERNEL_LAUNCH_HPP
#define SPP_KERNEL_LAUNCH_HPP

#include <cstdio>
#include <cstdlib>
#include <cstdint>



#define cudaCheck(stat) do {		\
	cudaError_t result = stat;		\
	if (cudaSuccess != result) {	\
		std::fprintf(stderr, "[error] %s: %s\n" "\tat line %d, file '%s'\n", cudaGetErrorName(result), cudaGetErrorString(result), __LINE__, __FILE__); \
		std::exit(EXIT_FAILURE);	\
	}								\
} while (false)



namespace spp {

	inline constexpr
	__host__ __device__
	uint32_t ceiled_div(uint32_t dividend, uint32_t divisor) {
		return (dividend + divisor - 1) / divisor;
	}

	inline constexpr
	__host__ __device__
	uint32_t cuda_aligned_bytes_for(uint32_t size) {
		return ceiled_div(size, 128) * 128;
	}

	inline
	uint32_t max_active_blocks_for(void const * func, int numThreads) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);

		int numBlocksPerSm = 0;
		cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, func, numThreads, 0));

		return uint32_t(numBlocksPerSm * deviceProp.multiProcessorCount);
	}

	template <typename ... Args>
	struct addresses_of {

		void * ptr_to_args[sizeof...(Args)];

		addresses_of(Args & ... args) : ptr_to_args{ (&args)... } {}

		void * * get() {
			return ptr_to_args;
		}

	};

}



#endif // SPP_KERNEL_LAUNCH_HPP