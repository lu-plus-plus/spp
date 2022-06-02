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
	uint32_t aligned_bytes_for(uint32_t size, uint32_t alignment = 128u) {
		return ceiled_div(size, alignment) * alignment;
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

	template <typename Fn, typename ... Args>
	cudaError_t launch_kernel(Fn && fn_in, dim3 grid_dim, dim3 block_dim, Args ... args_in) {
		auto fn{ reinterpret_cast<void const *>(std::forward<Fn>(fn_in)) };
		addresses_of args{ args_in ... };

		return cudaLaunchKernel(fn, grid_dim, block_dim, args.get());
	}

	template <typename Fn, typename ... Args>
	cudaError_t launch_cooperative_kernel(Fn && fn_in, dim3 block_dim, Args ... args_in) {
		auto fn{ reinterpret_cast<void const *>(std::forward<Fn>(fn_in)) };
		dim3 const grid_dim{ max_active_blocks_for(fn, block_dim.x * block_dim.y * block_dim.z) };
		addresses_of args{ args_in ... };

		return cudaLaunchCooperativeKernel(fn, grid_dim, block_dim, args.get());
	}

}



#endif // SPP_KERNEL_LAUNCH_HPP