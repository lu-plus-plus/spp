#ifndef SPP_KERNEL_LAUNCH_HPP
#define SPP_KERNEL_LAUNCH_HPP

#include <cstdint>



namespace spp::kernel {

	inline
	uint32_t get_max_active_blocks(void const * fn, int numThreads) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);

		int numBlocksPerSm = 0;
		cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fn, numThreads, 0));

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