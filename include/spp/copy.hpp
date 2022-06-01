#ifndef SPP_COPY_HPP
#define SPP_COPY_HPP

#include <cstdint>

#include "traits.hpp"
#include "kernel_launch.hpp"



namespace spp::global {

	template <uint32_t ThreadsPerBlock, uint32_t ItemsPerThread, typename OutputIterator, typename InputIterator>
	__global__
	void copy(OutputIterator data_out, InputIterator data_in, uint32_t size) {

		using InputType = std::decay_t<dereference_t<InputIterator>>;

		uint32_t constexpr ItemsPerBlock{ ItemsPerThread * ThreadsPerBlock };

		InputType items[ItemsPerBlock];

		for (uint32_t i = 0; i < ItemsPerBlock; ++i) {
			items[i] = *(data_in + i);
		}

		for (uint32_t i = 0; i < ItemsPerBlock; ++i) {
			*(data_out + i) = items[i];
		}

	}

}

namespace spp::kernel {

	template <typename OutputIterator, typename InputIterator>
	cudaError_t copy(OutputIterator data_out, InputIterator data_in, uint32_t size) {
		uint32_t const ThreadsPerBlock{ 256 };
		uint32_t const ItemsPerThread{ 4 };

		auto fn			= global::copy<ThreadsPerBlock, ItemsPerThread, OutputIterator, InputIterator>;
		auto grid_dim	= ceiled_div(size, ThreadsPerBlock);
		auto block_dim	= ThreadsPerBlock;
		
		return launch_kernel(fn, grid_dim, block_dim, data_out, data_in, size);
	}

}



#endif // SPP_COPY_HPP