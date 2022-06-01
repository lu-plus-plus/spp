#ifndef SPP_COPY_HPP
#define SPP_COPY_HPP

#include <cstdint>
#include <cooperative_groups.h>

#include "traits.hpp"
#include "kernel_launch.hpp"



namespace spp::global {

	template <uint32_t ItemsPerThread, typename OutputIterator, typename InputIterator>
	__global__
	void copy(OutputIterator data_out, InputIterator data_in, uint32_t size) {

		auto const grid{ cooperative_groups::this_grid() };
		auto const item_rank_begin{ uint32_t(grid.thread_rank()) * ItemsPerThread };

		using InputType = std::decay_t<dereference_t<InputIterator>>;
		InputType items[ItemsPerThread];

		for (uint32_t i = 0; i < ItemsPerThread; ++i) {
			uint32_t item_rank = item_rank_begin + i;
			if (item_rank < size) items[i] = *(data_in + item_rank);
		}

		for (uint32_t i = 0; i < ItemsPerThread; ++i) {
			uint32_t item_rank = item_rank_begin + i;
			if (item_rank < size) *(data_out + item_rank) = items[i];
		}

	}

}

namespace spp::kernel {

	template <typename OutputIterator, typename InputIterator>
	cudaError_t copy(OutputIterator data_out, InputIterator data_in, uint32_t size) {
		uint32_t const ThreadsPerBlock{ 256 };
		uint32_t const ItemsPerThread{ 4 };

		auto fn			= global::copy<ItemsPerThread, OutputIterator, InputIterator>;
		auto grid_dim	= ceiled_div(size, ThreadsPerBlock);
		auto block_dim	= ThreadsPerBlock;
		
		return launch_kernel(fn, grid_dim, block_dim, data_out, data_in, size);
	}

}



#endif // SPP_COPY_HPP