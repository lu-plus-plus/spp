#ifndef SPP_HISTOGRAM_HPP
#define SPP_HISTOGRAM_HPP

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "types.hpp"
#include "pipelined_for.hpp"



namespace spp {

	namespace cg = cooperative_groups;



	namespace global {

		template <usize ThreadsPerBlock, usize ItemsPerThread, typename InputIterator, typename Histogram, typename Binning>
		__global__
		void histogram(InputIterator data_in, usize size, Histogram * p_grid_histogram, Binning binning) {
		
			usize constexpr WarpsPerBlock	{ ThreadsPerBlock / 32 };
			usize constexpr ItemsPerWarp	{ ItemsPerThread * 32 };

			auto const grid		{ cg::this_grid() };
			auto const block	{ cg::this_thread_block() };
			auto const warp		{ cg::tiled_partition<32>(block) };

			Histogram & grid_histogram{ *p_grid_histogram };

			for (usize bin_rank = grid.thread_rank(); bin_rank < Histogram::size(); bin_rank += grid.num_threads()) {
				grid_histogram[bin_rank] = 0;
			}

			__shared__ Histogram block_histogram;

			for (usize bin_rank = block.thread_rank(); bin_rank < Histogram::size(); bin_rank += block.num_threads()) {
				block_histogram[bin_rank] = 0;
			}

			grid.sync();

			usize const i_warp_begin	{ usize(grid.block_rank()) * WarpsPerBlock + warp.meta_group_rank() };
			usize const i_warp_end		{ ceiled_div(size, ItemsPerWarp) };
			usize const i_warp_step		{ usize(grid.num_blocks()) * warp.meta_group_size() };

			for (usize i_warp = i_warp_begin; i_warp < i_warp_end; i_warp += i_warp_step) {
				usize const thread_rank_begin{ i_warp * ItemsPerWarp + warp.thread_rank() };

				u32 thread_items[ItemsPerThread];

				device::pipelined_for<ItemsPerThread, 2>([&] (usize j_item) {
					usize const item_rank{ thread_rank_begin + 32 * j_item };
					if (item_rank < size) {
						thread_items[j_item] = *(data_in + item_rank);
					}
				}, [&] (usize j_item) {
					usize const item_rank{ thread_rank_begin + 32 * j_item };
					if (item_rank < size) {
						usize const key = binning(thread_items[j_item]);
						atomicAdd(&(block_histogram[key]), 1);
					}
				});
			}

			block.sync();

			for (usize bin_rank = block.thread_rank(); bin_rank < Histogram::size(); bin_rank += block.num_threads()) {
				atomicAdd(&(grid_histogram[bin_rank]), block_histogram[bin_rank]);
			}

		} // histogram
	
	} // namespace global

} // namespace spp



#endif // SPP_HISTOGRAM_HPP