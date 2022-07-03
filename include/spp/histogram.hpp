#ifndef SPP_HISTOGRAM_HPP
#define SPP_HISTOGRAM_HPP

#include <cooperative_groups.h>

#include "types.hpp"
#include "traits.hpp"
#include "pipelined_for.hpp"
#include "kernel_launch.hpp"
#include "operators/math.hpp"



namespace spp {

	namespace cg = cooperative_groups;



	namespace global {

		template <usize ThreadsPerBlock, usize ItemsPerThread, typename InputIterator, typename Histogram, typename Binning>
		__global__
		void histogram(InputIterator data_in, usize size, Histogram * p_grid_histogram, Binning binning) {
		
			using InputType		= std::decay_t<dereference_t<InputIterator>>;
			using KeyType		= std::decay_t<apply_t<Binning, InputType>>;
			using ValueType		= std::decay_t<subscript_t<Histogram, KeyType>>;



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



			constexpr usize WarpsPerBlock	{ ThreadsPerBlock / 32 };
			constexpr usize ItemsPerWarp	{ ItemsPerThread * 32 };

			usize const i_warp_begin	{ usize(grid.block_rank()) * WarpsPerBlock + warp.meta_group_rank() };
			usize const i_warp_end		{ ceiled_div(size, ItemsPerWarp) };
			usize const i_warp_step		{ usize(grid.num_blocks()) * warp.meta_group_size() };

			for (usize i_warp = i_warp_begin; i_warp < i_warp_end; i_warp += i_warp_step) {

				usize const thread_rank_begin{ i_warp * ItemsPerWarp + warp.thread_rank() };

				InputType thread_items[ItemsPerThread];

				constexpr usize ItemsPerIteration = spp::op::min()(usize(2), ItemsPerThread);

				device::pipelined_for<ItemsPerThread, ItemsPerIteration>([&] (usize j_item) {
					usize const item_rank{ thread_rank_begin + 32 * j_item };
					if (item_rank < size) {
						thread_items[j_item] = *(data_in + item_rank);
					}
				}, [&] (usize j_item) {
					usize const item_rank{ thread_rank_begin + 32 * j_item };
					if (item_rank < size) {
						KeyType const key = binning(thread_items[j_item]);
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



	namespace kernel {

		template <typename InputIterator, typename Histogram, typename Binning>
		cudaError_t histogram(InputIterator data_in, usize size, Histogram * p_histogram, Binning binning) {
			constexpr usize ThreadsPerBlock{ 512 };
			constexpr usize ItemsPerThread{ recommended_thread_tile_v<std::decay_t<dereference_t<InputIterator>>> };
			
			auto fn{ global::histogram<ThreadsPerBlock, ItemsPerThread, InputIterator, Histogram, Binning> };
			
			return launch_cooperative_kernel(fn, ThreadsPerBlock, data_in, size, p_histogram, binning);
		}

	}

} // namespace spp



#endif // SPP_HISTOGRAM_HPP