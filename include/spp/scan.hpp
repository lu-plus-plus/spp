#ifndef SPP_REDUCE_HPP
#define SPP_REDUCE_HPP

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "types.hpp"
#include "math.hpp"
#include "barrier.hpp"
#include "identity.hpp"
#include "lookback.hpp"



namespace spp {

	namespace cg = ::cooperative_groups;



	namespace device {

		template <typename T, u32 WarpsPerBlock, u32 ProducerBarrierId, u32 ConsumerBarrierId>
		__device__
		T scan_warp_exclusive_prefix(T const & value, cg::thread_block_tile<32> const & warp) {

			__shared__ T warp_partial_results[WarpsPerBlock];

			if (warp.thread_rank() + 1 == 32) {
				warp_partial_results[warp.meta_group_rank()] = value;
			}

			barrier<ProducerBarrierId, WarpsPerBlock>		producer_barrier;
			barrier<ConsumerBarrierId, WarpsPerBlock - 1>	consumer_barrier;

			if (warp.meta_group_rank() == 0) {
				producer_barrier.arrive();

				return identity<T>()();
			}
			else if (warp.meta_group_rank() + 1 != WarpsPerBlock) {
				producer_barrier.arrive();
				
				consumer_barrier.wait();

				return warp_partial_results[warp.meta_group_rank()];
			}
			else {
				producer_barrier.wait();

				if (warp.thread_rank() < WarpsPerBlock) {
					cg::coalesced_group active = cg::coalesced_threads();
					T const warp_partial_result = warp_partial_results[warp.thread_rank()];
					warp_partial_results[warp.thread_rank()] = cg::exclusive_scan(active, warp_partial_result);					
				}

				consumer_barrier.arrive();

				return warp_partial_results[warp.meta_group_rank()];
			}

		}



		template <typename T, u32 WarpsPerBlock, u32 BarrierId>
		__device__
		T scan_block_exclusive_prefix(lookback<T> volatile * block_lookbacks,
			T const & warp_exclusive_prefix, T const & warp_reduction,
			cg::grid_group const & grid, cg::thread_block_tile<32> const & warp) {

			T block_exclusive_prefix;
			__shared__ T shared_block_exclusive_prefix;

			barrier<BarrierId, WarpsPerBlock> barrier;

			if (warp.meta_group_rank() == 0) {
				block_exclusive_prefix = identity<T>()();
				
				if (warp.thread_rank() == 0) {
					for (isize i_block = isize(grid.block_rank()) - 1_is; 0 <= i_block; --i_block) {
						lookback<T> block_lookback = block_lookbacks[i_block].wait_and_load();
						
						block_exclusive_prefix += block_lookback.get();

						if (block_lookback.is_prefixed()) break;
					}

					shared_block_exclusive_prefix = block_exclusive_prefix;
				}

				barrier.arrive();

				block_exclusive_prefix = warp.shfl(block_exclusive_prefix, 0);

				return block_exclusive_prefix;
			}
			else if (warp.meta_group_rank() + 1 != WarpsPerBlock) {
				barrier.wait();

				block_exclusive_prefix = shared_block_exclusive_prefix;

				return block_exclusive_prefix;
			}
			else {
				if (warp.thread_rank() + 1 == 32) {
					block_lookbacks[grid.block_rank()] = lookback<T>::make_aggregate(warp_exclusive_prefix + warp_reduction);
				}

				barrier.wait();
				
				block_exclusive_prefix = shared_block_exclusive_prefix;

				if (warp.thread_rank() + 1 == 32) {
					block_lookbacks[grid.block_rank()] = lookback<T>::make_prefix(block_exclusive_prefix + warp_exclusive_prefix + warp_reduction);
				}

				return block_exclusive_prefix;
			}

		}

	} // namespace device



	namespace global {

		template <typename T, u32 ThreadsPerBlock, u32 ItemsPerThread>
		__global__
		void inclusive_scan(T const * values, T * results, u32 length, lookback<T> volatile * block_desc) {

			u32 constexpr WARPS_PER_BLOCK	= ThreadsPerBlock / 32;

			u32 constexpr ItemsPerBlock	= ItemsPerThread * ThreadsPerBlock;
			u32 constexpr ItemsPerWarp	= ItemsPerThread * 32;

			auto const grid				= cg::this_grid();
			auto const block			= cg::this_thread_block();
			auto const warp				= cg::tiled_partition<32>(block);

			u32 const block_rank_begin	= grid.block_rank() * ItemsPerBlock;
			u32 const warp_rank_begin	= warp.meta_group_rank() * ItemsPerWarp;



			T items[ItemsPerThread];

			__shared__ T warp_partial_results[WARPS_PER_BLOCK];
			T warp_exclusive_prefix = identity<T>()();

			__shared__ T shared_block_exclusive_prefix;
			T block_exclusive_prefix = identity<T>()();



			for (u32 i_item = 0; i_item < ItemsPerThread; ++i_item) {
				u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_item + warp.thread_rank();
				items[i_item] = identity<T>()();
				if (item_rank < length) {
					Byte<sizeof(T)>::copy(items + i_item, values + item_rank);
				}
				items[i_item] = cg::inclusive_scan(warp, items[i_item]);
				if (i_item) {
					items[i_item] += warp.shfl(items[i_item - 1], 32 - 1);
				}
			}

			T const & warp_reduction = items[ItemsPerThread - 1];



			u32 constexpr warp_prefix_producer_barrier = 1;
			u32 constexpr warp_prefix_consumer_barrier = 2;
			u32 constexpr block_prefix_producer_barrier = 3;
			
			warp_exclusive_prefix = device::scan_warp_exclusive_prefix<T, WARPS_PER_BLOCK, warp_prefix_producer_barrier, warp_prefix_consumer_barrier>(warp_reduction, warp);

			block_exclusive_prefix = device::scan_block_exclusive_prefix<T, WARPS_PER_BLOCK, block_prefix_producer_barrier>(block_desc, warp_exclusive_prefix, warp_reduction, grid, warp);



			for (u32 i_item = 0; i_item < ItemsPerThread; ++i_item) {
				items[i_item] += block_exclusive_prefix + warp_exclusive_prefix;
				
				u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_item + warp.thread_rank();
				if (item_rank < length) {
					Byte<sizeof(T)>::copy(results + item_rank, items + i_item);
				}
			}

		}



		template <typename T>
		__global__
		void init_inclusive_scan(lookback<T> volatile * block_lookbacks, u32 num_descriptors) {
			auto const grid = cg::this_grid();

			for (u32 i = grid.thread_rank(); i < num_descriptors; i += grid.num_threads()) {
				block_lookbacks[i] = lookback<T>::zero();
			}
		}

	} // namespace global



	namespace kernel {

		template <typename T>
		cudaError_t inclusive_scan(T const * values, T * results, u32 length, void * temp_storage, u32 & temp_storage_bytes) {
			constexpr u32 threads_per_block = 512;
			constexpr u32 items_per_thread = 16;
			
			constexpr u32 items_per_block = items_per_thread * threads_per_block;

			auto num_blocks = ceiled_div(length, items_per_block);

			if (temp_storage == nullptr || temp_storage_bytes == 0) {
				temp_storage_bytes = num_blocks * sizeof(lookback<T>);
				return cudaSuccess;
			}
			else {
				/* init */ {
					auto fn			= reinterpret_cast<void const *>(global::init_inclusive_scan<T>);
					auto block_dim	= dim3(128, 1, 1);
					auto grid_dim	= dim3(ceiled_div(num_blocks, block_dim.x), 1, 1);
					void * args[]	= { &temp_storage, &num_blocks };
					
					if (auto e = cudaLaunchKernel(fn, grid_dim, block_dim, args); cudaSuccess != e) {
						return e;
					}
				}

				/* scan */ {
					auto fn			= reinterpret_cast<void const *>(global::inclusive_scan<T, threads_per_block, items_per_thread>);
					auto grid_dim	= dim3(num_blocks, 1, 1);
					auto block_dim	= dim3(threads_per_block, 1, 1);
					void * args[] = { &values, &results, &length, &temp_storage };

					return cudaLaunchKernel(fn, grid_dim, block_dim, args);
				}
			}
		}

	} // namespace kernel

} // namespace spp



#endif // SPP_REDUCE_HPP