#ifndef SPP_REDUCE_HPP
#define SPP_REDUCE_HPP

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "types.hpp"
#include "barrier.hpp"
#include "identity.hpp"
#include "math.hpp"



namespace spp {

	namespace cg = ::cooperative_groups;



	enum struct block_status : u32 {
		invalid = 0, aggregated = 1, prefixed = 2
	};

	template <typename T>
	struct block_descriptor {
		block_status status;
		T internal_aggregate;
		T inclusive_prefix;

		__device__
		void store_aggregate(T value) volatile noexcept {
			internal_aggregate = value;
			__threadfence();
			status = block_status::aggregated;
		}

		__device__
		void store_inclusive_prefix(T value) volatile noexcept {
			inclusive_prefix = value;
			__threadfence();
			status = block_status::prefixed;
		}

		__device__
		block_status load_status_spinning() const volatile noexcept {
			block_status ret;
			do ret = this->status; while (block_status::invalid == ret);
			return ret;
		}
	};



	namespace device {

		template <typename T, u32 WarpsPerBlock, u32 ProducerBarrierId, u32 ConsumerBarrierId>
		__device__
		T get_warp_exclusive_prefix(T value, u32 warp_rank, u32 thread_rank) {

			__shared__ T warp_partial_results[WarpsPerBlock];

			if (thread_rank + 1 == 32) {
				warp_partial_results[warp_rank] = value;
			}

			barrier<ProducerBarrierId, WarpsPerBlock>		producer_barrier;
			barrier<ConsumerBarrierId, WarpsPerBlock - 1>	consumer_barrier;

			if (warp_rank == 0) {
				producer_barrier.arrive();

				return identity<T>()();
			}
			else if (warp_rank + 1 != WarpsPerBlock) {
				producer_barrier.arrive();
				
				consumer_barrier.wait();

				return warp_partial_results[warp_rank];
			}
			else {
				producer_barrier.wait();

				if (thread_rank < WarpsPerBlock) {
					cg::coalesced_group active = cg::coalesced_threads();
					T const warp_partial_result = warp_partial_results[thread_rank];
					warp_partial_results[thread_rank] = cg::exclusive_scan(active, warp_partial_result);					
				}

				consumer_barrier.arrive();

				return warp_partial_results[warp_rank];
			}

		}



		template <typename T, u32 WarpsPerBlock, u32 BarrierId>
		__device__
		T get_block_exclusive_prefix(block_descriptor<T> volatile * block_desc, T warp_exclusive_prefix, T warp_reduction, u32 block_rank, u32 warp_rank, u32 thread_rank) {

			T block_exclusive_prefix;
			__shared__ T shared_block_exclusive_prefix;

			barrier<BarrierId, WarpsPerBlock> barrier;

			if (warp_rank == 0) {
				block_exclusive_prefix = identity<T>()();
				
				if (thread_rank == 0) {
					for (i32 i_block = block_rank - 1; 0 <= i_block; --i_block) {
						block_status status = block_desc[i_block].load_status_spinning();
						
						if (block_status::prefixed == status) {
							block_exclusive_prefix += block_desc[i_block].inclusive_prefix;
							break;
						} else {
							block_exclusive_prefix += block_desc[i_block].internal_aggregate;
						}
					}

					shared_block_exclusive_prefix = block_exclusive_prefix;
				}

				barrier.arrive();

				block_exclusive_prefix = __shfl_sync(0xFFFFFFFF, block_exclusive_prefix, 0);

				return block_exclusive_prefix;
			}
			else if (warp_rank + 1 != WarpsPerBlock) {
				barrier.wait();

				block_exclusive_prefix = shared_block_exclusive_prefix;

				return block_exclusive_prefix;
			}
			else {
				if (thread_rank + 1 == 32) {
					block_desc[block_rank].store_aggregate(warp_exclusive_prefix + warp_reduction);
				}

				barrier.wait();
				
				block_exclusive_prefix = shared_block_exclusive_prefix;

				if (thread_rank + 1 == 32) {
					block_desc[block_rank].store_inclusive_prefix(block_exclusive_prefix + warp_exclusive_prefix + warp_reduction);
				}

				return block_exclusive_prefix;
			}

		}

	} // namespace device



	namespace global {

		template <typename T, u32 THREADS_PER_BLOCK, u32 ITEMS_PER_THREAD>
		__global__
		void inclusive_scan(T const * values, T * results, u32 length, block_descriptor<T> volatile * block_desc) {

			constexpr u32 ITEMS_PER_BLOCK	= ITEMS_PER_THREAD * THREADS_PER_BLOCK;
			constexpr u32 ITEMS_PER_WARP	= ITEMS_PER_THREAD * 32;

			auto const grid		= cg::this_grid();
			auto const block	= cg::this_thread_block();
			auto const warp		= cg::tiled_partition<32>(block);

			u32 const block_rank_begin	= grid.block_rank() * ITEMS_PER_BLOCK;
			u32 const warp_rank_begin	= warp.meta_group_rank() * ITEMS_PER_WARP;



			constexpr u32 WARPS_PER_BLOCK	= THREADS_PER_BLOCK / 32;

			u32 constexpr warp_prefix_producer_barrier = 1;
			u32 constexpr warp_prefix_consumer_barrier = 2;
			u32 constexpr block_prefix_producer_barrier = 3;

			T items[ITEMS_PER_THREAD];

			__shared__ T warp_partial_results[WARPS_PER_BLOCK];
			T warp_exclusive_prefix = identity<T>()();

			__shared__ T shared_block_exclusive_prefix;
			T block_exclusive_prefix = identity<T>()();



			for (u32 i_item = 0; i_item < ITEMS_PER_THREAD; ++i_item) {
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

			T const & warp_reduction = items[ITEMS_PER_THREAD - 1];



			warp_exclusive_prefix = device::get_warp_exclusive_prefix<T, WARPS_PER_BLOCK, warp_prefix_producer_barrier, warp_prefix_consumer_barrier>(warp_reduction, warp.meta_group_rank(), warp.thread_rank());

			block_exclusive_prefix = device::get_block_exclusive_prefix<T, WARPS_PER_BLOCK, block_prefix_producer_barrier>(block_desc, warp_exclusive_prefix, warp_reduction, grid.block_rank(), warp.meta_group_rank(), warp.thread_rank());



			for (u32 i_item = 0; i_item < ITEMS_PER_THREAD; ++i_item) {
				items[i_item] += block_exclusive_prefix + warp_exclusive_prefix;
				
				u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_item + warp.thread_rank();
				if (item_rank < length) {
					Byte<sizeof(T)>::copy(results + item_rank, items + i_item);
				}
			}

		}



		template <typename T>
		__global__
		void init_inclusive_scan(block_descriptor<T> volatile * descriptors, u32 num_descriptors) {
			auto const grid = cg::this_grid();

			for (u32 i = grid.thread_rank(); i < num_descriptors; i += grid.num_threads()) {
				descriptors[i].status = block_status::invalid;
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
				temp_storage_bytes = num_blocks * sizeof(block_descriptor<T>);
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