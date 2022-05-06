#ifndef SPP_REDUCE_HPP
#define SPP_REDUCE_HPP

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#include "types.hpp"



namespace spp {

	namespace cg = ::cooperative_groups;



	template <typename T>
	__host__ __device__
	constexpr T identity() { return T(0); }



	#define ptx_barrier_arrive(id, count) asm volatile ("barrier.arrive %0, %1;" : : "n"(id), "n"(count))
	#define ptx_barrier_sync(id, count) asm volatile ("barrier.sync %0, %1;" : : "n"(id), "n"(count))

	template <u8 Id, u8 WarpCount>
	struct barrier {
		__device__
		void wait() const noexcept {
			ptx_barrier_sync(Id, WarpCount * 32);
		}

		__device__
		void arrive() const noexcept {
			ptx_barrier_arrive(Id, WarpCount * 32);
		}
	};



	__host__ __device__
	constexpr u32 ceiled_div(u32 dividend, u32 divisor) {
		return (dividend + divisor - 1) / divisor;
	}



	BEGIN_NAMESPACE(kernel)



	enum struct block_status : u32 {
		invalid = 0, aggregated = 1, prefixed = 2
	};

	template <typename T>
	struct block_descriptor {
		block_status status;
		T aggregate;
		T inclusive_prefix;
	};



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

		barrier<1, WARPS_PER_BLOCK>		bar_block_prefix_produced;
		barrier<2, WARPS_PER_BLOCK>		bar_warp_prefix_produced;
		barrier<3, WARPS_PER_BLOCK - 1>	bar_warp_prefix_consumed;

		T items[ITEMS_PER_THREAD];

		__shared__ T warp_partial_results[WARPS_PER_BLOCK];
		T warp_exclusive_prefix = identity<T>();

		__shared__ T shared_block_exclusive_prefix;
		T block_exclusive_prefix = identity<T>();



		for (u32 i_item = 0; i_item < ITEMS_PER_THREAD; ++i_item) {
			u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_item + warp.thread_rank();
			items[i_item] = identity<T>();
			if (item_rank < length) {
				Byte<sizeof(T)>::copy(items + i_item, values + item_rank);
			}
			items[i_item] = cg::inclusive_scan(warp, items[i_item]);
			if (i_item) {
				items[i_item] += warp.shfl(items[i_item - 1], 32 - 1);
			}
		}

		T const & warp_reduction = items[ITEMS_PER_THREAD - 1];



		if (warp.thread_rank() + 1 == warp.num_threads()) {
			warp_partial_results[warp.meta_group_rank()] = warp_reduction;
		}

		if (warp.meta_group_rank() == 0) {
			// the first warp calculates the exclusive prefix of this block

			bar_warp_prefix_produced.arrive();

			if (warp.thread_rank() == 0) {
				for (i32 i = grid.block_rank() - 1; 0 <= i; --i) {
					block_status status;
					do status = block_desc[i].status; while (block_status::invalid == status);
					if (block_status::prefixed == status) {
						block_exclusive_prefix += block_desc[i].inclusive_prefix;
						break;
					} else {
						block_exclusive_prefix += block_desc[i].aggregate;
					}
				}
			}

			block_exclusive_prefix = warp.shfl(block_exclusive_prefix, 0);

			if (warp.thread_rank() == 0) {
				shared_block_exclusive_prefix = block_exclusive_prefix;
			}

			bar_block_prefix_produced.arrive();
		}
		else if (warp.meta_group_rank() + 1 != warp.meta_group_size()) {
			// warps neither first nor last

			bar_warp_prefix_produced.arrive();
			bar_warp_prefix_consumed.wait();
			warp_exclusive_prefix = warp_partial_results[warp.meta_group_rank()];

			bar_block_prefix_produced.wait();
			block_exclusive_prefix = shared_block_exclusive_prefix;
		}
		else {
			// the last warp calculates the exclusive prefixes of all warps

			bar_warp_prefix_produced.wait();

			if (warp.thread_rank() < WARPS_PER_BLOCK) {
				cg::coalesced_group active = cg::coalesced_threads();
				T const warp_partial_result = warp_partial_results[warp.thread_rank()];
				warp_partial_results[warp.thread_rank()] = cg::exclusive_scan(active, warp_partial_result);					
			}
			warp_exclusive_prefix = warp_partial_results[warp.meta_group_rank()];

			bar_warp_prefix_consumed.arrive();

			if (warp.thread_rank() + 1 == warp.num_threads()) {
				block_desc[grid.block_rank()].aggregate = warp_exclusive_prefix + warp_reduction;
				__threadfence();
				block_desc[grid.block_rank()].status = block_status::aggregated;
			}

			bar_block_prefix_produced.wait();
			block_exclusive_prefix = shared_block_exclusive_prefix;

			if (warp.thread_rank() + 1 == warp.num_threads()) {
				block_desc[grid.block_rank()].inclusive_prefix = block_exclusive_prefix + warp_exclusive_prefix + warp_reduction;
				__threadfence();
				block_desc[grid.block_rank()].status = block_status::prefixed;
			}
		}



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



	END_NAMESPACE(kernel)



	template <typename T>
	cudaError_t inclusive_scan(T const * values, T * results, u32 length, void * temp_storage, u32 & temp_storage_bytes) {
		constexpr u32 threads_per_block = 512;
		constexpr u32 items_per_thread = 16;
		
		constexpr u32 items_per_block = items_per_thread * threads_per_block;

		auto num_blocks = ceiled_div(length, items_per_block);

		if (temp_storage == nullptr || temp_storage_bytes == 0) {
			temp_storage_bytes = num_blocks * sizeof(kernel::block_descriptor<T>);
			return cudaSuccess;
		}
		else {
			/* init */ {
				auto fn			= reinterpret_cast<void const *>(kernel::init_inclusive_scan<T>);
				auto block_dim	= dim3(128, 1, 1);
				auto grid_dim	= dim3(ceiled_div(num_blocks, block_dim.x), 1, 1);
				void * args[]	= { &temp_storage, &num_blocks };
				
				if (auto e = cudaLaunchKernel(fn, grid_dim, block_dim, args); cudaSuccess != e) {
					return e;
				}
			}

			/* scan */ {
				auto fn			= reinterpret_cast<void const *>(kernel::inclusive_scan<T, threads_per_block, items_per_thread>);
				auto grid_dim	= dim3(num_blocks, 1, 1);
				auto block_dim	= dim3(threads_per_block, 1, 1);
				void * args[] = { &values, &results, &length, &temp_storage };

				return cudaLaunchKernel(fn, grid_dim, block_dim, args);
			}
		}
	}

}



#endif // SPP_REDUCE_HPP