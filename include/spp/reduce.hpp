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

	template <typename T, u32 S, u32 ID = 0>
	__device__
	T * unsafe_shared() {
		__shared__ T buffer[S];
		return buffer;
	}



	template <typename T, bool Sync = true>
	__device__
	T reduce(cg::thread_block const & block, bool is_active, T const & value) {
		
		cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

		u32 const warp_rank		= warp.meta_group_rank();
		u32 const num_warps		= warp.meta_group_size();
		u32 const thread_rank	= warp.thread_rank();

		bool * is_warp_active	= unsafe_shared<bool, 32>();
		T * buffer				= unsafe_shared<T, 32>();

		bool const is_any_thread_active = warp.any(is_active);
		if (thread_rank == 0) is_warp_active[warp_rank] = is_any_thread_active;

		if (is_active) {
			cg::coalesced_group active = cg::coalesced_threads();
			T const partial_result = reduce(active, value, cg::plus<T>());
			if (thread_rank == 0) buffer[warp_rank] = partial_result;
		}

		block.sync();
			
		if (warp_rank == 0 and thread_rank < num_warps and is_warp_active[thread_rank]) {
			cg::coalesced_group active = cg::coalesced_threads();
			buffer[thread_rank] = reduce(active, buffer[thread_rank], cg::plus<T>());
		}

		block.sync();

		T const final_result = buffer[warp_rank];

		if constexpr (Sync) block.sync();

		return final_result;

	}



	template <typename T, bool Sync = true>
	__device__
	T inclusive_scan(cg::thread_block const & block, bool is_active, T const & value) {
		
		cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

		u32 const warp_rank = warp.meta_group_rank();
		u32 const num_warps = warp.meta_group_size();
		
		u32 const thread_rank = warp.thread_rank();
		u32 const num_threads = warp.num_threads();

		bool * is_warp_active = unsafe_shared<bool, 32>();
		T * buffer = unsafe_shared<T, 32>();

		bool const is_any_thread_active = warp.any(is_active);
		if (thread_rank == 0) is_warp_active[warp_rank] = is_any_thread_active;

		T thread_partial_result;

		if (is_active) {
			cg::coalesced_group active = cg::coalesced_threads();
			thread_partial_result = inclusive_scan(active, value, cg::plus<T>());
			if (thread_rank == num_threads - 1) buffer[warp_rank] = thread_partial_result;
		}

		block.sync();

		if (warp_rank == 0 and thread_rank < num_warps and is_warp_active[thread_rank]) {
			cg::coalesced_group active = cg::coalesced_threads();
			buffer[thread_rank] = exclusive_scan(active, buffer[thread_rank], cg::plus<T>());
		}

		block.sync();

		T const final_result = buffer[warp_rank] + thread_partial_result;
		
		if constexpr (Sync) block.sync();
		
		return final_result;
	}



	template <typename T, bool Sync = true>
	__device__
	T exclusive_scan(cg::thread_block const & block, bool is_active, T const & value) {
		return inclusive_scan<T, Sync>(block, is_active, value) - value;
	}



	#define ptx_barrier_arrive(id, count) asm volatile ("barrier.arrive %0, %1;" : : "n"(id), "n"(count))
	#define ptx_barrier_sync(id, count) asm volatile ("barrier.sync %0, %1;" : : "n"(id), "n"(count))



	__host__ __device__
	constexpr u32 ceiled_div(u32 dividend, u32 divisor) {
		return (dividend + divisor - 1) / divisor;
	}

	enum struct block_status : u32 {
		invalid = 0, aggregated = 1, prefixed = 2
	};

	template <typename T>
	struct block_descriptor {
		block_status status;
		T aggregate;
		T inclusive_prefix;
	};

	__device__
	void block_memset(cg::thread_block const & block, void volatile * dest, u32 bytes, u8 value) {
		for (u32 i = block.thread_rank(); i < bytes; i += block.num_threads()) {
			reinterpret_cast<u8 volatile *>(dest)[i] = value;
		}
	}



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



	template <typename T, u32 THREADS_PER_BLOCK, u32 TILELETS_PER_THREAD, u32 VALUES_PER_TILELET>
	__global__
	void device_inclusive_scan(T const * values, T * results, u32 length, block_descriptor<T> volatile * block_desc) {

		constexpr u32 WARPS_PER_BLOCK	= THREADS_PER_BLOCK / 32;

		constexpr u32 VALUES_PER_THREAD	= VALUES_PER_TILELET * TILELETS_PER_THREAD;
		constexpr u32 VALUES_PER_WARP	= VALUES_PER_THREAD * 32;
		constexpr u32 VALUES_PER_BLOCK	= VALUES_PER_THREAD * THREADS_PER_BLOCK;



		barrier<1, WARPS_PER_BLOCK>		bar_block_prefix_produced;
		barrier<2, WARPS_PER_BLOCK>		bar_warp_prefix_produced;
		barrier<3, WARPS_PER_BLOCK - 1>	bar_warp_prefix_consumed;



		cg::grid_group const grid				= cg::this_grid();
		cg::thread_block const block			= cg::this_thread_block();
		cg::thread_block_tile<32> const warp	= cg::tiled_partition<32>(block);

		u32 const block_rank_begin	= grid.block_rank() * VALUES_PER_BLOCK;
		u32 const warp_rank_begin	= warp.meta_group_rank() * VALUES_PER_WARP;



		/*
			(tilelet, thread):	| 0, 0 | 0, 1 | ... | 0, 31 | 1, 0 | 1, 1 | ... | 1, 31 | 2, 0 | 2, 1 | ...
			cluster:			| 0							| 1							| 2
			| tilelet 0, thread 0 | tilelet 0, thread 
		*/

		T tilelets[TILELETS_PER_THREAD][VALUES_PER_TILELET];
		__shared__ T tilelet_exclusive_prefixes[TILELETS_PER_THREAD][THREADS_PER_BLOCK];
		__shared__ T cluster_exclusive_prefixes[TILELETS_PER_THREAD + 1][WARPS_PER_BLOCK];

		if (warp.thread_rank() == 0) {
			cluster_exclusive_prefixes[0][warp.meta_group_rank()] = identity<T>();
		}

		if (block_rank_begin + warp_rank_begin + VALUES_PER_WARP <= length) {
			for (u32 i_tilelet = 0; i_tilelet < TILELETS_PER_THREAD; ++i_tilelet) {
				u32 const tilelet_rank_begin = block_rank_begin + warp_rank_begin + (32 * i_tilelet + warp.thread_rank()) * VALUES_PER_TILELET;
				
				Byte<sizeof(tilelets[i_tilelet])>::copy(tilelets[i_tilelet], values + tilelet_rank_begin);
			}
		}
		else {
			for (u32 i_tilelet = 0; i_tilelet < TILELETS_PER_THREAD; ++i_tilelet) {
				u32 const tilelet_rank_begin = block_rank_begin + warp_rank_begin + (32 * i_tilelet + warp.thread_rank()) * VALUES_PER_TILELET;

				for (u32 j_value = 0; j_value < VALUES_PER_TILELET; ++j_value) {
					u32 const value_rank = tilelet_rank_begin + j_value;
					T const value = value_rank < length ? values[value_rank] : identity<T>();
					tilelets[i_tilelet][j_value] = value;
				}
			}
		}

		for (u32 i_tilelet = 0; i_tilelet < TILELETS_PER_THREAD; ++i_tilelet) {
			for (u32 j_value = 1; j_value < VALUES_PER_TILELET; ++j_value) {
				tilelets[i_tilelet][j_value] += tilelets[i_tilelet][j_value - 1];
			}
			
			tilelet_exclusive_prefixes[i_tilelet][block.thread_rank()] = cg::exclusive_scan(warp, tilelets[i_tilelet][VALUES_PER_TILELET - 1]);

			T const tilelet_inclusive_prefix = warp.shfl(tilelet_exclusive_prefixes[i_tilelet][block.thread_rank()] + tilelets[i_tilelet][VALUES_PER_TILELET - 1], 32 - 1);
			
			if (warp.thread_rank() == 0) {
				cluster_exclusive_prefixes[1 + i_tilelet][warp.meta_group_rank()] = tilelet_inclusive_prefix + cluster_exclusive_prefixes[i_tilelet][warp.meta_group_rank()];
			}
		}

		T const & warp_reduction = cluster_exclusive_prefixes[TILELETS_PER_THREAD][warp.meta_group_rank()];



		__shared__ T warp_partial_results[WARPS_PER_BLOCK];

		__shared__ T shared_block_exclusive_prefix;

		T warp_exclusive_prefix		= identity<T>();
		T block_exclusive_prefix 	= identity<T>();

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



		for (u32 i_tilelet = 0; i_tilelet < TILELETS_PER_THREAD; ++i_tilelet) {
			for (u32 j_value = 0; j_value < VALUES_PER_TILELET; ++j_value) {
				tilelets[i_tilelet][j_value] += block_exclusive_prefix
					+ warp_exclusive_prefix
					+ cluster_exclusive_prefixes[i_tilelet][warp.meta_group_rank()]
					+ tilelet_exclusive_prefixes[i_tilelet][block.thread_rank()];
			}
		}

		if (block_rank_begin + warp_rank_begin + VALUES_PER_WARP <= length) {
			for (u32 i_tilelet = 0; i_tilelet < TILELETS_PER_THREAD; ++i_tilelet) {
				u32 const tilelet_rank_begin = block_rank_begin + warp_rank_begin + (32 * i_tilelet + warp.thread_rank()) * VALUES_PER_TILELET;
				
				Byte<sizeof(tilelets[i_tilelet])>::copy(results + tilelet_rank_begin, tilelets[i_tilelet]);
			}
		}
		else {
			for (u32 i_tilelet = 0; i_tilelet < TILELETS_PER_THREAD; ++i_tilelet) {				
				u32 const tilelet_rank_begin = block_rank_begin + warp_rank_begin + (32 * i_tilelet + warp.thread_rank()) * VALUES_PER_TILELET;
				
				for (u32 j_value = 0; j_value < VALUES_PER_TILELET; ++j_value) {
					u32 const value_rank = tilelet_rank_begin + j_value;
					
					if (value_rank < length) {
						results[value_rank] = tilelets[i_tilelet][j_value];
					}
				}
			}
		}

	}



	template <typename T>
	__global__
	void device_init_inclusive_scan(block_descriptor<T> volatile * descriptors, u32 num_descriptors) {
		auto const grid = cg::this_grid();

		for (u32 i = grid.thread_rank(); i < num_descriptors; i += grid.num_threads()) {
			descriptors[i].status = block_status::invalid;
		}
	}



	template <typename T>
	cudaError_t inclusive_scan(T const * values, T * results, u32 length, void * temp_storage, u32 & temp_storage_bytes) {
		constexpr u32 threads_per_block = 512;
		constexpr u32 tilelets_per_thread = 4;
		constexpr u32 values_per_tilelet = 4;
		
		constexpr u32 values_per_block = values_per_tilelet * tilelets_per_thread * threads_per_block;

		auto num_blocks = ceiled_div(length, values_per_block);

		if (temp_storage == nullptr || temp_storage_bytes == 0) {
			temp_storage_bytes = num_blocks * sizeof(block_descriptor<T>);
			return cudaSuccess;
		}
		else {
			/* init */ {
				auto fn			= reinterpret_cast<void const *>(device_init_inclusive_scan<T>);
				auto block_dim	= dim3(128, 1, 1);
				auto grid_dim	= dim3(ceiled_div(num_blocks, block_dim.x), 1, 1);
				void * args[]	= { &temp_storage, &num_blocks };
				
				if (auto e = cudaLaunchKernel(fn, grid_dim, block_dim, args); cudaSuccess != e) {
					return e;
				}
			}

			/* scan */ {
				auto fn			= reinterpret_cast<void const *>(device_inclusive_scan<T, threads_per_block, tilelets_per_thread, values_per_tilelet>);
				auto grid_dim	= dim3(num_blocks, 1, 1);
				auto block_dim	= dim3(threads_per_block, 1, 1);
				void * args[] = { &values, &results, &length, &temp_storage };

				return cudaLaunchKernel(fn, grid_dim, block_dim, args);
			}
		}
	}

}



#endif // SPP_REDUCE_HPP