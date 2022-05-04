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



	template <typename T, u32 THREADS_PER_BLOCK, u32 VALUES_PER_THREAD>
	__global__
	void global_inclusive_scan(u32 length, T const * values, T * results) {

		constexpr u32 VALUES_PER_BLOCK	= THREADS_PER_BLOCK * VALUES_PER_THREAD;
		constexpr u32 WARPS_PER_BLOCK	= THREADS_PER_BLOCK / 32;

		constexpr u32 BARRIER_PRODUCER	= 1;
		constexpr u32 BARRIER_CONSUMER	= 2;

		cg::grid_group grid				= cg::this_grid();
		cg::thread_block block			= cg::this_thread_block();
		cg::thread_block_tile<32> warp	= cg::tiled_partition<32>(block);

		u32 const block_rank_begin	= grid.block_rank();
		u32 const block_rank_step	= grid.num_blocks();
		u32 const block_rank_count	= (length + VALUES_PER_BLOCK - 1) / VALUES_PER_BLOCK;



		if (grid.block_rank() == 0 and block.thread_rank() == 0) {
			*reinterpret_cast<T volatile * *>(results) = static_cast<T volatile *>(malloc(sizeof(T) * block_rank_count));
		}

		grid.sync();
		
		T volatile * block_partial_results = *reinterpret_cast<T volatile * *>(results);



		__shared__ T warp_partial_results[WARPS_PER_BLOCK];



		if (warp.meta_group_rank() == 0) {
			ptx_barrier_arrive(BARRIER_CONSUMER, THREADS_PER_BLOCK);
		}

		for (u32 block_rank = block_rank_begin; block_rank < block_rank_count; block_rank += block_rank_step) {
			u32 const value_rank_begin = block_rank * VALUES_PER_BLOCK + block.thread_rank() * VALUES_PER_THREAD;

			T thread_tile[VALUES_PER_THREAD];
			T thread_value = identity<T>();

			if (value_rank_begin + VALUES_PER_THREAD <= length) {
				for (u32 i = 0; i < sizeof(thread_tile) / sizeof(float4); ++i) {
					reinterpret_cast<float4 *>(thread_tile)[i] = reinterpret_cast<float4 const *>(values + value_rank_begin)[i];
				}
			} else {
				for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
					const u32 value_rank = value_rank_begin + i;
					thread_tile[i] = value_rank < length ? values[value_rank] : identity<T>();
				}
			}
			
			for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
				thread_value += thread_tile[i];
			}

			bool const is_thread_active = value_rank_begin < length;
			T thread_reduction = identity<T>();

			if (is_thread_active) {
				cg::coalesced_group active = cg::coalesced_threads();
				thread_reduction = cg::reduce(active, thread_value, cg::plus<T>());	
			}

			if (warp.meta_group_rank() != 0) {
				ptx_barrier_sync(BARRIER_CONSUMER, THREADS_PER_BLOCK);
			}

			if (warp.thread_rank() == 0) {
				warp_partial_results[warp.meta_group_rank()] = thread_reduction;
			}

			if (warp.meta_group_rank() != 0) {
				ptx_barrier_arrive(BARRIER_PRODUCER, THREADS_PER_BLOCK);
			} else {
				ptx_barrier_sync(BARRIER_PRODUCER, THREADS_PER_BLOCK);

				if (warp.thread_rank() < WARPS_PER_BLOCK) {
					T const warp_reduction = warp_partial_results[warp.thread_rank()];
					T const block_reduction = cg::reduce(cg::coalesced_threads(), warp_reduction, cg::plus<T>());
					if (warp.thread_rank() == 0) {
						block_partial_results[block_rank] = block_reduction;
					}
				}
				
				ptx_barrier_arrive(BARRIER_CONSUMER, THREADS_PER_BLOCK);
			}
		}

		grid.sync();

		if (grid.block_rank() == 0) {
			T global_prefix = identity<T>();
			
			for (u32 i = 0; i < (block_rank_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; ++i) {
				u32 agg_rank = i * THREADS_PER_BLOCK + block.thread_rank();
				bool const is_active = agg_rank < block_rank_count;

				T const aggregate = is_active ? block_partial_results[agg_rank] : identity<T>();
				T const local_prefix = exclusive_scan(block, is_active, aggregate);
				if (is_active)
					block_partial_results[agg_rank] = global_prefix + local_prefix;
				
				__shared__ T aggregate_reduction;

				if (block.thread_rank() == block.num_threads() - 1)
					aggregate_reduction = local_prefix + aggregate;
				
				block.sync();
				
				global_prefix += aggregate_reduction;
				
				block.sync();
			}
		}

		grid.sync();

		for (u32 block_rank = block_rank_begin; block_rank < block_rank_count; block_rank += block_rank_step) {
			u32 const value_rank_begin = block_rank * VALUES_PER_BLOCK + block.thread_rank() * VALUES_PER_THREAD;
			
			T thread_tile[VALUES_PER_THREAD];

			if (value_rank_begin + VALUES_PER_THREAD <= length) {
				for (u32 i = 0; i < sizeof(thread_tile) / sizeof(float4); ++i) {
					reinterpret_cast<float4 *>(thread_tile)[i] = reinterpret_cast<float4 const *>(values + value_rank_begin)[i];
				}
			} else {
				for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
					const u32 value_rank = value_rank_begin + i;
					thread_tile[i] = value_rank < length ? values[value_rank] : identity<T>();
				}
			}

			for (u32 i = 1; i < VALUES_PER_THREAD; ++i) {
				thread_tile[i] += thread_tile[i - 1];
			}

			bool const is_active = value_rank_begin < length;
			T const thread_partial_result = exclusive_scan(block, is_active, thread_tile[VALUES_PER_THREAD - 1]);
		
			for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
				thread_tile[i] += block_partial_results[block_rank] + thread_partial_result;
			}

			if (value_rank_begin + VALUES_PER_THREAD <= length) {
				for (u32 i = 0; i < sizeof(thread_tile) / sizeof(float4); ++i) {
					reinterpret_cast<float4 *>(results + value_rank_begin)[i] = reinterpret_cast<float4 const *>(thread_tile)[i];
				}
			} else {
				for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
					const u32 value_rank = value_rank_begin + i;
					if (value_rank < length) {
						results[value_rank] = thread_tile[i];
					}
				}
			}
		}

		grid.sync();

		if (block_rank_begin == 0 and block.thread_rank() == 0)
			free(static_cast<void *>(const_cast<T *>(block_partial_results)));
		
	} // inclusive_scan



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

	template <typename T, u32 THREADS_PER_BLOCK, u32 TILELETS_PER_THREAD, u32 VALUES_PER_TILELET>
	__global__
	void kernel_inclusive_scan_look_back(u32 length, T const * values, T * results, block_descriptor<T> volatile * block_desc, u32 * block_finished_count) {

		constexpr u32 WARPS_PER_BLOCK	= THREADS_PER_BLOCK / 32;

		constexpr u32 VALUES_PER_THREAD	= VALUES_PER_TILELET * TILELETS_PER_THREAD;
		constexpr u32 VALUES_PER_WARP	= VALUES_PER_THREAD * 32;
		constexpr u32 VALUES_PER_BLOCK	= VALUES_PER_THREAD * THREADS_PER_BLOCK;



		constexpr u32 BAR_BLOCK_PRODUCED			= 1;
		constexpr u32 BAR_BLOCK_PRODUCED_THREADS	= THREADS_PER_BLOCK;

		constexpr u32 BAR_WARP_PRODUCED				= 2;
		constexpr u32 BAR_WARP_PRODUCED_THREADS		= THREADS_PER_BLOCK;

		constexpr u32 BAR_WARP_CONSUMED				= 3;
		constexpr u32 BAR_WARP_CONSUMED_THREADS		= THREADS_PER_BLOCK - 32;



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
				
				using bytes = Byte<sizeof(tilelets[i_tilelet])>;
				*reinterpret_cast<bytes *>(tilelets[i_tilelet]) = *reinterpret_cast<bytes const *>(values + tilelet_rank_begin);
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

			ptx_barrier_arrive(BAR_WARP_PRODUCED, BAR_WARP_PRODUCED_THREADS);

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

			ptx_barrier_arrive(BAR_BLOCK_PRODUCED, BAR_BLOCK_PRODUCED_THREADS);
		}
		else if (warp.meta_group_rank() + 1 != warp.meta_group_size()) {
			// warps neither first nor last

			ptx_barrier_arrive(BAR_WARP_PRODUCED, BAR_WARP_PRODUCED_THREADS);

			ptx_barrier_sync(BAR_WARP_CONSUMED, BAR_WARP_CONSUMED_THREADS);
			warp_exclusive_prefix = warp_partial_results[warp.meta_group_rank()];

			ptx_barrier_sync(BAR_BLOCK_PRODUCED, BAR_BLOCK_PRODUCED_THREADS);
			block_exclusive_prefix = shared_block_exclusive_prefix;
		}
		else {
			// the last warp calculates the exclusive prefixes of all warps

			ptx_barrier_sync(BAR_WARP_PRODUCED, BAR_WARP_PRODUCED_THREADS);

			if (warp.thread_rank() < WARPS_PER_BLOCK) {
				cg::coalesced_group active = cg::coalesced_threads();
				T const warp_partial_result = warp_partial_results[warp.thread_rank()];
				warp_partial_results[warp.thread_rank()] = cg::exclusive_scan(active, warp_partial_result);					
			}
			warp_exclusive_prefix = warp_partial_results[warp.meta_group_rank()];

			ptx_barrier_arrive(BAR_WARP_CONSUMED, BAR_WARP_CONSUMED_THREADS);

			if (warp.thread_rank() + 1 == warp.num_threads()) {
				block_desc[grid.block_rank()].aggregate = warp_exclusive_prefix + warp_reduction;
				__threadfence();
				block_desc[grid.block_rank()].status = block_status::aggregated;
			}

			ptx_barrier_sync(BAR_BLOCK_PRODUCED, BAR_BLOCK_PRODUCED_THREADS);
			block_exclusive_prefix = shared_block_exclusive_prefix;

			if (warp.thread_rank() + 1 == warp.num_threads()) {
				block_desc[grid.block_rank()].inclusive_prefix = block_exclusive_prefix + warp_exclusive_prefix + warp_reduction;
				__threadfence();
				block_desc[grid.block_rank()].status = block_status::prefixed;
			}

			if (atomicAdd(block_finished_count, 1) + 1 == grid.num_blocks()) {
				free(const_cast<void *>(reinterpret_cast<void volatile *>(block_desc)));
				free(block_finished_count);
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
				
				using bytes = Byte<sizeof(tilelets[i_tilelet])>;
				*reinterpret_cast<bytes *>(results + tilelet_rank_begin) = *reinterpret_cast<bytes const *>(tilelets[i_tilelet]);
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



	template <typename T, u32 THREADS_PER_BLOCK, u32 TILELETS_PER_THREAD, u32 VALUES_PER_TILELET>
	__global__
	void kernel_inclusive_scan_entry(u32 length, T const * values, T * results) {

		constexpr u32 VALUES_PER_BLOCK = VALUES_PER_TILELET * TILELETS_PER_THREAD * THREADS_PER_BLOCK;

		cg::thread_block block	= cg::this_thread_block();

		u32 const block_dim			= THREADS_PER_BLOCK;
		u32 const grid_dim			= (length + VALUES_PER_BLOCK - 1) / VALUES_PER_BLOCK;
		u32 const size_in_bytes		= sizeof(block_descriptor<T>) * grid_dim;

		block_descriptor<T> volatile * block_desc;
		u32 * block_finished_count;
		__shared__ block_descriptor<T> volatile * shared_block_desc;

		if (block.thread_rank() == 0) {
			block_desc = static_cast<block_descriptor<T> *>(malloc(size_in_bytes));
			shared_block_desc = block_desc;

			block_finished_count = static_cast<u32 *>(malloc(sizeof(u32)));
			*block_finished_count = 0;
		}

		block.sync();

		block_desc = shared_block_desc;

		block_memset(block, block_desc, size_in_bytes, u8(0));

		block.sync();

		if (block.thread_rank() == 0) {
			kernel_inclusive_scan_look_back<T, THREADS_PER_BLOCK, TILELETS_PER_THREAD, VALUES_PER_TILELET><<<grid_dim, block_dim>>>(length, values, results, block_desc, block_finished_count);
			cudaError_t e = cudaGetLastError();
			if (e != cudaSuccess) {
				printf("error %d\n", int(e));
			}
		}

	}



	template <typename T>
	cudaError_t inclusive_scan(u32 length, T const * values, T * results) {
		constexpr u32 threads_per_block = 512;
		constexpr u32 tilelets_per_thread = 4;
		constexpr u32 values_per_tilelet = 4;

		void * fn = (void *)kernel_inclusive_scan_entry<T, threads_per_block, tilelets_per_thread, values_per_tilelet>;
		dim3 const grid_dim(1, 1, 1);
		dim3 const block_dim(64, 1, 1);
		void * args[] = { &length, &values, &results };

		return cudaLaunchKernel(fn, grid_dim, block_dim, args);

		// cudaDeviceProp device_prop;
		// cudaGetDeviceProperties(&device_prop, 0);

		// i32 blocks_per_sm = 0;
		// cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, threads_per_block, 0);

		// printf("Max Blocks Per Processor: %d\n", device_prop.maxBlocksPerMultiProcessor);
		// printf("Max Threads Per Processor: %d\n", device_prop.maxThreadsPerMultiProcessor);
		// printf("Queried Number of Processor: %d\n", device_prop.multiProcessorCount);
		// printf("Queried Blocks per Processor: %d\n", blocks_per_sm);
	}

}



#endif // SPP_REDUCE_HPP