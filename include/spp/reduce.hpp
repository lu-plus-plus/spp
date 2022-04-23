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

	template <typename T>
	struct warp_descriptor {
		u32 status;
		T aggregate;
		T inclusive_prefix;
	};

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

	template <typename T, u32 THREADS_PER_BLOCK, u32 VALUES_PER_THREAD>
	__global__
	void kernel_inclusive_scan_look_back(u32 length, T const * values, T * results) {

		constexpr u32 VALUES_PER_BLOCK	= VALUES_PER_THREAD * THREADS_PER_BLOCK;
		constexpr u32 WARPS_PER_BLOCK	= THREADS_PER_BLOCK / 32;

		cg::grid_group grid				= cg::this_grid();
		cg::thread_block block			= cg::this_thread_block();
		cg::thread_block_tile<32> warp	= cg::tiled_partition<32>(block);



		constexpr u32 BAR_BLOCK_PRODUCED			= 1;
		constexpr u32 BAR_BLOCK_PRODUCED_THREADS	= THREADS_PER_BLOCK;

		constexpr u32 BAR_BLOCK_CONSUMED			= 2;
		constexpr u32 BAR_BLOCK_CONSUMED_THREADS	= THREADS_PER_BLOCK;

		constexpr u32 BAR_WARP_PRODUCED				= 3;
		constexpr u32 BAR_WARP_PRODUCED_THREADS		= THREADS_PER_BLOCK;

		constexpr u32 BAR_OTHER_WARPS_CONSUMED			= 4;
		constexpr u32 BAR_OTHER_WARPS_CONSUMED_THREADS	= THREADS_PER_BLOCK - 32;

		constexpr u32 BAR_FIRST_WARP_CONSUMED			= 5;
		constexpr u32 BAR_FIRST_WARP_CUNSUMED_THREADS	= 32 * 2;



		block_descriptor<T> volatile * block_desc;

		if (grid.block_rank() == 0) {
			u32 const size_in_bytes = sizeof(block_descriptor<T>) * grid.num_blocks();

			__shared__ block_descriptor<T> volatile * s_block_desc;

			if (block.thread_rank() == 0) {
				block_desc = static_cast<block_descriptor<T> *>(malloc(size_in_bytes));
				s_block_desc = block_desc;
			}
			
			block.sync();

			if (block.thread_rank() == 0) {
				*reinterpret_cast<block_descriptor<T> volatile * *>(results) = block_desc;
			} else {
				block_desc = s_block_desc;
			}

			block_memset(block, block_desc, size_in_bytes, u8(0));
		}

		grid.sync();

		if (grid.block_rank() != 0) {
			block_desc = *reinterpret_cast<block_descriptor<T> * *>(results);
		}

		grid.sync();



		__shared__ T warp_partial_results[WARPS_PER_BLOCK];



		u32 const block_rank_begin		= grid.block_rank();
		u32 const block_rank_step		= grid.num_blocks();
		u32 const block_rank_count		= ceiled_div(length, VALUES_PER_BLOCK);

		if (warp.meta_group_rank() != 0) {
			ptx_barrier_arrive(BAR_BLOCK_CONSUMED, BAR_BLOCK_CONSUMED_THREADS);
		}

		if (warp.meta_group_rank() + 1 == warp.meta_group_size()) {
			ptx_barrier_arrive(BAR_FIRST_WARP_CONSUMED, BAR_FIRST_WARP_CUNSUMED_THREADS);
		}

		for (u32 block_rank = block_rank_begin; block_rank < block_rank_count; block_rank += block_rank_step) {
			
			u32 const value_rank_begin		= block_rank * VALUES_PER_BLOCK + block.thread_rank() * VALUES_PER_THREAD;

			// bool const is_thread_active		= value_rank_begin < length;
			// bool const is_warp_active		= warp.any(is_thread_active);

			T thread_tile[VALUES_PER_THREAD];
			T thread_reduction = identity<T>();

			__shared__ T shared_block_exclusive_prefix;

			if (value_rank_begin + VALUES_PER_THREAD <= length) {
				for (u32 i = 0; i < sizeof(thread_tile) / sizeof(float4); ++i) {
					*(reinterpret_cast<float4 *>(thread_tile) + i) = *(reinterpret_cast<float4 const *>(values + value_rank_begin) + i);
				}
				for (u32 i = 1; i < VALUES_PER_THREAD; ++i) {
					thread_tile[i] += thread_tile[i - 1];
				}
				thread_reduction = thread_tile[VALUES_PER_THREAD - 1];
			} else {
				for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
					u32 const value_rank = value_rank_begin + i;
					if (value_rank >= length) break;
					
					T const value = values[value_rank];
					
					thread_reduction += value;
					thread_tile[i] = thread_reduction;
				}
			}

			T thread_exclusive_prefix	= cg::exclusive_scan(warp, thread_reduction);
			T warp_exclusive_prefix		= identity<T>();
			T block_exclusive_prefix 	= identity<T>();

			if (warp.meta_group_rank() == 0) {
				ptx_barrier_sync(BAR_FIRST_WARP_CONSUMED, BAR_FIRST_WARP_CUNSUMED_THREADS);
			}

			if (warp.thread_rank() + 1 == warp.num_threads()) {
				warp_partial_results[warp.meta_group_rank()] = thread_exclusive_prefix + thread_reduction;
			}

			if (warp.meta_group_rank() == 0) {
				// the first warp calculates the exclusive prefix of this block

				ptx_barrier_arrive(BAR_WARP_PRODUCED, BAR_WARP_PRODUCED_THREADS);

				if (warp.thread_rank() == 0) {
					for (i32 i = block_rank - 1; 0 <= i; --i) {
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

				ptx_barrier_sync(BAR_BLOCK_CONSUMED, BAR_BLOCK_CONSUMED_THREADS);
				if (warp.thread_rank() == 0) {
					shared_block_exclusive_prefix = block_exclusive_prefix;
				}
				ptx_barrier_arrive(BAR_BLOCK_PRODUCED, BAR_BLOCK_PRODUCED_THREADS);
			}
			else if (warp.meta_group_rank() + 1 != warp.meta_group_size()) {
				// warps neither first nor last

				ptx_barrier_arrive(BAR_WARP_PRODUCED, BAR_WARP_PRODUCED_THREADS);

				ptx_barrier_sync(BAR_OTHER_WARPS_CONSUMED, BAR_OTHER_WARPS_CONSUMED_THREADS);
				warp_exclusive_prefix = warp_partial_results[warp.meta_group_rank()];

				ptx_barrier_sync(BAR_BLOCK_PRODUCED, BAR_BLOCK_PRODUCED_THREADS);
				block_exclusive_prefix = shared_block_exclusive_prefix;
				ptx_barrier_arrive(BAR_BLOCK_CONSUMED, BAR_BLOCK_CONSUMED_THREADS);
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

				ptx_barrier_arrive(BAR_FIRST_WARP_CONSUMED, BAR_FIRST_WARP_CUNSUMED_THREADS);
				ptx_barrier_arrive(BAR_OTHER_WARPS_CONSUMED, BAR_OTHER_WARPS_CONSUMED_THREADS);

				if (warp.thread_rank() + 1 == warp.num_threads()) {
					block_desc[block_rank].aggregate = warp_exclusive_prefix + thread_exclusive_prefix + thread_reduction;
					__threadfence();
					block_desc[block_rank].status = block_status::aggregated;
				}

				ptx_barrier_sync(BAR_BLOCK_PRODUCED, BAR_BLOCK_PRODUCED_THREADS);
				block_exclusive_prefix = shared_block_exclusive_prefix;
				ptx_barrier_arrive(BAR_BLOCK_CONSUMED, BAR_BLOCK_CONSUMED_THREADS);

				if (warp.thread_rank() + 1 == warp.num_threads()) {
					block_desc[block_rank].inclusive_prefix = block_exclusive_prefix + warp_exclusive_prefix + thread_exclusive_prefix + thread_reduction;
					__threadfence();
					block_desc[block_rank].status = block_status::prefixed;
				}
			}



			if (value_rank_begin + 4 <= length) {
				for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
					thread_tile[i] += block_exclusive_prefix + warp_exclusive_prefix + thread_exclusive_prefix;
				}
				for (u32 i = 0; i < sizeof(thread_tile) / sizeof(float4); ++i) {
					*(reinterpret_cast<float4 *>(results + value_rank_begin) + i) = *(reinterpret_cast<float4 const *>(thread_tile) + i);
				}
			} else {
				for (u32 i = 0; i < VALUES_PER_THREAD; ++i) {
					u32 const value_rank = value_rank_begin + i;
					if (value_rank >= length) break;
					
					results[value_rank] = block_exclusive_prefix + warp_exclusive_prefix + thread_exclusive_prefix + thread_tile[i];
				}
			}

			if (block_rank + 1 == block_rank_count and block.thread_rank() + 1 == block.num_threads()) {
				free(const_cast<void *>(reinterpret_cast<void volatile *>(block_desc)));
			}
			
		} // for block_rank

	}



	template <typename T>
	cudaError_t inclusive_scan(u32 length, T const * values, T * results) {
		constexpr u32 threads_per_block = 1024;
		constexpr u32 items_per_thread = 16;

		void * fn = (void *)kernel_inclusive_scan_look_back<T, threads_per_block, items_per_thread>;

		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, 0);

		i32 blocks_per_sm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, threads_per_block, 0);

		// printf("Max Blocks Per Processor: %d\n", device_prop.maxBlocksPerMultiProcessor);
		// printf("Max Threads Per Processor: %d\n", device_prop.maxThreadsPerMultiProcessor);
		// printf("Queried Number of Processor: %d\n", device_prop.multiProcessorCount);
		// printf("Queried Blocks per Processor: %d\n", blocks_per_sm);

		dim3 const grid_dim(blocks_per_sm * device_prop.multiProcessorCount, 1, 1);
		dim3 const block_dim(threads_per_block, 1, 1);
		void * args[] = { &length, &values, &results };
		return cudaLaunchCooperativeKernel(fn, grid_dim, block_dim, args);
	}

}



#endif // SPP_REDUCE_HPP