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
	T identity() { return T(0); }

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



	template <typename T, u32 BLOCK_THREADS, u32 THREAD_ITEMS>
	__global__
	void global_inclusive_scan(u32 length, T const * values, T * results) {

		constexpr u32 BLOCK_ITEMS	= BLOCK_THREADS * THREAD_ITEMS;

		cg::grid_group grid			= cg::this_grid();
		cg::thread_block block		= cg::this_thread_block();

		u32 const block_rank_begin	= grid.block_rank();
		u32 const block_rank_step	= grid.num_blocks();
		u32 const block_rank_end	= (length + BLOCK_ITEMS - 1) / BLOCK_ITEMS;

		u32 const thread_rank		= block.thread_rank();

		if (block_rank_begin == 0 and thread_rank == 0) {
			*reinterpret_cast<T * *>(results) = static_cast<T *>(malloc(sizeof(T) * block_rank_end));
		}

		grid.sync();
		
		T * block_aggregates = *reinterpret_cast<T * *>(results);

		for (u32 block_rank = block_rank_begin; block_rank < block_rank_end; block_rank += block_rank_step) {
			u32 const item_rank_begin = block_rank * BLOCK_ITEMS + thread_rank * THREAD_ITEMS;

			bool const is_active = item_rank_begin < length;
			T thread_item = identity<T>();
			for (u32 i = 0; i < THREAD_ITEMS; ++i) {
				const u32 item_rank = item_rank_begin + i;
				if (not (item_rank < length)) break;
				thread_item += values[item_rank];
			}
			T const block_reduction = reduce(block, is_active, thread_item);

			if (thread_rank == 0) block_aggregates[block_rank] = block_reduction;
		}

		grid.sync();

		if (grid.block_rank() == 0) {
			T global_prefix = identity<T>();
			
			for (u32 i = 0; i < (block_rank_end + BLOCK_THREADS - 1) / BLOCK_THREADS; ++i) {
				u32 agg_rank = i * BLOCK_THREADS + thread_rank;
				bool const is_active = agg_rank < block_rank_end;

				T const aggregate = is_active ? block_aggregates[agg_rank] : identity<T>();
				T const local_prefix = exclusive_scan(block, is_active, aggregate);
				if (is_active)
					block_aggregates[agg_rank] = global_prefix + local_prefix;
				
				__shared__ T aggregate_reduction;

				if (block.thread_rank() == block.num_threads() - 1)
					aggregate_reduction = local_prefix + aggregate;
				
				block.sync();
				
				global_prefix += aggregate_reduction;
				
				block.sync();
			}
		}

		grid.sync();

		for (u32 block_rank = block_rank_begin; block_rank < block_rank_end; block_rank += block_rank_step) {
			u32 const item_rank_begin = block_rank * BLOCK_ITEMS + thread_rank * THREAD_ITEMS;

			bool const is_active = item_rank_begin < length;
			
			T thread_items[THREAD_ITEMS] = {};

			for (u32 i = 0; i < THREAD_ITEMS; ++i) {
				u32 const item_rank = item_rank_begin + i;
				if (not (item_rank < length)) break;
				T const item = values[item_rank];
				for (u32 j = i; j < THREAD_ITEMS; ++j) thread_items[j] += item;
			}

			T const thread_partial_result = exclusive_scan(block, is_active, thread_items[THREAD_ITEMS - 1]);
		
			for (u32 i = 0; i < THREAD_ITEMS; ++i) {
				u32 const item_rank = item_rank_begin + i;
				if (not (item_rank < length)) break;
				results[item_rank] = block_aggregates[block_rank] + thread_partial_result + thread_items[i];
			}
		}

		grid.sync();

		if (block_rank_begin == 0 and thread_rank == 0)
			free(block_aggregates);
		
	} // inclusive_scan



	template <typename T>
	cudaError_t inclusive_scan(u32 length, T const * values, T * results) {
		constexpr u32 threads_per_block = 128;
		constexpr u32 items_per_thread = 4;

		void * fn = (void *)global_inclusive_scan<T, threads_per_block, items_per_thread>;

		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, 0);

		i32 blocks_per_sm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, threads_per_block, 0);

		// printf("Number of Processor: %d\n", device_prop.multiProcessorCount);
		// printf("Blocks per Processor: %d\n", blocks_per_sm);

		dim3 const grid_dim(blocks_per_sm * device_prop.multiProcessorCount, 1, 1);
		dim3 const block_dim(threads_per_block, 1, 1);
		void * args[] = { &length, &values, &results };
		return cudaLaunchCooperativeKernel(fn, grid_dim, block_dim, args);
	}



	// template <typename T, typename K>
	// __device__
	// void warp_segmented_reduce(T & value, K & key, T * buffer, uint32_t buffer_base) {
	// 	for (uint32_t delta = 1; delta < 32; delta *= 2) {
	// 		K const neighbor_key = __shfl_down_sync(0xFFFFFFFFu, key, delta);
			
	// 		bool const is_partial = key == neighbor_key;
	// 		bool const any_partial = __any_sync(0xFFFFFFFFu, is_partial);
	// 		if (not any_partial) break;

	// 		if constexpr (warp_shufflable_v<T>) {
	// 			if (is_partial) value += __shfl_down_sync(0xFFFFFFFFu, value, delta);
	// 		} else {
	// 			if (is_partial) buffer[buffer_base] = value;
	// 			__syncwarp();
	// 			if (is_partial) value += buffer[buffer_base + delta];
	// 			__syncwarp();
	// 		}
	// 	}
	// }



	// // template <typename Value, typename Index>
	// // void reduce(Value const * values, Index const * keys, Value * results, Index size);

	// template <typename Value, typename Index>
	// __global__
	// void reduce_impl(Value const * values, Index const * keys, Value * results, Index size) {

	// 	constexpr Index BlockDim		= 1024;
	// 	constexpr Index WarpDim			= 32;
	// 	constexpr Index WarpCount		= 32;

	// 	constexpr unsigned WarpMask		= 0xFFFFFFFFu;

	// 	Index const warp_idx			= threadIdx.x / WarpDim;
	// 	Index const thread_idx_in_warp	= threadIdx.x % WarpDim;

	// 	Index const block_head_offset	= blockIdx.x * BlockDim;
	// 	Index const block_head_seg		= keys[block_head_offset];

	// 	Index const block_last_offset	= block_head_offset + BlockDim - 1;
	// 	Index const block_last_seg		= block_last_offset < size ? keys[block_last_offset] : all_bits<Index>();

	// 	Index const warp_head_offset	= block_head_offset + warp_idx * WarpDim;
	// 	Index const warp_head_seg		= warp_head_offset < size ? keys[warp_head_offset] : all_bits<Index>();

	// 	Index const warp_last_offset	= warp_head_offset + WarpDim - 1;
	// 	Index const warp_last_seg		= warp_last_offset < size ? keys[warp_last_offset] : all_bits<Index>();

	// 	Index const this_offset			= block_head_offset + threadIdx.x;
	// 	Value this_value				= this_offset < size ? values[this_offset] : identity<Value>();

	// 	/* the segments to which this thread and the last one belong */

	// 	Index last_and_this_segs[2] = { all_bits<Index>(), all_bits<Index>() };
	// 	if (this_offset == 0) {
	// 		last_and_this_segs[1] = keys[this_offset];
	// 	} else if (this_offset < size) {
	// 		last_and_this_segs[0] = keys[this_offset - 1];
	// 		last_and_this_segs[1] = keys[this_offset];
	// 		// *(uint64_t *)(last_and_this_segs) = *(uint64_t const *)(keys + this_offset - 1);
	// 	}
	// 	Index const & last_seg = last_and_this_segs[0];
	// 	Index const & this_seg = last_and_this_segs[1];

	// 	bool const is_this_inbound	= this_seg != all_bits<Index>();
	// 	bool const is_atomic_needed	= is_this_inbound and (this_seg == block_head_seg or this_seg == block_last_seg);
	// 	bool const is_shared_needed	= is_this_inbound and (this_seg == warp_head_seg or this_seg == warp_last_seg);
	// 	bool const is_seg_head		= is_this_inbound and (block_head_offset == this_offset or last_seg != this_seg);

	// 	__shared__ Value warp_elem_exchange[WarpDim];
	// 	__shared__ Index warp_seg_exchange[WarpDim];

	// 	if (thread_idx_in_warp == 0) {
	// 		warp_elem_exchange[warp_idx] = this_value;
	// 		warp_seg_exchange[warp_idx] = this_seg;
	// 	}

	// 	/* reduction among threads inside a warp */

	// 	__shared__ Value thread_elem_exchange[BlockDim + WarpDim];

	// 	for (Index delta = 1; delta < WarpDim; delta *= 2) {
	// 		Index const neighbor_seg = __shfl_down_sync(WarpMask, this_seg, delta);

	// 		bool const is_partial = is_this_inbound and this_seg == neighbor_seg;
	// 		bool const any_partial = __any_sync(WarpMask, is_partial);
	// 		if (not any_partial) break;
			
	// 		if (is_partial) thread_elem_exchange[threadIdx.x] = this_value;
	// 		__syncwarp();
	// 		if (is_partial) this_value += thread_elem_exchange[threadIdx.x + delta];
	// 		__syncwarp();
	// 	}

	// 	if (is_seg_head and not is_shared_needed) {
	// 		results[this_seg] = this_value;
	// 	}

	// 	__syncthreads();

	// 	/* reduction among warps inside a block */

	// 	if (warp_idx + 1 < WarpDim) {

	// 		Value warp_elem = warp_elem_exchange[thread_idx_in_warp];
	// 		Index const this_warp_tail_seg = __shfl_sync(WarpMask, this_seg, WarpDim - 1);
	// 		Index const all_warps_head_seg = warp_seg_exchange[thread_idx_in_warp];
	// 		if (all_warps_head_seg != this_warp_tail_seg) all_warps_head_seg += 32;

	// 		for (Index delta = 1; delta < WarpDim; delta *= 2) {
	// 			Index const neighbor_seg = __shfl_down_sync(WarpMask, all_warps_head_seg, delta);

	// 			bool const is_partial = all_warps_head_seg == neighbor_seg;
	// 			bool const any_partial = __any_sync(WarpMask, is_partial);
	// 			if (not any_partial) break;
				
	// 			if (is_partial) thread_elem_exchange[threadIdx.x] = warp_elem;
	// 			__syncwarp();
	// 			if (is_partial) warp_elem += thread_elem_exchange[threadIdx.x + delta];
	// 			__syncwarp();
	// 		}

	// 		Index const next_warp_head_seg = __shfl_sync(WarpMask, all_warps_head_seg, warp_idx + 1);
	// 		if (this_warp_tail_seg == next_warp_head_seg) {
	// 			if (thread_idx_in_warp == warp_idx + 1) thread_elem_exchange[warp_idx * 32] = warp_elem;
	// 			__syncwarp();
	// 			if (is_seg_head) this_value += thread_elem_exchange[warp_idx * 32];
	// 			__syncwarp();
	// 		}
			
	// 	}

	// 	if (is_seg_head and not is_atomic_needed) {
	// 		results[this_seg] = this_value;
	// 	} else {
	// 		while (atomicCAS(locks + this_seg, 0, 1) == 0) continue;
	// 		results[this_seg] = this_value;
	// 		locks[this_seg] = 0;
	// 	}

	// }

}



#endif // SPP_REDUCE_HPP