#ifndef SPP_REDUCE_HPP
#define SPP_REDUCE_HPP

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "types.hpp"
#include "math.hpp"
#include "operators.hpp"
#include "barrier.hpp"
#include "lookback.hpp"



namespace spp {

	namespace cg = ::cooperative_groups;



	namespace device {

		template <typename T, typename IdentityOp, u32 WarpsPerBlock, u32 ProducerBarrierId, u32 ConsumerBarrierId>
		__device__
		T scan_warp_exclusive_prefix(T const & value, cg::thread_block_tile<32> const & warp, IdentityOp identity_op) {

			__shared__ T warp_partial_results[WarpsPerBlock];

			if (warp.thread_rank() + 1 == 32) {
				warp_partial_results[warp.meta_group_rank()] = value;
			}

			barrier<ProducerBarrierId, WarpsPerBlock>		producer_barrier;
			barrier<ConsumerBarrierId, WarpsPerBlock - 1>	consumer_barrier;

			if (warp.meta_group_rank() == 0) {
				producer_barrier.arrive();

				return identity_op();
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



		template <typename T, typename IdentityOp, u32 WarpsPerBlock, u32 BarrierId>
		__device__
		T scan_block_exclusive_prefix(lookback<T> volatile * block_lookbacks,
			T const & warp_exclusive_prefix, T const & warp_reduction,
			cg::grid_group const & grid, cg::thread_block_tile<32> const & warp,
			IdentityOp identity_op) {

			T block_exclusive_prefix;
			__shared__ T shared_block_exclusive_prefix;

			barrier<BarrierId, WarpsPerBlock> barrier;

			if (warp.meta_group_rank() == 0) {
				block_exclusive_prefix = identity_op();
				
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



	namespace op {

		template <typename T>
		struct inclusive_scan {
			__device__
			T operator()(cg::thread_block_tile<32> const & warp, T const & value) const noexcept {
				return cg::inclusive_scan(warp, value);
			}
		};

		template <typename T>
		struct exclusive_scan {
			__device__
			T operator()(cg::thread_block_tile<32> const & warp, T const & value) const noexcept {
				return cg::exclusive_scan(warp, value);
			}
		};

	} // namespace op



	namespace global {

		template <typename ValueTy, typename ResultTy, typename ScanOp, typename IdentityOp, u32 ThreadsPerBlock, u32 ItemsPerThread, bool IsInclusive>
		__global__
		void generic_lookback_scan(ValueTy const * values, ResultTy * results, usize size, ScanOp scan_op, IdentityOp identity_op, lookback<ValueTy> volatile * block_lookbacks) {

			auto const grid						= cg::this_grid();
			auto const block					= cg::this_thread_block();
			auto const warp						= cg::tiled_partition<32>(block);

			usize constexpr WarpsPerBlock		= ThreadsPerBlock / 32;

			usize constexpr ItemsPerBlock		= ItemsPerThread * ThreadsPerBlock;
			usize constexpr ItemsPerWarp		= ItemsPerThread * 32;

			usize const block_rank_begin		= grid.block_rank() * ItemsPerBlock;
			usize const warp_rank_begin			= warp.meta_group_rank() * ItemsPerWarp;



			ValueTy item_prefixes[ItemsPerThread];
			ValueTy warp_reduction = identity_op();

			for (usize i_tile = 0; i_tile < ItemsPerThread; ++i_tile) {
				usize const item_rank = block_rank_begin + warp_rank_begin + 32 * i_tile + warp.thread_rank();
				
				ValueTy item = identity_op();
				if (item_rank < size) {
					Byte<sizeof(ValueTy)>::copy(&item, values + item_rank);
				}

				ValueTy const tile_prefix = scan_op(warp, item);
				item_prefixes[i_tile] = warp_reduction + tile_prefix;

				if constexpr (IsInclusive) {
					warp_reduction += warp.shfl(tile_prefix, 32 - 1);
				}
				else {
					warp_reduction += warp.shfl(tile_prefix + item, 32 - 1);
				}
			}



			u32 constexpr WarpPrefixProduced	= 1;
			u32 constexpr WarpPrefixConsumed	= 2;
			u32 constexpr BlockPrefixProduced	= 3;
			
			ValueTy const warp_exclusive_prefix = device::scan_warp_exclusive_prefix<ValueTy, IdentityOp, WarpsPerBlock, WarpPrefixProduced, WarpPrefixConsumed>(warp_reduction, warp, identity_op);

			ValueTy const block_exclusive_prefix = device::scan_block_exclusive_prefix<ValueTy, IdentityOp, WarpsPerBlock, BlockPrefixProduced>(block_lookbacks, warp_exclusive_prefix, warp_reduction, grid, warp, identity_op);



			for (usize i_tile = 0; i_tile < ItemsPerThread; ++i_tile) {
				u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_tile + warp.thread_rank();
				
				if (item_rank < size) {
					ResultTy result = block_exclusive_prefix + warp_exclusive_prefix + item_prefixes[i_tile];
					Byte<sizeof(ResultTy)>::copy(results + item_rank, &result);
				}
			}

		}



		template <typename T>
		__global__
		void init_inclusive_scan(lookback<T> volatile * block_lookbacks, usize num_blocks) {
			auto const grid = cg::this_grid();

			for (usize i = grid.thread_rank(); i < num_blocks; i += grid.num_threads()) {
				block_lookbacks[i] = lookback<T>::zero();
			}
		}

	} // namespace global



	namespace kernel {

		template <typename ValueTy, typename ResultTy>
		cudaError_t inclusive_scan(void * temp_storage, usize & temp_storage_bytes, ValueTy const * values, ResultTy * results, usize size) {

			usize constexpr ThreadsPerBlock = 128;
			usize constexpr ItemsPerThread = 16;
			
			usize constexpr ItemsPerBlock = ItemsPerThread * ThreadsPerBlock;

			usize num_blocks = ceiled_div(size, ItemsPerBlock);

			if (temp_storage == nullptr) {
				temp_storage_bytes = num_blocks * sizeof(lookback<ValueTy>);
				return cudaSuccess;
			}
			else {
				/* init */ {
					auto fn			= reinterpret_cast<void const *>(global::init_inclusive_scan<ValueTy>);
					auto block_dim	= dim3(128, 1, 1);
					auto grid_dim	= dim3(ceiled_div(num_blocks, block_dim.x), 1, 1);
					void * args[]	= { &temp_storage, &num_blocks };
					
					if (auto e = cudaLaunchKernel(fn, grid_dim, block_dim, args); cudaSuccess != e) {
						return e;
					}
				}

				/* scan */ {
					using ScanOp		= op::inclusive_scan<ValueTy>;
					using IdentityOp	= op::identity<ValueTy>;

					auto fn				= reinterpret_cast<void const *>(global::generic_lookback_scan<ValueTy, ResultTy, ScanOp, IdentityOp, ThreadsPerBlock, ItemsPerThread, true>);
					auto grid_dim		= dim3(num_blocks, 1, 1);
					auto block_dim		= dim3(ThreadsPerBlock, 1, 1);

					auto scan_op		= ScanOp();
					auto identity_op	= IdentityOp();
					void * args[] = { &values, &results, &size, &scan_op, &identity_op, &temp_storage };

					return cudaLaunchKernel(fn, grid_dim, block_dim, args);
				}
			}

		} // inclusive_scan

	} // namespace kernel

} // namespace spp



#endif // SPP_REDUCE_HPP