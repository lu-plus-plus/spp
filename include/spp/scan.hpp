#ifndef SPP_REDUCE_HPP
#define SPP_REDUCE_HPP

#include <type_traits>
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



		template <typename ValueTy, typename ScanOp>
		__device__
		std::decay_t<ValueTy> warp_inclusive_scan(cg::thread_block_tile<32> const & warp, ValueTy && value, ScanOp && scan_op) {
			return cg::inclusive_scan(warp, std::forward<ValueTy>(value), std::forward<ScanOp>(scan_op));
		}

		template <typename ValueTy, typename ScanOp>
		__device__
		std::decay_t<ValueTy> warp_exclusive_scan(cg::thread_block_tile<32> const & warp, ValueTy && value, ScanOp && scan_op) {
			return cg::exclusive_scan(warp, std::forward<ValueTy>(value), std::forward<ScanOp>(scan_op));
		}

	} // namespace device



	namespace global {

		template <
			typename InputIterator, typename OutputIterator,
			typename Prologue, typename Epilogue,
			typename ScanOp, typename IdentityOp,
			usize ThreadsPerBlock, usize ItemsPerThread, bool IsInclusive
		>
		__global__
		void generic_lookback_scan(
			InputIterator data_in, OutputIterator data_out, usize size,
			Prologue prologue, Epilogue epilogue,
			ScanOp scan_op, IdentityOp identity_op,
			lookback<decltype(prologue(*data_in))> volatile * block_lookbacks
		) {

			using InputType = std::decay_t<decltype(*data_in)>;
			using OutputType = std::decay_t<decltype(*data_out)>;
			using ComputeType = std::decay_t<decltype(prologue(*data_in))>;

			auto const grid						= cg::this_grid();
			auto const block					= cg::this_thread_block();
			auto const warp						= cg::tiled_partition<32>(block);

			usize constexpr WarpsPerBlock		= ThreadsPerBlock / 32;

			usize constexpr ItemsPerBlock		= ItemsPerThread * ThreadsPerBlock;
			usize constexpr ItemsPerWarp		= ItemsPerThread * 32;

			usize const block_rank_begin		= grid.block_rank() * ItemsPerBlock;
			usize const warp_rank_begin			= warp.meta_group_rank() * ItemsPerWarp;



			ComputeType item_prefixes[ItemsPerThread];
			ComputeType warp_reduction = identity_op();

			for (usize i_tile = 0; i_tile < ItemsPerThread; ++i_tile) {
				usize const item_rank = block_rank_begin + warp_rank_begin + 32 * i_tile + warp.thread_rank();

				ComputeType item = identity_op();
				if (item_rank < size) {
					// if constexpr (std::is_same_v<InputType, ComputeType>) {
					// 	Byte<sizeof(ValueTy)>::copy(&item, &data_in[item_rank]);
					// 	item = prologue(item);
					// }
					InputType input;
					Byte<sizeof(InputType)>::copy(&input, &data_in[item_rank]);
					item = prologue(input);
				}

				if constexpr (IsInclusive) {
					ComputeType const tile_prefix = device::warp_inclusive_scan(warp, item, scan_op);
					item_prefixes[i_tile] = warp_reduction + tile_prefix;					
					warp_reduction += warp.shfl(tile_prefix, 32 - 1);
				}
				else {
					ComputeType const tile_prefix = device::warp_exclusive_scan(warp, item, scan_op);
					item_prefixes[i_tile] = warp_reduction + tile_prefix;
					warp_reduction += warp.shfl(tile_prefix + item, 32 - 1);
				}
			}



			u32 constexpr WarpPrefixProduced	= 1;
			u32 constexpr WarpPrefixConsumed	= 2;
			u32 constexpr BlockPrefixProduced	= 3;
			
			ComputeType const warp_exclusive_prefix = device::scan_warp_exclusive_prefix<ComputeType, IdentityOp, WarpsPerBlock, WarpPrefixProduced, WarpPrefixConsumed>(warp_reduction, warp, identity_op);

			ComputeType const block_exclusive_prefix = device::scan_block_exclusive_prefix<ComputeType, IdentityOp, WarpsPerBlock, BlockPrefixProduced>(block_lookbacks, warp_exclusive_prefix, warp_reduction, grid, warp, identity_op);



			for (usize i_tile = 0; i_tile < ItemsPerThread; ++i_tile) {
				u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_tile + warp.thread_rank();
				
				if (item_rank < size) {
					OutputType item = epilogue(block_exclusive_prefix + warp_exclusive_prefix + item_prefixes[i_tile]);
					Byte<sizeof(OutputType)>::copy(&data_out[item_rank], &item);
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

		template <
			typename InputIterator, typename OutputIterator,
			typename PrologueType = op::identity_function<>,
			typename EpilogueType = op::identity_function<>,
			typename ScanType = op::plus<>,
			typename IdentityElementType = op::identity_element<decltype(std::declval<PrologueType>()(*std::declval<InputIterator>()))>
		>
		cudaError_t inclusive_scan(
			void * temp_storage, usize & temp_storage_bytes,
			InputIterator data_in, OutputIterator data_out, usize size,
			PrologueType prologue = PrologueType(),
			EpilogueType epilogue = EpilogueType(),
			ScanType scan = ScanType(),
			IdentityElementType identity_element = IdentityElementType()
		) {

			using InputType = std::decay_t<decltype(*data_in)>;
			using OutputType = std::decay_t<decltype(*data_out)>;
			using ComputeType = std::decay_t<decltype(prologue(*data_in))>;

			usize constexpr ThreadsPerBlock = 128;
			usize constexpr ItemsPerThread = 16;
			usize constexpr ItemsPerBlock = ItemsPerThread * ThreadsPerBlock;

			usize scan_num_blocks = ceiled_div(size, ItemsPerBlock);

			if (temp_storage == nullptr) {
				temp_storage_bytes = scan_num_blocks * sizeof(lookback<ComputeType>);
				return cudaSuccess;
			}
			else {
				/* init */ {
					auto fn			= reinterpret_cast<void const *>(global::init_inclusive_scan<ComputeType>);
					auto block_dim	= dim3(128, 1, 1);
					auto grid_dim	= dim3(ceiled_div(scan_num_blocks, block_dim.x), 1, 1);
					void * args[]	= { &temp_storage, &scan_num_blocks };
					
					if (auto e = cudaLaunchKernel(fn, grid_dim, block_dim, args); cudaSuccess != e) {
						return e;
					}
				}

				/* scan */ {
					auto fn				= reinterpret_cast<void const *>(global::generic_lookback_scan<
						InputIterator, OutputIterator,
						PrologueType, EpilogueType,
						ScanType, IdentityElementType,
						ThreadsPerBlock, ItemsPerThread, true
					>);
					auto grid_dim		= dim3(scan_num_blocks, 1, 1);
					auto block_dim		= dim3(ThreadsPerBlock, 1, 1);

					void * args[] = { &data_in, &data_out, &size, &prologue, &epilogue, &scan, &identity_element, &temp_storage };

					return cudaLaunchKernel(fn, grid_dim, block_dim, args);
				}
			}

		} // inclusive_scan

	} // namespace kernel

} // namespace spp



#endif // SPP_REDUCE_HPP