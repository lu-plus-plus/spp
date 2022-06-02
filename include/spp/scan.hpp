#ifndef SPP_SCAN_HPP
#define SPP_SCAN_HPP

#include <type_traits>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "types.hpp"
#include "operators.hpp"
#include "barrier.hpp"
#include "lookback.hpp"
#include "traits.hpp"



namespace spp {

	namespace cg = ::cooperative_groups;



	namespace device {

		template <u32 WarpsPerBlock, u32 ProducerBarrierId, u32 ConsumerBarrierId,
			typename ComputeType, typename Binary, typename Identity>
		__device__
		ComputeType scan_warp_exclusive_prefix(cg::thread_block_tile<32> const & warp,
			ComputeType const & value, Binary && binary, Identity && identity) {

			__shared__ ComputeType warp_partial_results[WarpsPerBlock];

			if (warp.thread_rank() + 1 == 32) {
				warp_partial_results[warp.meta_group_rank()] = value;
			}

			barrier<ProducerBarrierId, WarpsPerBlock>		producer_barrier;
			barrier<ConsumerBarrierId, WarpsPerBlock - 1>	consumer_barrier;

			if (warp.meta_group_rank() == 0) {
				producer_barrier.arrive();

				return std::forward<Identity>(identity)();
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
					ComputeType const warp_partial_result = warp_partial_results[warp.thread_rank()];
					warp_partial_results[warp.thread_rank()] = cg::exclusive_scan(active, warp_partial_result, std::forward<Binary>(binary));
				}

				consumer_barrier.arrive();

				return warp_partial_results[warp.meta_group_rank()];
			}

		}



		template <u32 WarpsPerBlock, u32 BarrierId,
			typename ComputeType, typename Binary, typename Identity>
		__device__
		ComputeType scan_block_exclusive_prefix(cg::grid_group const & grid, cg::thread_block_tile<32> const & warp,
			lookback<ComputeType> volatile * block_lookbacks,
			ComputeType const & warp_exclusive_prefix, ComputeType const & warp_reduction,
			Binary && binary, Identity && identity) {

			ComputeType block_exclusive_prefix;
			__shared__ ComputeType shared_block_exclusive_prefix;

			barrier<BarrierId, WarpsPerBlock> barrier;

			if (warp.meta_group_rank() == 0) {				
				if (warp.thread_rank() == 0) {
					block_exclusive_prefix = std::forward<Identity>(identity)();

					for (isize i_block = isize(grid.block_rank()) - 1_is; 0_is <= i_block; --i_block) {
						bool is_prefixed;
						ComputeType const value{ block_lookbacks[i_block].spin_and_load(is_prefixed) };

						block_exclusive_prefix = std::forward<Binary>(binary)(block_exclusive_prefix, value);

						if (is_prefixed) break;
					}

					shared_block_exclusive_prefix = block_exclusive_prefix;
				}

				barrier.arrive();

				block_exclusive_prefix = warp.shfl(block_exclusive_prefix, 0);
			}
			else if (warp.meta_group_rank() + 1 != WarpsPerBlock) {
				barrier.wait();

				block_exclusive_prefix = shared_block_exclusive_prefix;
			}
			else {
				if (warp.thread_rank() + 1 == 32) {
					block_lookbacks[grid.block_rank()].store_aggregate(warp_exclusive_prefix + warp_reduction);
					// block_lookbacks[grid.block_rank()] = lookback<ComputeType>::make_aggregate(warp_exclusive_prefix + warp_reduction);
				}

				barrier.wait();
				
				block_exclusive_prefix = shared_block_exclusive_prefix;

				if (warp.thread_rank() + 1 == 32) {
					block_lookbacks[grid.block_rank()].store_prefix(block_exclusive_prefix + warp_exclusive_prefix + warp_reduction);
					// block_lookbacks[grid.block_rank()] = lookback<ComputeType>::make_prefix(block_exclusive_prefix + warp_exclusive_prefix + warp_reduction);
				}
			}

			return block_exclusive_prefix;

		}



		template <typename T, typename Binary>
		__device__
		std::decay_t<T> warp_inclusive_scan(cg::thread_block_tile<32> const & warp, T && value, Binary && binary) {
			return cg::inclusive_scan(warp, std::forward<T>(value), std::forward<Binary>(binary));
		}

		template <typename T, typename Binary>
		__device__
		std::decay_t<T> warp_exclusive_scan(cg::thread_block_tile<32> const & warp, T && value, Binary && binary) {
			return cg::exclusive_scan(warp, std::forward<T>(value), std::forward<Binary>(binary));
		}

	} // namespace device



	namespace global {

		template <
			typename InputIterator, typename OutputIterator,
			typename Prologue, typename Epilogue,
			typename Binary, typename Identity,
			usize ThreadsPerBlock, usize ItemsPerThread, bool IsInclusive
		>
		__global__
		void generic_lookback_scan(
			InputIterator data_in, OutputIterator data_out, usize size,
			Prologue prologue, Epilogue epilogue,
			Binary binary, Identity identity,
			lookback<std::decay_t< apply_t<Prologue, dereference_t<InputIterator>, usize> >> volatile * block_lookbacks
		) {

			using InputType		= std::decay_t< dereference_t<InputIterator> >;
			using OutputType	= std::decay_t< dereference_t<OutputIterator> >;
			using ComputeType	= std::decay_t< apply_t<Prologue, dereference_t<InputIterator>, usize> >;

			auto const grid		= cg::this_grid();
			auto const block	= cg::this_thread_block();
			auto const warp		= cg::tiled_partition<32>(block);

			usize constexpr WarpsPerBlock		= ThreadsPerBlock / 32;

			usize constexpr ItemsPerBlock		= ItemsPerThread * ThreadsPerBlock;
			usize constexpr ItemsPerWarp		= ItemsPerThread * 32;

			usize const block_rank_begin		= grid.block_rank() * ItemsPerBlock;
			usize const warp_rank_begin			= warp.meta_group_rank() * ItemsPerWarp;



			ComputeType item_prefixes[ItemsPerThread];
			ComputeType warp_reduction = identity();

			for (usize i_tile = 0; i_tile < ItemsPerThread; ++i_tile) {
				usize const item_rank = block_rank_begin + warp_rank_begin + 32 * i_tile + warp.thread_rank();

				ComputeType item = identity();
				if (item_rank < size) {
					InputType input;
					bytes_of<InputType>::copy(&input, &(*(data_in + item_rank)));
					item = prologue(input, item_rank);
				}

				if constexpr (IsInclusive) {
					ComputeType const tile_prefix = device::warp_inclusive_scan(warp, item, binary);
					item_prefixes[i_tile] = warp_reduction + tile_prefix;					
					warp_reduction += warp.shfl(tile_prefix, 32 - 1);
				}
				else {
					ComputeType const tile_prefix = device::warp_exclusive_scan(warp, item, binary);
					item_prefixes[i_tile] = warp_reduction + tile_prefix;
					warp_reduction += warp.shfl(tile_prefix + item, 32 - 1);
				}
			}



			u32 constexpr WarpPrefixProduced	= 1;
			u32 constexpr WarpPrefixConsumed	= 2;
			u32 constexpr BlockPrefixProduced	= 3;
			
			ComputeType const warp_exclusive_prefix = device::scan_warp_exclusive_prefix<WarpsPerBlock, WarpPrefixProduced, WarpPrefixConsumed>(warp, warp_reduction, binary, identity);

			ComputeType const block_exclusive_prefix = device::scan_block_exclusive_prefix<WarpsPerBlock, BlockPrefixProduced>(grid, warp, block_lookbacks, warp_exclusive_prefix, warp_reduction, binary, identity);



			ComputeType const block_warp_exclusive_prefix = binary(block_exclusive_prefix, warp_exclusive_prefix);

			for (usize i_tile = 0; i_tile < ItemsPerThread; ++i_tile) {
				u32 const item_rank = block_rank_begin + warp_rank_begin + 32 * i_tile + warp.thread_rank();
				
				if (item_rank < size) {
					OutputType item = epilogue(binary(block_warp_exclusive_prefix, item_prefixes[i_tile]), item_rank);
					bytes_of<OutputType>::copy(&(*(data_out + item_rank)), &item);
				}
			}

		}



		template <typename T>
		__global__
		void init_inclusive_scan(lookback<T> volatile * block_lookbacks, usize num_blocks) {
			auto const grid = cg::this_grid();

			for (usize i = grid.thread_rank(); i < num_blocks; i += grid.num_threads()) {
				block_lookbacks[i].store_invalid();
				// block_lookbacks[i] = lookback<T>::make_invalid();
			}
		}

	} // namespace global



	namespace kernel {

		template <bool IsInclusive,
			typename InputIterator, typename OutputIterator,
			typename Prologue, typename Epilogue,
			typename Binary, typename Identity>
		cudaError_t generic_scan(void * temp_storage, usize & temp_storage_bytes,
			InputIterator data_in, OutputIterator data_out, usize size,
			Prologue prologue, Epilogue epilogue,
			Binary binary, Identity identity) {

			using InputType		= std::decay_t< dereference_t<InputIterator> >;
			using OutputType	= std::decay_t< dereference_t<OutputIterator> >;
			using ComputeType	= std::decay_t< apply_t<Prologue, dereference_t<InputIterator>, usize> >;

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
						Prologue, Epilogue,
						Binary, Identity,
						ThreadsPerBlock, ItemsPerThread, IsInclusive
					>);
					auto grid_dim		= dim3(scan_num_blocks, 1, 1);
					auto block_dim		= dim3(ThreadsPerBlock, 1, 1);

					void * args[] = { &data_in, &data_out, &size, &prologue, &epilogue, &binary, &identity, &temp_storage };

					return cudaLaunchKernel(fn, grid_dim, block_dim, args);
				}
			}

		} // generic_scan

		template <typename InputIterator, typename OutputIterator,
			typename Prologue = op::identity_function<>,
			typename Epilogue = op::identity_function<>,
			typename Binary = op::plus<>,
			typename Identity = op::identity_element<std::decay_t< apply_t<Prologue, dereference_t<InputIterator>, usize> >> >
		cudaError_t inclusive_scan(void * temp_storage, usize & temp_storage_bytes,
			InputIterator data_in, OutputIterator data_out, usize size,
			Prologue prologue = Prologue(),
			Epilogue epilogue = Epilogue(),
			Binary binary = Binary(),
			Identity identity = Identity()) {
			
			return generic_scan<true>(
				temp_storage, temp_storage_bytes,
				data_in, data_out, size,
				prologue, epilogue,
				binary, identity
			);
		}

		template <typename InputIterator, typename OutputIterator,
			typename Prologue = op::identity_function<>,
			typename Epilogue = op::identity_function<>,
			typename Binary = op::plus<>,
			typename Identity = op::identity_element<std::decay_t< apply_t<Prologue, dereference_t<InputIterator>, usize> >> >
		cudaError_t exclusive_scan(void * temp_storage, usize & temp_storage_bytes,
			InputIterator data_in, OutputIterator data_out, usize size,
			Prologue prologue = Prologue(),
			Epilogue epilogue = Epilogue(),
			Binary binary = Binary(),
			Identity identity = Identity()) {
			
			return generic_scan<false>(
				temp_storage, temp_storage_bytes,
				data_in, data_out, size,
				prologue, epilogue,
				binary, identity
			);
		}

	} // namespace kernel

} // namespace spp



#endif // SPP_SCAN_HPP