#ifndef SPP_REDUCE_HPP
#define SPP_REDUCE_HPP

#include <cstdint>
#include <type_traits>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "traits.hpp"
#include "operators.hpp"
#include "pipelined_for.hpp"
#include "kernel_launch.hpp"



namespace spp {

	namespace cg = cooperative_groups;



	namespace global {

		template <uint32_t ThreadsPerBlock, uint32_t ItemsPerThread,
			typename InputIterator, typename OutputIterator,
			typename Binary, typename Identity, typename Epilogue>
		__global__
		void reduce(InputIterator data_in, OutputIterator data_out, uint32_t size,
			Binary binary, Identity identity, Epilogue epilogue,
			std::decay_t<dereference_t<OutputIterator>> * block_partial_results) {

			using ComputeType = std::decay_t<dereference_t<InputIterator>>;
			using ResultType = std::decay_t<dereference_t<OutputIterator>>;

			auto const grid{ cg::this_grid() };
			auto const block{ cg::this_thread_block() };
			auto const warp{ cg::tiled_partition<32>(block) };

			uint32_t constexpr WarpsPerBlock{ ThreadsPerBlock / 32 };
			uint32_t constexpr ItemsPerBlock{ ItemsPerThread * ThreadsPerBlock };
			uint32_t constexpr ItemsPerWarp{ ItemsPerThread * 32 };

			uint32_t const tile_rank_begin{ uint32_t(grid.block_rank()) * ItemsPerBlock + warp.meta_group_rank() * ItemsPerWarp };
			uint32_t const tile_rank_step{ uint32_t(grid.num_blocks()) * ItemsPerBlock };

			ComputeType thread_partial_result{ identity() };

			for (uint32_t warp_rank_begin = tile_rank_begin; warp_rank_begin < size; warp_rank_begin += tile_rank_step) {
				uint32_t const thread_rank_begin{ warp_rank_begin + warp.thread_rank() };
				
				ComputeType thread_items[ItemsPerThread];

				device::pipelined_for<ItemsPerThread, 2>([&] (uint32_t i_item) {
					uint32_t const item_rank = thread_rank_begin + 32 * i_item;
					thread_items[i_item] = item_rank < size ? *(data_in + item_rank) : identity();
				}, [&] (uint32_t i_item) {
					thread_partial_result = binary(thread_partial_result, thread_items[i_item]);
				});
			}

			warp.sync();

			__shared__ ResultType warp_partial_results[WarpsPerBlock];

			ComputeType const warp_partial_result = cg::reduce(warp, thread_partial_result, binary);
			if (warp.thread_rank() == 0) {
				warp_partial_results[warp.meta_group_rank()] = warp_partial_result;
			}

			block.sync();

			if (warp.meta_group_rank() == 0) {
				if (warp.thread_rank() < WarpsPerBlock) {
					ComputeType const block_partial_result = cg::reduce(cg::coalesced_threads(), warp_partial_results[warp.thread_rank()], binary);
					if (warp.thread_rank() == 0) {
						block_partial_results[grid.block_rank()] = block_partial_result;
					}
				}			
			}

			grid.sync();

			if (grid.block_rank() == 0 and warp.meta_group_rank() == 0) {
				ComputeType partial_result = identity();
				for (uint32_t block_rank = warp.thread_rank(); block_rank < grid.num_blocks(); block_rank += 32) {
					partial_result = binary(partial_result, block_partial_results[block_rank]);
				}
				
				ComputeType const final_partial_result = cg::reduce(warp, partial_result, binary);

				if (warp.thread_rank() == 0) {
					*data_out = epilogue(final_partial_result);
				}
			}

		}

	}



	namespace kernel {

		template <typename InputIterator, typename OutputIterator,
			typename Binary = op::plus<>,
			typename Identity = op::identity_element<std::decay_t<dereference_t<InputIterator>>>,
			typename Epilogue = op::identity_function<>>
		cudaError_t reduce(void * d_temp_storage, uint32_t & d_temp_storage_bytes,
			InputIterator data_in, OutputIterator data_out, uint32_t size,
			Binary binary = Binary(), Identity identity = Identity(), Epilogue epilogue = Epilogue()) {

			using ResultType = std::decay_t<dereference_t<OutputIterator>>;

			uint32_t constexpr ThreadsPerBlock{ 256 };
			uint32_t constexpr ItemsPerThread{ 16 };

			auto fn{ reinterpret_cast<void const *>(global::reduce<ThreadsPerBlock, ItemsPerThread, InputIterator, OutputIterator, Binary, Identity, Epilogue>) };

			uint32_t const max_active_blocks{ max_active_blocks_for(fn, ThreadsPerBlock) };
			uint32_t const d_required_bytes{ ceiled_div(sizeof(ResultType) * max_active_blocks, 128) };

			if (d_temp_storage == nullptr) {
				d_temp_storage_bytes = d_required_bytes;
				
				return cudaSuccess;
			}
			else {
				dim3 const grid_dim{ max_active_blocks };
				dim3 const block_dim{ ThreadsPerBlock };
				addresses_of args{ data_in, data_out, size, binary, identity, epilogue, d_temp_storage };

				return cudaLaunchCooperativeKernel(fn, grid_dim, block_dim, args.get());
			}

		}

	}

}



#endif // SPP_REDUCE_HPP