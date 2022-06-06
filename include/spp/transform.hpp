#ifndef SPP_TRANSFORM_HPP
#define SPP_TRANSFORM_HPP

#include <cstdint>
#include <cooperative_groups.h>

#include "traits.hpp"
#include "pipelined_for.hpp"
#include "kernel_launch.hpp"



namespace spp {

	namespace cg = cooperative_groups;



	namespace global {
		
		template <uint32_t ThreadsPerBlock, uint32_t ItemsPerThread, typename InputIterator, typename OutputIterator, typename UnaryOp>
		__global__
		void transform(InputIterator data_in, OutputIterator data_out, uint32_t size, UnaryOp unary_op) {
			
			using ComputeType = std::decay_t<spp::dereference_t<InputIterator>>;
			using ResultType = std::decay_t<spp::apply_t<UnaryOp, ComputeType>>;

			auto const grid{ cg::this_grid() };
			auto const item_rank_begin{ uint32_t(grid.thread_rank()) };
			auto const item_rank_step{ uint32_t(grid.num_threads()) };

			ComputeType items[ItemsPerThread];

			for (uint32_t i_item = 0; i_item < ItemsPerThread; ++i_item) {
				uint32_t const item_rank = item_rank_begin + item_rank_step * i_item;
				if (item_rank < size) {
					items[i_item] = *(data_in + item_rank);
				}
			}

			for (uint32_t i_item = 0; i_item < ItemsPerThread; ++i_item) {
				uint32_t const item_rank = item_rank_begin + item_rank_step * i_item;
				if (item_rank < size) {
					if constexpr (std::is_same_v<ResultType, void>) {
						unary_op(items[i_item]);
					}
					else {
						*(data_out + item_rank) = unary_op(items[i_item]);
					}
				}
			}

		}

	}



	namespace kernel {

		template <typename InputIterator, typename OutputIterator, typename UnaryOp>
		cudaError_t transform(InputIterator data_in, OutputIterator data_out, uint32_t size, UnaryOp unary_op) {

			uint32_t constexpr ThreadsPerBlock = 128;
			uint32_t constexpr ItemsPerThread = 4;

			auto const fn{ reinterpret_cast<void const *>(global::transform<ThreadsPerBlock, ItemsPerThread, InputIterator, OutputIterator, UnaryOp>) };
			dim3 const grid_dim{ ceiled_div(size, ItemsPerThread * ThreadsPerBlock) };
			dim3 const block_dim{ ThreadsPerBlock };
			addresses_of args{ data_in, data_out, size, unary_op };

			return cudaLaunchKernel(fn, grid_dim, block_dim, args.get());

		}

	}

}



#endif // SPP_TRANSFORM_HPP