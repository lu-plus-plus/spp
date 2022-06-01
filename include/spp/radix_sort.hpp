#ifndef SPP_RADIX_SORT_HPP
#define SPP_RADIX_SORT_HPP

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>

#include "types.hpp"
#include "lookback.hpp"
#include "kernel_launch.hpp"
#include "memory_sections.hpp"

#include "iterator.hpp"
#include "copy.hpp"
#include "histogram.hpp"
#include "transform.hpp"



namespace spp {

	namespace cg = cooperative_groups;



	template <typename T, usize S>
	struct vector {

		static_assert(std::is_trivial_v<T>, "T must be a trivial type.");

		// properties

		using element_type = T;

		static constexpr
		__host__ __device__
		usize size() {
			return S;
		}

		// members and ctor / dtor

		T data[S];

		vector() = default;

		// access

		__host__ __device__
		T & operator[](usize idx) noexcept {
			return data[idx];
		}

		__host__ __device__
		T const & operator[](usize idx) const noexcept {
			return data[idx];
		}

		__host__ __device__
		T volatile & operator[](usize idx) volatile noexcept {
			return data[idx];
		}

		__host__ __device__
		T const volatile & operator[](usize idx) const volatile noexcept {
			return data[idx];
		}

		__host__ __device__
		T * addr_at(usize idx) noexcept {
			return data + idx;
		}

		__host__ __device__
		T const * addr_at(usize idx) const noexcept {
			return data + idx;
		}

		__host__ __device__
		T volatile * addr_at(usize idx) volatile noexcept {
			return data + idx;
		}

		__host__ __device__
		T const volatile * addr_at(usize idx) const volatile noexcept {
			return data + idx;
		}

	};



	inline
	__device__
	u32 thread_mask_lt(cg::thread_block_tile<32> const & warp) {
		return (1 << warp.thread_rank()) - 1;
	}



	template <usize ThreadsPerBlock, template <typename> typename Histogram>
	__global__
	void initialize_histograms(Histogram<lookback<u32>> * block_histograms, Histogram<u32> * p_grid_histogram) {
		static_assert(ThreadsPerBlock == (*p_grid_histogram).size(), "invalid block dimension");
		
		usize constexpr WarpsPerBlock{ ThreadsPerBlock / 32 };

		auto const grid		{ cg::this_grid() };
		auto const block	{ cg::this_thread_block() };
		auto const warp		{ cg::tiled_partition<32>(block) };
				
		block_histograms[grid.block_rank()][block.thread_rank()].store_invalid();

		if (grid.block_rank() == 0) {
			u32 const radix_partial_result{ (*p_grid_histogram)[block.thread_rank()] };
			u32 const radix_exclusive_prefix{ cg::exclusive_scan(warp, radix_partial_result) };

			__shared__ u32 shared_grid_histogram[WarpsPerBlock];
			if (warp.thread_rank() + 1 == 32) {
				shared_grid_histogram[warp.meta_group_rank()] = radix_exclusive_prefix + radix_partial_result;
			}
			block.sync();

			if (warp.meta_group_rank() == 0 and warp.thread_rank() < ThreadsPerBlock / 32) {
				u32 const warp_partial_result{ shared_grid_histogram[warp.thread_rank()] };
				u32 const warp_exclusive_prefix{ cg::exclusive_scan(cg::coalesced_threads(), warp_partial_result) };
				shared_grid_histogram[warp.thread_rank()] = warp_exclusive_prefix;
			}
			block.sync();

			u32 const warp_exclusive_prefix{ shared_grid_histogram[warp.meta_group_rank()] };
			(*p_grid_histogram)[block.thread_rank()] = warp_exclusive_prefix + radix_exclusive_prefix;
		}
	}



	template <bool WithIndices, usize ThreadsPerBlock, usize ItemsPerThread,
		template <typename> typename Histogram, typename Binning, typename MatchAny>
	__global__
	void generic_radix_sort_scan_lane(u32 const * keys_in, u32 * keys_out, usize size,
		u32 const * indices_in, u32 * indices_out,
		Histogram<lookback<u32>> volatile * block_radix_histograms,
		Histogram<u32> const * p_grid_radix_histogram,
		Binning binning, MatchAny match_any) {
		
		usize constexpr WarpsPerBlock	= ThreadsPerBlock / 32;
		usize constexpr ItemsPerWarp	= ItemsPerThread * 32;
		usize constexpr ItemsPerBlock	= ItemsPerThread * ThreadsPerBlock;

		usize constexpr HistogramSize	= Histogram<u32>::size();

		auto const grid					= cg::this_grid();
		auto const block				= cg::this_thread_block();
		auto const warp					= cg::tiled_partition<32>(block);

		usize const block_rank_begin	= grid.block_rank() * ItemsPerBlock;
		usize const warp_rank_begin		= warp.meta_group_rank() * ItemsPerWarp;



		u32 thread_keys[ItemsPerThread];
		u16 thread_exclusive_prefixes[ItemsPerThread];

		__shared__ Histogram<u16> shared_warp_exclusive_prefixes[WarpsPerBlock];
		__shared__ Histogram<u32> shared_block_exclusive_prefixes;
		__shared__ Histogram<u32> shared_grid_exclusive_prefix;

		Histogram<u16> & this_warp_exclusive_prefix = shared_warp_exclusive_prefixes[warp.meta_group_rank()];

		if (block.thread_rank() < HistogramSize) {
			usize const radix_bank = block.thread_rank();
			for (usize i_warp = 0; i_warp < WarpsPerBlock; ++i_warp) {
				shared_warp_exclusive_prefixes[i_warp][radix_bank] = 0_u16;
			}
			shared_block_exclusive_prefixes[radix_bank] = 0_u32;
			shared_grid_exclusive_prefix[radix_bank] = (*p_grid_radix_histogram)[radix_bank];
		}

		block.sync();



		auto load_key = [&] (usize i_key) {
			usize const in_rank = block_rank_begin + warp_rank_begin + 32 * i_key + warp.thread_rank();
			bool const is_active = in_rank < size;
			if (is_active) {
				thread_keys[i_key] = keys_in[in_rank];
				thread_exclusive_prefixes[i_key] = 0;
			}
			else {
				thread_keys[i_key] = ~ 0_u32;
				thread_exclusive_prefixes[i_key] = ~ 0_u16;
			}
		};

		auto scan_thread_prefix = [&] (usize i_key) {
			bool const is_active = thread_exclusive_prefixes[i_key] != ~ 0_u32;

			usize const radix_bank = binning(thread_keys[i_key]);

			u32 const match_mask = match_any(warp, is_active, radix_bank);
			u32 const match_count = __popc(match_mask);
			u32 const match_exclusive_prefix = __popc(match_mask & thread_mask_lt(warp));
			
			thread_exclusive_prefixes[i_key] = this_warp_exclusive_prefix[radix_bank] + u16(match_exclusive_prefix);
			warp.sync();

			if (match_exclusive_prefix + 1 == match_count) {
				this_warp_exclusive_prefix[radix_bank] += match_count;
			}
			warp.sync();
		};

		usize constexpr ItemsPerLoading = 2;
		usize constexpr LoadingsPerThread = ItemsPerThread / ItemsPerLoading;

		for (usize i_key = 0; i_key < ItemsPerLoading; ++i_key) {
			load_key(i_key);
		}

		for (usize i_loaded = 0; i_loaded < LoadingsPerThread; ++i_loaded) {
			usize const ii_loading = i_loaded + 1;

			if (ii_loading != LoadingsPerThread) {
				for (usize j_key = 0; j_key < ItemsPerLoading; ++j_key) {
					usize const k_load = ii_loading * ItemsPerLoading + j_key;
					load_key(k_load);
				}
			}

			for (usize j_key = 0; j_key < ItemsPerLoading; ++j_key) {
				usize const k_load = i_loaded * ItemsPerLoading + j_key;
				scan_thread_prefix(k_load);
			}
		}

		block.sync();



		if (block.thread_rank() < HistogramSize) {
			usize const radix_bank = block.thread_rank();

			auto block_reduction = 0_u32;

			for (usize i_warp = 0; i_warp < WarpsPerBlock; ++i_warp) {
				u16 const warp_reduction = shared_warp_exclusive_prefixes[i_warp][radix_bank];
				shared_warp_exclusive_prefixes[i_warp][radix_bank] = block_reduction;
				block_reduction += warp_reduction;
			}

			block_radix_histograms[grid.block_rank()][radix_bank].store_aggregate(block_reduction);

			auto block_exclusive_prefix = 0_u32;

			for (isize i_block = isize(grid.block_rank()) - 1_is; 0 <= i_block; --i_block) {
				bool is_prefixed;
				u32 const block_lookback = block_radix_histograms[i_block][radix_bank].spin_and_load(is_prefixed);

				block_exclusive_prefix += block_lookback;
				if (is_prefixed) break;
			}

			shared_block_exclusive_prefixes[radix_bank] = block_exclusive_prefix;

			block_radix_histograms[grid.block_rank()][radix_bank].store_prefix(block_exclusive_prefix + block_reduction);
		}

		block.sync();



		for (usize i_key = 0; i_key < ItemsPerThread; ++i_key) {
			usize const in_rank = block_rank_begin + warp_rank_begin + 32 * i_key + warp.thread_rank();

			if (in_rank < size) {
				usize const radix_bank = binning(thread_keys[i_key]);

				usize const out_rank = shared_grid_exclusive_prefix[radix_bank]
					+ shared_block_exclusive_prefixes[radix_bank]
					+ this_warp_exclusive_prefix[radix_bank]
					+ usize(thread_exclusive_prefixes[i_key]);
					
				keys_out[out_rank] = thread_keys[i_key];

				if constexpr (WithIndices) {
					indices_out[out_rank] = indices_in[in_rank];
				}
			}
		}

	}



	namespace details {

		inline constexpr
		usize radix_lane_bits = 8_us;

		inline constexpr
		usize radix_lane_banks = 1_us << radix_lane_bits;

		template <typename T>
		using radix_histogram = vector<T, radix_lane_banks>;

		struct radix_binning {
			usize radix_lane_idx;

			radix_binning(usize idx) : radix_lane_idx(idx) {}

			__host__ __device__
			usize operator()(u32 key) const noexcept {
				return (key >> (radix_lane_idx * radix_lane_bits)) & (radix_lane_banks - 1);
			}
		};

		struct match_any {
			__device__
			u32 operator()(cg::thread_block_tile<32> const & warp, bool is_thread_active, u32 value) const noexcept {
				// for (usize i = 0; i < ballot_buffer.size / 32; ++i) {
				// 	ballot_buffer[i * 32 + warp.thread_rank()] = 0_u32;
				// }
				// warp.sync();

				// if (is_thread_active) {
				// 	atomicOr(ballot_buffer.addr_at(value), 1_u32 << warp.thread_rank());
				// }
				// warp.sync();

				// return ballot_buffer[value];

				u32 mask = warp.ballot(is_thread_active);
				for (usize i = 0; i < radix_lane_bits; ++i) {
					bool const bit = (value >> i) & 1_u32;
					u32 const ballot = warp.ballot(bit);
					mask &= bit ? ballot : (~ ballot);
				}
				return mask;
			}
		};

	}

	

	namespace kernel {

		template <bool WithIndices, typename InputIterator, typename OutputIterator>
		cudaError_t generic_radix_sort(void * d_temp_storage, usize & d_temp_storage_bytes,
			u32 const * keys_in, u32 * keys_out, usize size,
			InputIterator values_in, OutputIterator values_out) {
			
			usize constexpr	radix_sort_items_per_thread		= 16;
			usize constexpr	radix_sort_threads_per_block	= details::radix_lane_banks;
			usize constexpr	radix_sort_items_per_block		= radix_sort_items_per_thread * radix_sort_threads_per_block;
			
			usize const		radix_sort_num_blocks			= ceiled_div(size, radix_sort_items_per_block);

			usize const		d_grid_histogram_items			= 1_us;
			usize const		d_block_histogram_items			= radix_sort_num_blocks;
			usize const		d_keys_temp_items				= size;
			usize const		d_indices_0_items				= WithIndices ? size : 0_us;
			usize const		d_indices_1_items				= WithIndices ? size : 0_us;
			memory_sections < details::radix_histogram<u32>, details::radix_histogram<lookback<u32>>, u32, u32, u32 > sections {
				d_temp_storage, { d_grid_histogram_items, d_block_histogram_items, d_keys_temp_items, d_indices_0_items, d_indices_1_items }
			};

			if (d_temp_storage == nullptr) {
				d_temp_storage_bytes = sections.size();

				return cudaSuccess;
			}
			else {
				auto [p_grid_histogram, block_histograms, keys_temp, indices_0, indices_1] = sections.ptrs();

				u32 const *	keys_in_pass[]		= { keys_in,	keys_temp,	keys_out,	keys_temp };
				u32 *		keys_out_pass[]		= { keys_temp,	keys_out,	keys_temp,	keys_out };
				u32 const * indices_in_pass[]	= { indices_0,	indices_1,	indices_0,	indices_1 };
				u32 *		indices_out_pass[]	= { indices_1,	indices_0,	indices_1,	indices_0 };

				if constexpr (WithIndices) {
					auto result = kernel::copy(indices_0, counting_iterator<u32>(0_u32), size);
					if (cudaSuccess != result) return result;
				}

				for (usize i_pass = 0; i_pass < 4; ++i_pass) {

					auto binning = details::radix_binning(i_pass);
					auto match_any = details::match_any();

					/* histogram */ {
						auto result = kernel::histogram(keys_in_pass[i_pass], size, p_grid_histogram, binning);
						if (cudaSuccess != result) return result;
					}

					/* initialize block histograms, and exclusively scan the grid histogram */ {
						auto fn			= initialize_histograms<details::radix_histogram<u32>::size(), details::radix_histogram>;
						auto grid_dim	= dim3(radix_sort_num_blocks);
						auto block_dim	= dim3(details::radix_histogram<u32>::size());

						auto result		= launch_kernel(fn, grid_dim, block_dim,
							block_histograms, p_grid_histogram
						);
						if (cudaSuccess != result) return result;
					}

					/* radix sort */ {
						auto fn			= generic_radix_sort_scan_lane<WithIndices, radix_sort_threads_per_block, radix_sort_items_per_thread, details::radix_histogram, details::radix_binning, details::match_any>;
						auto grid_dim	= dim3(radix_sort_num_blocks);
						auto block_dim	= dim3(radix_sort_threads_per_block);

						auto result		= launch_kernel(fn, grid_dim, block_dim,
							keys_in_pass[i_pass], keys_out_pass[i_pass], size,
							indices_in_pass[i_pass], indices_out_pass[i_pass],
							block_histograms, p_grid_histogram, binning, match_any
						);
						if (cudaSuccess != result) return result;
					}

				}

				if constexpr (WithIndices) {
					auto result = kernel::transform(indices_0, values_out, size,
						[=] __device__ (u32 const index) {
							return values_in[index];
						}
					);
					if (cudaSuccess != result) return result;
				}

				return cudaSuccess;

			}

		} // generic_radix_sort



		inline
		cudaError_t radix_sort(void * d_temp_storage, usize & d_temp_storage_bytes,
			u32 const * keys_in, u32 * keys_out, usize size) {

			return generic_radix_sort<false>(
				d_temp_storage, d_temp_storage_bytes,
				keys_in, keys_out, size,
				nullptr, nullptr
			);
		}

	} // namespace kernel

} // namespace spp



#endif // SPP_RADIX_SORT_HPP