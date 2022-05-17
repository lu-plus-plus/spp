#ifndef SPP_RADIX_SORT_HPP
#define SPP_RADIX_SORT_HPP

#include "scan.hpp"

#include <cooperative_groups/reduce.h>



namespace spp {

	template <typename T, usize S>
	struct vector {

		static_assert(std::is_trivially_default_constructible_v<T>, "T must be trivially default-constructible.");
		static_assert(std::is_trivially_copy_constructible_v<T>, "T must be trivially copy-constructible.");
		static_assert(std::is_trivially_move_constructible_v<T>, "T must be trivially move-constructible.");

		// properties

		using element_type = T;

		static constexpr
		usize size = S;

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

		// __host__ __device__
		// vector & operator+=(vector const & other) noexcept {
		// 	for (usize i = 0; i < S; ++i) {
		// 		data[i] += other.data[i];
		// 	}
		// }

		// __host__ __device__
		// vector volatile & operator+=(vector const & other) volatile noexcept {
		// 	for (usize i = 0; i < S; ++i) {
		// 		data[i] += other.data[i];
		// 	}
		// }

		// template <typename T, usize S>
		// __host__ __device__
		// friend vector<T, S> operator+(vector<T, S> const & a, vector<T, S> const & b) noexcept {
		// 	return vector(a) += b;
		// }
		
		template <usize TileSize, typename = std::enable_if_t<S == TileSize>>
		__host__ __device__
		static
		T bind(cg::thread_block_tile<TileSize> const & tile) noexcept {
			return T();
		}

		template <usize TileSize, typename = std::enable_if_t<S == TileSize>>
		__host__ __device__
		T & bind(cg::thread_block_tile<TileSize> const & tile) noexcept {
			return (*this)[tile.thread_rank()];
		}

		template <usize TileSize, typename = std::enable_if_t<S == TileSize>>
		__host__ __device__
		T const & bind(cg::thread_block_tile<TileSize> const & tile) const noexcept {
			return (*this)[tile.thread_rank()];
		}

		template <usize TileSize, typename = std::enable_if_t<S == TileSize>>
		__host__ __device__
		T volatile & bind(cg::thread_block_tile<TileSize> const & tile) volatile noexcept {
			return (*this)[tile.thread_rank()];
		}

		template <usize TileSize, typename = std::enable_if_t<S == TileSize>>
		__host__ __device__
		T const volatile & bind(cg::thread_block_tile<TileSize> const & tile) const volatile noexcept {
			return (*this)[tile.thread_rank()];
		}

	};



	template <typename UInt, typename = std::enable_if_t<std::is_unsigned_v<UInt>>>
	struct lookback {

		static constexpr
		usize bits = sizeof(UInt) * 8;

		static constexpr
		UInt prefix_bit = UInt(1) << (bits - 1);

		static constexpr
		UInt aggregate_bit = UInt(1) << (bits - 2);

		static constexpr
		UInt value_mask = (~ UInt(0)) >> 2;

		static constexpr
		UInt flags_mask = UInt(0b11) << (bits - 2);

		UInt data;

		lookback() noexcept = default;
		lookback(lookback const &) noexcept = default;
		lookback(lookback &&) noexcept = default;
		~lookback() noexcept = default;

		lookback(UInt) noexcept = delete;

		lookback & operator=(lookback const &) noexcept = default;
		lookback & operator=(lookback &&) noexcept = default;
		
		__host__ __device__
		lookback & operator=(lookback const volatile & rhs) noexcept {
			data = rhs.data;
		}

		__host__ __device__
		lookback volatile & operator=(lookback const & rhs) volatile noexcept {
			data = rhs.data;
		}

		static constexpr
		__host__ __device__
		lookback zero() noexcept {
			return lookback{ 0 };
		}

		static constexpr
		__host__ __device__
		lookback make_aggregate(UInt value) noexcept {
			return lookback{ (value & value_mask) | aggregate_bit };
		}

		static constexpr
		__host__ __device__
		lookback make_prefix(UInt value) noexcept {
			return lookback{ (value & value_mask) | prefix_bit };
		}

		__host__ __device__
		bool is_prefixed() const noexcept {
			return (data & prefix_bit) != 0;
		}

		__host__ __device__
		bool is_aggregated() const noexcept {
			return (data & aggregate_bit) != 0;
		}

		__host__ __device__
		bool is_invalid() const noexcept {
			return (data & flags_mask) == 0;
		}

		__host__ __device__
		UInt get() const noexcept {
			return data & value_mask;
		}
		
	};

	constexpr bool test = std::is_trivially_copyable_v<lookback<u32>>;



	inline
	__device__
	u32 thread_mask_lt(cg::thread_block_tile<32> const & warp) {
		return (1 << warp.thread_rank()) - 1;
	}



	template <usize ThreadsPerBlock, usize ItemsPerThread, typename Histogram, typename Binning>
	__global__
	void histogram(u32 const * items, usize num_items, Histogram * p_grid_histogram, Binning binning) {
	
		usize constexpr WarpsPerBlock	= ThreadsPerBlock / 32;
		usize constexpr ItemsPerWarp	= ItemsPerThread * 32;

		auto const grid		= cg::this_grid();
		auto const block	= cg::this_thread_block();
		auto const warp		= cg::tiled_partition<32>(block);

		Histogram & grid_histogram = *p_grid_histogram;

		for (usize bin_rank = grid.thread_rank(); bin_rank < Histogram::size; bin_rank += grid.num_threads()) {
			grid_histogram[bin_rank] = 0;
		}

		__shared__ Histogram block_histogram;

		for (usize bin_rank = block.thread_rank(); bin_rank < Histogram::size; bin_rank += block.num_threads()) {
			block_histogram[bin_rank] = 0;
		}

		grid.sync();

		usize const i_warp_begin	= grid.block_rank() * WarpsPerBlock + warp.meta_group_rank();
		usize const i_warp_end		= ceiled_div(num_items, ItemsPerWarp);
		usize const i_warp_step		= grid.num_blocks() * warp.meta_group_size();

		for (usize i_warp = i_warp_begin; i_warp < i_warp_end; i_warp += i_warp_step) {
			usize const thread_rank_begin = i_warp * ItemsPerWarp + warp.thread_rank();

			u32 thread_items[ItemsPerThread];

			for (usize j_item = 0; j_item < ItemsPerThread; ++j_item) {
				usize const item_rank = thread_rank_begin + 32 * j_item;
				if (item_rank < num_items) {
					thread_items[j_item] = items[item_rank];
				}
			}

			for (usize j_item = 0; j_item < ItemsPerThread; ++j_item) {
				usize const item_rank = thread_rank_begin + 32 * j_item;
				if (item_rank < num_items) {
					usize const key = binning(thread_items[j_item]);
					atomicAdd(&(block_histogram[key]), 1);
				}
			}
		}

		block.sync();

		for (usize bin_rank = block.thread_rank(); bin_rank < Histogram::size; bin_rank += block.num_threads()) {
			atomicAdd(&(grid_histogram[bin_rank]), block_histogram[bin_rank]);
		}

		grid.sync();

		if (grid.block_rank() == 0 and warp.meta_group_rank() == 0) {
			usize partial_result = 0_us;
			
			for (usize i = 0; i < Histogram::size / 32; ++i) {
				usize const bin_rank = i * 32 + warp.thread_rank();
				usize const bin_value = grid_histogram[bin_rank];
				
				usize const exclusive_prefix = partial_result + cg::exclusive_scan(warp, bin_value);
				grid_histogram[bin_rank] = exclusive_prefix;
				
				usize const inclusive_prefix = exclusive_prefix + bin_value;
				partial_result = warp.shfl(inclusive_prefix, 31);
			}
		}

	}



	template <template <typename> typename Histogram>
	__global__
	void init_block_radix_histograms(Histogram<lookback<u32>> * block_radix_histograms) {
		auto const grid		= cg::this_grid();
		auto const block	= cg::this_thread_block();
		block_radix_histograms[grid.block_rank()][block.thread_rank()] = lookback<u32>::zero();
	}



	template <usize ItemsPerThread, usize ThreadsPerBlock, template <typename> typename Histogram, typename Binning, typename MatchAny>
	__global__
	void radix_scan_by_key_with_large_block(u32 const * keys_in, u32 * keys_out, usize num_keys,
		Histogram<lookback<u32>> volatile * block_radix_histograms,
		Histogram<u32> const * p_grid_radix_histogram,
		Binning binning, MatchAny match_any) {
		
		usize constexpr WarpsPerBlock	= ThreadsPerBlock / 32;
		usize constexpr ItemsPerWarp	= ItemsPerThread * 32;
		usize constexpr ItemsPerBlock	= ItemsPerThread * ThreadsPerBlock;

		usize constexpr HistogramSize	= Histogram<u32>::size;

		auto const grid					= cg::this_grid();
		auto const block				= cg::this_thread_block();
		auto const warp					= cg::tiled_partition<32>(block);

		usize const block_rank_begin	= grid.block_rank() * ItemsPerBlock;
		usize const warp_rank_begin		= warp.meta_group_rank() * ItemsPerWarp;



		u32 thread_keys[ItemsPerThread];
		usize thread_exclusive_prefixes[ItemsPerThread];

		__shared__ Histogram<u32> shared_warp_exclusive_prefixes[WarpsPerBlock];
		__shared__ Histogram<u32> shared_block_exclusive_prefixes;
		__shared__ Histogram<u32> shared_grid_exclusive_prefix;

		if (block.thread_rank() < HistogramSize) {
			usize const radix_bank = block.thread_rank();
			for (usize i_warp = 0; i_warp < WarpsPerBlock; ++i_warp) {
				shared_warp_exclusive_prefixes[i_warp][radix_bank] = 0_u32;
			}
			shared_block_exclusive_prefixes[radix_bank] = 0_u32;
			shared_grid_exclusive_prefix[radix_bank] = (*p_grid_radix_histogram)[radix_bank];
		}

		block.sync();



		for (usize i_key = 0; i_key < ItemsPerThread; ++i_key) {
			usize const in_rank = block_rank_begin + warp_rank_begin + 32 * i_key + warp.thread_rank();
			bool const is_active = in_rank < num_keys;

			auto const key = is_active ? keys_in[in_rank] : 0_u32;
			auto const radix_bank = binning(key);

			auto const active_mask = warp.ballot(is_active);
			auto const match_mask = match_any(active_mask, radix_bank);
			auto const match_count = __popc(match_mask);
			auto const match_exclusive_prefix = __popc(match_mask & thread_mask_lt(warp));
			
			auto const thread_exclusive_prefix = shared_warp_exclusive_prefixes[warp.meta_group_rank()][radix_bank] + match_exclusive_prefix;
			warp.sync();

			thread_keys[i_key] = key;
			thread_exclusive_prefixes[i_key] = thread_exclusive_prefix;

			if (match_exclusive_prefix + 1 == match_count) {
				shared_warp_exclusive_prefixes[warp.meta_group_rank()][radix_bank] += match_count;
			}
			warp.sync();
		}

		block.sync();



		if (block.thread_rank() < HistogramSize) {
			usize const radix_bank = block.thread_rank();

			auto block_reduction = 0_u32;

			for (usize i_warp = 0; i_warp < WarpsPerBlock; ++i_warp) {
				u32 const warp_reduction = shared_warp_exclusive_prefixes[i_warp][radix_bank];
				shared_warp_exclusive_prefixes[i_warp][radix_bank] = block_reduction;
				block_reduction += warp_reduction;
			}

			block_radix_histograms[grid.block_rank()][radix_bank] = lookback<u32>::make_aggregate(block_reduction);

			auto block_exclusive_prefix = 0_u32;

			for (isize i_block = isize(grid.block_rank()) - 1_is; 0 <= i_block; --i_block) {
				auto block_lookback = lookback<u32>::zero();
				do block_lookback = block_radix_histograms[i_block][radix_bank]; while (block_lookback.is_invalid());

				block_exclusive_prefix += block_lookback.get();
				if (block_lookback.is_prefixed()) break;
			}

			shared_block_exclusive_prefixes[radix_bank] = block_exclusive_prefix;

			block_radix_histograms[grid.block_rank()][radix_bank] = lookback<u32>::make_prefix(block_exclusive_prefix + block_reduction);
		}

		block.sync();



		for (usize i_key = 0; i_key < ItemsPerThread; ++i_key) {
			usize const in_rank = block_rank_begin + warp_rank_begin + 32 * i_key + warp.thread_rank();

			if (in_rank < num_keys) {
				usize const radix_bank = binning(thread_keys[i_key]);

				usize const out_rank = shared_grid_exclusive_prefix[radix_bank]
					+ shared_block_exclusive_prefixes[radix_bank]
					+ shared_warp_exclusive_prefixes[warp.meta_group_rank()][radix_bank]
					+ thread_exclusive_prefixes[i_key];
					
				keys_out[out_rank] = thread_keys[i_key];

				// if (out_rank < 16) {
				// 	printf("i_key = %u, in rank = %u, out rank = %u, value = %x, (%u, %u, %u, %x) from thread %u\n",
				// 		i_key,
				// 		in_rank,
				// 		out_rank,
				// 		thread_keys[i_key],
				// 		u32(shared_grid_exclusive_prefix[radix_bank]),
				// 		u32(shared_block_exclusive_prefixes[radix_bank]),
				// 		u32(shared_warp_exclusive_prefixes[warp.meta_group_rank()][radix_bank]),
				// 		u32(thread_exclusive_prefixes[i_key]),
				// 		u32(grid.thread_rank())
				// 	);
				// }
			}
		}

	}



	// template <usize ItemsPerThread, usize ThreadsPerBlock, usize RadixLaneBits, typename Histogram, typename Binning, typename MatchAny>
	// __global__
	// void radix_scan_by_key(u32 const * keys_in, u32 * keys_out, usize num_keys,
	// 	block_descriptor<Histogram> volatile * block_descriptors,
	// 	Histogram const * p_radix_bank_histogram,
	// 	Binning binning, MatchAny match_any) {

	// 	usize constexpr WarpsPerBlock	= ThreadsPerBlock / 32;
	// 	usize constexpr ItemsPerWarp	= ItemsPerThread * 32;
	// 	usize constexpr ItemsPerBlock	= ItemsPerThread * ThreadsPerBlock;



	// 	auto const grid					= cg::this_grid();
	// 	auto const block				= cg::this_thread_block();
	// 	auto const warp					= cg::tiled_partition<32>(block);

	// 	usize const block_rank_begin	= grid.block_rank() * ItemsPerBlock;
	// 	usize const warp_rank_begin		= warp.meta_group_rank() * ItemsPerWarp;
		


	// 	u32 thread_keys[ItemsPerThread];
	// 	usize thread_key_exclusive_prefixes[ItemsPerThread];

	// 	__shared__ Histogram warp_exclusive_prefixes[WarpsPerBlock];
	// 	__shared__ Histogram block_exclusive_prefix;

	// 	static_assert(block_exclusive_prefix.size() % 32 == 0, "The width of histogram is not a multiple of 32.");

	// 	for (usize i = 0; i < block_exclusive_prefix.size() / 32; ++i) {
	// 		usize const radix_bank = i * 32 + warp.meta_group_rank();
	// 		warp_exclusive_prefixes[warp.meta_group_rank()][radix_bank] = 0;
	// 	}

	// 	__shared__ Histogram radix_bank_histogram;

	// 	for (usize radix_bank = block.thread_rank(); radix_bank < radix_bank_histogram.size(); radix_bank += block.num_threads()) {
	// 		radix_bank_histogram[block.thread_rank()] = (*p_radix_bank_histogram)[block.thread_rank()];
	// 	}

	// 	block.sync();



	// 	barrier<1, WarpsPerBlock>		warp_reduction_ready;
	// 	barrier<2, WarpsPerBlock - 1>	warp_prefix_ready;
	// 	barrier<3, WarpsPerBlock>		block_prefix_ready;
		


	// 	for (usize i_key = 0; i_key < ItemsPerThread; ++i_key) {
	// 		usize const in_rank = block_rank_begin + warp_rank_begin + 32 * i_key + warp.thread_rank();
			
	// 		if (in_rank < num_keys) {
	// 			auto const key = keys_in[in_rank];

	// 			auto const active_mask = __activemask();
	// 			auto const match_mask = match_any(active_mask, key);
	// 			auto const match_count = __popc(match_mask);
	// 			auto const match_exclusive_prefix = __popc(match_mask & thread_mask_lt(warp));

	// 			auto const key_bank = binning(key);
	// 			if (match_exclusive_prefix + 1 == match_count) {
	// 				warp_exclusive_prefixes[key_bank] += match_count;
	// 			}

	// 			thread_keys[i_key] = key;
	// 			thread_key_exclusive_prefixes[i_key] = match_exclusive_prefix;
	// 		}
	// 		else {
	// 			thread_keys[i_key] = 0;
	// 			thread_key_exclusive_prefixes[i_key] = ~ 0_us;
	// 		}
	// 	}



	// 	if (warp.meta_group_rank() == 0) {
	// 		warp_reduction_ready.arrive();

	// 		auto add_histogram = [&] (Histogram & a, Histogram const & b) {
	// 			for (usize i = 0; i < a.size() / 32; ++i) {
	// 				usize const radix_bank = i * 32 + warp.thread_rank();
	// 				a[radix_bank] += b[radix_bank];
	// 			}
	// 		};

	// 		for (auto i_block = isize(grid.block_rank()) - 1; 0 <= i_block; --i_block) {
	// 			block_status const status = block_descriptors[i_block].load_status_spinning();
				
	// 			if (status == block_status::aggregated) {
	// 				add_histogram(block_exclusive_prefix, block_descriptors[i_block].internal_aggregate);
	// 			}
	// 			else {
	// 				add_histogram(block_exclusive_prefix, block_descriptors[i_block].inclusive_prefix);
	// 				break;
	// 			}
	// 		}

	// 		block_prefix_ready.arrive();
	// 	}
	// 	else if (warp.meta_group_rank() + 1 == WarpsPerBlock) {
	// 		warp_reduction_ready.wait();

	// 		auto radix_vector = vector<u16, 32>::bind(warp);
			
	// 		auto const subwarp = cg::tiled_partition<RadixBankCount>(warp);

	// 		if (subwarp.meta_group_rank() == 0) {
	// 			for (usize i_warp = 0; i_warp < WarpsPerBlock; ++i_warp) {
	// 				auto const prefix = warp_bank_exclusive_prefixes[i_warp].bind(subwarp);
	// 				warp_bank_exclusive_prefixes[i_warp].bind(subwarp) += radix_vector;
	// 				radix_vector += prefix;
	// 			}
	// 		}

	// 		warp_prefix_ready.arrive();

	// 		if (subwarp.meta_group_rank() == 0) {
	// 			block_descriptors[grid.block_rank()].internal_aggregate.bind(subwarp) = radix_vector;
	// 			__threadfence();
	// 			block_descriptors[grid.block_rank()].status = block_status::aggregated;
	// 		}

	// 		block_prefix_ready.wait();

	// 		if (subwarp.meta_group_rank() == 0) {
	// 			block_descriptors[grid.block_rank()].internal_aggregate.bind(subwarp) = block_exclusive_prefix.bind(subwarp) + radix_vector;
	// 			__threadfence();
	// 			block_descriptors[grid.block_rank()].status = block_status::prefixed;
	// 		}
	// 	}
	// 	else {
	// 		warp_reduction_ready.arrive();

	// 		warp_prefix_ready.wait();

	// 		block_prefix_ready.wait();
	// 	}



	// 	for (usize i_key = 0; i_key < ItemsPerThread; ++i_key) {
	// 		usize const radix_bank = binning(thread_keys[i_key]);
	// 		usize const in_rank = block_rank_begin + warp_rank_begin + 32 * i_key + warp.thread_rank();
	// 		usize const out_rank = radix_bank_histogram[radix_bank] + block_exclusive_prefix[radix_bank] + warp_exclusive_prefixes[warp.meta_group_rank()][radix_bank] + thread_exclusive_prefixes[i_key];

	// 		if (in_rank < num_keys) keys_out[out_rank] = thread_keys[i_key];
	// 		else break;
	// 	}

	// }



	namespace details {

		inline constexpr
		usize radix_lane_bits = 8;

		template <typename T>
		using radix_histogram = vector<T, (1 << radix_lane_bits)>;

		struct radix_binning {
			usize radix_lane_idx;

			radix_binning(usize idx) : radix_lane_idx(idx) {}

			__host__ __device__
			usize operator()(u32 key) const noexcept {
				return (key >> (radix_lane_idx * radix_lane_bits)) & ((1 << radix_lane_bits) - 1);
			}
		};

		struct match_any {
			__device__
			u32 operator()(u32 mask, u32 value) const noexcept {
				u32 match = mask;
				for (usize i = 0; i < radix_lane_bits; ++i) {
					bool const bit = (value >> i) & 1_u32;
					u32 const ballot = __ballot_sync(mask, bit);
					match &= bit ? ballot : (~ ballot);
				}
				return match;
			}
		};

	}

	

	namespace kernel {

		inline
		cudaError_t radix_sort_by_key(u32 const * keys_in, u32 * keys_out, usize num_keys,
			void * d_temp_storage, usize & d_temp_storage_bytes) {
			
			usize constexpr items_per_thread = 16;
			usize constexpr threads_per_block = 1 << details::radix_lane_bits;
			usize constexpr items_per_block = items_per_thread * threads_per_block;
			
			usize const radix_sort_num_blocks = ceiled_div(num_keys, items_per_block);

			usize const d_histogram_bytes = ceiled_div(sizeof(details::radix_histogram<u32>), 128) * 128;
			usize const d_radix_sort_bytes = ceiled_div(sizeof(details::radix_histogram<lookback<u32>>) * radix_sort_num_blocks, 128) * 128;
			usize const d_keys_temp_bytes = ceiled_div(sizeof(u32) * num_keys, 128) * 128;

			if (d_temp_storage == nullptr or d_temp_storage_bytes == 0) {
				d_temp_storage_bytes = d_histogram_bytes + d_radix_sort_bytes + d_keys_temp_bytes;

				return cudaSuccess;
			}
			else {
				auto p_grid_radix_histogram = static_cast<std::byte *>(d_temp_storage);
				auto block_radix_histograms = static_cast<std::byte *>(d_temp_storage) + d_histogram_bytes;
				auto keys_temp = static_cast<std::byte *>(d_temp_storage) + d_histogram_bytes + d_radix_sort_bytes;

				void * keys_ins[] = { &keys_in, &keys_temp, &keys_out, &keys_temp };
				void * keys_outs[] = { &keys_temp, &keys_out, &keys_temp, &keys_out };

				// [todo] initialize block lookbacks

				for (usize i_pass = 0; i_pass < 4; ++i_pass) {

					auto binning = details::radix_binning(i_pass);
					auto match_any = details::match_any();

					/* histogram */ {
						usize constexpr histogram_block_threads = 512;
						usize constexpr histogram_thread_items = 16;

						auto fn_histogram = reinterpret_cast<void const *>(histogram<histogram_block_threads, histogram_thread_items, details::radix_histogram<u32>, details::radix_binning>);
						
						i32 histogram_grid_blocks = 0;
						cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&histogram_grid_blocks, fn_histogram, histogram_block_threads, 0));
						
						cudaDeviceProp prop;
						cudaGetDeviceProperties(&prop, 0);
						histogram_grid_blocks *= prop.multiProcessorCount;

						auto grid_dim	= dim3(histogram_grid_blocks);
						auto block_dim	= dim3(histogram_block_threads);
						void * args[]	= { keys_ins[i_pass], &num_keys, &p_grid_radix_histogram, &binning };

						auto result = cudaLaunchCooperativeKernel(fn_histogram, grid_dim, block_dim, args);
						if (cudaSuccess != result) return result;
					}

					/* init radix lookback */ {
						auto fn_init_lookback = reinterpret_cast<void const *>(init_block_radix_histograms<details::radix_histogram>);

						auto block_dim	= dim3(256);
						auto grid_dim	= dim3(radix_sort_num_blocks);
						void * args[]	= { &block_radix_histograms };

						auto result		= cudaLaunchKernel(fn_init_lookback, grid_dim, block_dim, args);
						if (cudaSuccess != result) return result;
					}

					/* radix sort */ {
						usize constexpr radix_pass_threads_per_block = 1 << details::radix_lane_bits;
						usize constexpr radix_pass_items_per_thread = 16;

						auto fn_radix_pass = reinterpret_cast<void const *>(radix_scan_by_key_with_large_block<radix_pass_items_per_thread, radix_pass_threads_per_block, details::radix_histogram, details::radix_binning, details::match_any>);

						auto grid_dim	= dim3(radix_sort_num_blocks);
						auto block_dim	= dim3(radix_pass_threads_per_block);
						void * args[]	= { keys_ins[i_pass], keys_outs[i_pass], &num_keys, &block_radix_histograms, &p_grid_radix_histogram, &binning, &match_any };

						auto result = cudaLaunchKernel(fn_radix_pass, grid_dim, block_dim, args);
						if (cudaSuccess != result) return result;
					}

				}

				return cudaSuccess;

			}
		}

	}

} // namespace spp



#endif // SPP_RADIX_SORT_HPP