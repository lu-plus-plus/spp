
#include <vector>
#include <random>
#include <numeric>
#include <iostream>
#include <iomanip>

#include "cub/device/device_scan.cuh"

#include "spp/types.hpp"
#include "spp/log.hpp"
#include "spp/event.hpp"
#include "spp/device_ptr.hpp"
#include "spp/scan.hpp"



using spp::usize;

using data_t = spp::u32;
using data_dist_t = std::uniform_int_distribution<data_t>;



#define spp_test_functor_of(fn) [] (auto && ... args) { return fn(std::forward<decltype(args)>(args)...); }

namespace spp {

	template <typename SizeTy>
	struct tester {

		spp::device_ptr<void> temp_storage;
		SizeTy temp_storage_bytes;

		static constexpr
		SizeTy default_warmup = 3;
		
		static constexpr
		SizeTy default_repeat = 10;

		SizeTy warmup = default_warmup;
		SizeTy repeat = default_repeat;

		template <typename Fn, typename ... Args>
		tester(Fn && fn, Args && ... args) : temp_storage(nullptr), temp_storage_bytes(0) {
			
			void * ptr = nullptr;
			SizeTy bytes = 0;
			
			cudaCheck(std::forward<Fn>(fn)(ptr, bytes, std::forward<Args>(args)...));
			cudaCheck(cudaMalloc(&ptr, bytes));

			temp_storage = ptr;
			temp_storage_bytes = bytes;
		}

		template <typename Fn, typename ... Args>
		void run(Fn && fn, Args && ... args) {
			cudaCheck(std::forward<Fn>(fn)(temp_storage.get(), temp_storage_bytes, std::forward<Args>(args)...));
		}

		tester & config(SizeTy warmup_ = default_warmup, SizeTy repeat_ = default_repeat) {
			warmup = warmup_;
			repeat = repeat_;

			return *this;
		}

		template <typename Fn, typename ... Args>
		tester & benchmark(Fn && fn, Args && ... args) {
			for (SizeTy i = 0; i < warmup; ++i) {
				run(std::forward<Fn>(fn), std::forward<Args>(args)...);
			}

			float total_time = 0.0f;

			with_(spp::event start, stop) {
				for (SizeTy i = 0; i < repeat; ++i) {
					start.record();
					
					run(std::forward<Fn>(fn), std::forward<Args>(args)...);
					
					stop.record();
					stop.synchronize();
					
					float const time = stop.elapsed_time_from(start);
					total_time += time;
					std::cout << "[repeat " << std::setw(4) << i << "] time = " << time << " ms\n";
				}
			}

			std::cout << "[average] time = " << total_time / repeat << " ms\n\n";

			return *this;
		}

	};

}



int main(void) {

	usize constexpr item_count = 1u << 22;

	std::mt19937 gen(std::random_device{}());
	data_dist_t dist(0, 2);
	auto rand_data = [&] () -> data_t { return dist(gen); };

	std::vector<data_t> h_data_in;
	h_data_in.reserve(item_count);
	for (usize i = 0; i < item_count; ++i) {
		h_data_in.emplace_back(rand_data());
	}
	
	auto d_data_in = spp::device_alloc<data_t>(h_data_in.data(), h_data_in.data() + item_count);

	auto d_data_out_cub = spp::device_alloc<data_t>(item_count);
	auto d_data_out_spp = spp::device_alloc<data_t>(item_count);

	spp::tester<size_t>(
		spp_test_functor_of(cub::DeviceScan::InclusiveSum),
		d_data_in.get(), d_data_out_cub.get(), item_count
	).benchmark(
		spp_test_functor_of(cub::DeviceScan::InclusiveSum),
		d_data_in.get(), d_data_out_cub.get(), item_count
	);

	spp::tester<spp::usize>(
		spp_test_functor_of(spp::kernel::inclusive_scan),
		d_data_in.get(), d_data_out_spp.get(), item_count
	).benchmark(
		spp_test_functor_of(spp::kernel::inclusive_scan),
		d_data_in.get(), d_data_out_spp.get(), item_count
	);

	std::vector<data_t> h_data_out_cub(item_count);
	cudaMemcpy(h_data_out_cub.data(), d_data_out_cub.get(), sizeof(data_t) * item_count, cudaMemcpyDeviceToHost);

	std::vector<data_t> h_data_out_spp(item_count);
	cudaMemcpy(h_data_out_spp.data(), d_data_out_spp.get(), sizeof(data_t) * item_count, cudaMemcpyDeviceToHost);

	usize error_count = 0;

	for (usize i = 0; i < item_count; ++i) {
		data_t const truth = h_data_out_cub[i];
		data_t const test = h_data_out_spp[i];
		if (truth != test) {
			if (error_count == 0) std::cout << "element " << i << ": ground truth = " << truth << ", test = " << test << std::endl;
			++error_count;
		}
	}

	std::cout << "total length = " << item_count << std::endl;
	std::cout << "error count = " << error_count << std::endl;



	return 0;
}