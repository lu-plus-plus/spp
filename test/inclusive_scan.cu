
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
#include "spp/device_vector.hpp"
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



	template <typename T>
	void compare(std::vector<T> const & ground_truth, std::vector<T> const & test, std::size_t max_print_count = 16) {
		std::size_t error_count = 0;

		for (std::size_t i = 0; i < std::min(ground_truth.size(), test.size()); ++i) {
			auto const a = ground_truth[i];
			auto const b = test[i];
			if (a != b) {
				if (error_count++ < max_print_count) {
					std::cout << "[error] at position (" << i << "), ground truth = " << a << ", test data = " << b << std::endl;
				}
			}
		}

		if (ground_truth.size() != test.size()) {
			std::cout << "[error] the size of ground truth " << ground_truth.size() << " is not equal to the size of test input " << test.size() << std::endl;
		}
		std::cout << "[info] element count = " << std::min(ground_truth.size(), test.size()) << std::endl;
		std::cout << "[info] error count = " << error_count << std::endl;
	}

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

	auto d_data_in = spp::device_vector<data_t>(h_data_in);

	auto d_data_out_cub = spp::device_vector<data_t>(item_count);
	auto d_data_out_spp = spp::device_vector<data_t>(item_count);

	spp::tester<size_t>(
		spp_test_functor_of(cub::DeviceScan::InclusiveSum),
		d_data_in.data(), d_data_out_cub.data(), item_count
	).benchmark(
		spp_test_functor_of(cub::DeviceScan::InclusiveSum),
		d_data_in.data(), d_data_out_cub.data(), item_count
	);

	spp::tester<spp::usize>(
		spp_test_functor_of(spp::kernel::inclusive_scan),
		d_data_in.data(), d_data_out_spp.data(), item_count
	).benchmark(
		spp_test_functor_of(spp::kernel::inclusive_scan),
		d_data_in.data(), d_data_out_spp.data(), item_count
	);

	auto h_data_out_cub = d_data_out_cub.to_host();
	auto h_data_out_spp = d_data_out_spp.to_host();

	spp::compare(h_data_out_cub, h_data_out_spp);

	return 0;
}