#ifndef SPP_TEST_TESTER_HPP
#define SPP_TEST_TESTER_HPP

#include <iostream>
#include <iomanip>

#include "log.hpp"
#include "event.hpp"
#include "device_ptr.hpp"



#define spp_test_functor_of(fn) [] (auto && ... args) { return fn(std::forward<decltype(args)>(args)...); }

namespace spp::test {

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
			void * ptr = temp_storage.get();
			cudaCheck(std::forward<Fn>(fn)(ptr, temp_storage_bytes, std::forward<Args>(args)...));
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
	bool compare(std::vector<T> const & ground_truth, std::vector<T> const & test, std::size_t max_print_count = 16) {
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

		return error_count == 0;
	}

}



#endif // SPP_TEST_TESTER_HPP