
#include <vector>
#include <random>
#include <numeric>
#include <iostream>
#include <iterator>

#include "thrust/execution_policy.h"
#include "thrust/scan.h"

#include "spp/types.hpp"
#include "spp/log.hpp"
#include "spp/event.hpp"
#include "spp/device_ptr.hpp"
#include "spp/reduce.hpp"



int main(void) {
	using spp::u32;
	using data_t = spp::i32;

	using index_dist_t = std::uniform_int_distribution<u32>;
	using data_dist_t = std::uniform_int_distribution<data_t>;

	constexpr u32 seg_count = 128;

	std::mt19937 gen(std::random_device{}());

	index_dist_t dist_seg_type(1, seg_count);
	index_dist_t dist_seg_length[] = {
		index_dist_t(1, 32),
		index_dist_t(32 + 1, 1024),
		index_dist_t(1024 + 1, 1024 * 8)
	};
	auto rand_length = [&] () -> u32 {
		u32 const r = dist_seg_type(gen);
		if (r < seg_count * 0.7) return dist_seg_length[0](gen);
		else if (r < seg_count * 0.95) return dist_seg_length[1](gen);
		else return dist_seg_length[2](gen);
	};

	data_dist_t dist_data(1, 2);

	std::vector<u32> seg_lengths;
	for (u32 i = 0; i < seg_count; ++i) seg_lengths.push_back(rand_length());

	u32 const total_length = std::accumulate(seg_lengths.begin(), seg_lengths.end(), u32(0));
	std::inclusive_scan(seg_lengths.begin(), seg_lengths.end(), seg_lengths.begin());

	std::vector<u32> h_keys;
	h_keys.reserve(total_length);
	for (u32 i = 0, key = 0; i < total_length; ++i) {
		if (i == seg_lengths[key]) ++key;
		h_keys.push_back(key);
	}
	
	auto d_in_keys = spp::device_alloc<u32>(h_keys.data(), h_keys.data() + total_length);
	auto d_out_keys = spp::device_alloc<u32>(seg_count);

	std::vector<data_t> h_data;
	h_data.reserve(total_length);
	for (u32 i = 0; i < total_length; ++i) h_data.push_back(dist_data(gen));
	
	auto d_in_data = spp::device_alloc<data_t>(h_data.data(), h_data.data() + total_length);

	auto d_out_data_truth = spp::device_alloc<data_t>(h_data.size());
	auto d_out_data_test = spp::device_alloc<data_t>(h_data.size());



	$with(spp::event start, stop) {
		start.record();

		thrust::inclusive_scan(
			thrust::device,
			d_in_data.get(), d_in_data.get() + total_length,
			d_out_data_truth.get()
		);

		stop.record();
		stop.synchronize();
		std::cout << "ground truth time = " << stop.elapsed_time_from(start) << std::endl;
	}

	$with(spp::event start, stop) {
		start.record();

		cudaCheck(spp::inclusive_scan(total_length, d_in_data.get(), d_out_data_test.get()));

		stop.record();
		stop.synchronize();
		std::cout << "test time = " << stop.elapsed_time_from(start) << std::endl;
	}



	// std::vector<u32> h_out_keys(seg_count);
	// cudaMemcpy(h_out_keys.data(), d_out_keys.get(), sizeof(u32) * seg_count, cudaMemcpyDeviceToHost);
	// std::copy(h_out_keys.begin(), h_out_keys.end(), std::ostream_iterator<u32>(std::cout, " "));
	// std::cout << std::endl;

	std::vector<data_t> h_out_data_truth(seg_count);
	cudaMemcpy(h_out_data_truth.data(), d_out_data_truth.get(), sizeof(data_t) * seg_count, cudaMemcpyDeviceToHost);
	std::copy(h_out_data_truth.begin(), h_out_data_truth.end(), std::ostream_iterator<data_t>(std::cout, " "));
	std::cout << std::endl;

	std::vector<data_t> h_out_data_test(seg_count);
	cudaMemcpy(h_out_data_test.data(), d_out_data_test.get(), sizeof(data_t) * seg_count, cudaMemcpyDeviceToHost);
	std::copy(h_out_data_test.begin(), h_out_data_test.end(), std::ostream_iterator<data_t>(std::cout, " "));
	std::cout << std::endl;



	return 0;
}