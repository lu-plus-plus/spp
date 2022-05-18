
#include <vector>
#include <random>
#include <numeric>
#include <iostream>
#include <iterator>

#include "cub/device/device_scan.cuh"

#include "spp/types.hpp"
#include "spp/log.hpp"
#include "spp/event.hpp"
#include "spp/device_ptr.hpp"
#include "spp/scan.hpp"



int main(void) {
	using spp::u32;
	using data_t = spp::u32;

	using index_dist_t = std::uniform_int_distribution<u32>;
	using data_dist_t = std::uniform_int_distribution<data_t>;

	constexpr u32 seg_count = 1024 * 32;

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

	auto d_out_data_truth = spp::device_alloc<data_t>(total_length);
	auto d_out_data_test = spp::device_alloc<data_t>(total_length);



	with_(spp::event start, stop) {

		void * temp_storage = nullptr;
		size_t temp_storage_bytes = 0;

		cub::DeviceScan::InclusiveSum(
			temp_storage, temp_storage_bytes,
			d_in_data.get(), d_out_data_truth.get(), total_length
		);

		cudaCheck(cudaMalloc(&temp_storage, temp_storage_bytes));

		start.record();

		cub::DeviceScan::InclusiveSum(
			temp_storage, temp_storage_bytes,
			d_in_data.get(), d_out_data_truth.get(), total_length
		);

		stop.record();
		stop.synchronize();

		std::cout << "[cub] baseline time = " << stop.elapsed_time_from(start) << std::endl;

		cudaCheck(cudaFree(temp_storage));
	}

	with_(spp::event start, stop) {

		spp::device_ptr<spp::byte> temp_storage = nullptr;
		spp::u32 temp_storage_bytes = 0;

		cudaCheck(spp::kernel::inclusive_scan<data_t>(
			d_in_data, d_out_data_test, total_length,
			temp_storage, temp_storage_bytes
		));

		temp_storage = spp::device_alloc<spp::byte>(temp_storage_bytes);

		start.record();

		cudaCheck(spp::kernel::inclusive_scan<data_t>(
			d_in_data, d_out_data_test, total_length,
			temp_storage, temp_storage_bytes
		));

		stop.record();
		stop.synchronize();

		std::cout << "[spp] test time = " << stop.elapsed_time_from(start) << std::endl;
	}



	std::vector<data_t> h_out_data_truth(total_length);
	cudaMemcpy(h_out_data_truth.data(), d_out_data_truth.get(), sizeof(data_t) * total_length, cudaMemcpyDeviceToHost);

	std::vector<data_t> h_out_data_test(total_length);
	cudaMemcpy(h_out_data_test.data(), d_out_data_test.get(), sizeof(data_t) * total_length, cudaMemcpyDeviceToHost);

	// std::copy(h_out_data_truth.begin(), h_out_data_truth.end(), std::ostream_iterator<data_t>(std::cout, " "));
	// std::cout << std::endl;
	// std::copy(h_out_data_test.begin(), h_out_data_test.end(), std::ostream_iterator<data_t>(std::cout, " "));
	// std::cout << std::endl;

	u32 error_count = 0;

	for (u32 i = 0; i < total_length; ++i) {
		data_t const truth = h_out_data_truth[i];
		data_t const test = h_out_data_test[i];
		if (truth != test) {
			if (error_count == 0) std::cout << "element " << i << ": ground truth = " << truth << ", test = " << test << std::endl;
			++error_count;
		}
	}

	std::cout << "total length = " << total_length << std::endl;
	std::cout << "error count = " << error_count << std::endl;



	return 0;
}