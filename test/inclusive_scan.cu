
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



using spp::usize;

using data_t = spp::u32;
using data_dist_t = std::uniform_int_distribution<data_t>;



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

	with_(spp::event start, stop) {

		void * d_temp_storage = nullptr;
		size_t d_temp_storage_bytes = 0;

		cudaCheck(cub::DeviceScan::InclusiveSum(
			d_temp_storage, d_temp_storage_bytes,
			d_data_in.get(), d_data_out_cub.get(), item_count
		));

		cudaCheck(cudaMalloc(&d_temp_storage, d_temp_storage_bytes));

		start.record();
		cudaCheck(cub::DeviceScan::InclusiveSum(
			d_temp_storage, d_temp_storage_bytes,
			d_data_in.get(), d_data_out_cub.get(), item_count
		));
		stop.record();
		stop.synchronize();

		std::cout << "[cub] baseline time = " << stop.elapsed_time_from(start) << std::endl;

		cudaCheck(cudaFree(d_temp_storage));
	}

	with_(spp::event start, stop) {

		spp::device_ptr<spp::byte> d_temp_storage = nullptr;
		spp::usize d_temp_storage_bytes = 0;

		cudaCheck(spp::kernel::inclusive_scan(
			d_temp_storage, d_temp_storage_bytes,
			d_data_in.get(), d_data_out_spp.get(), item_count
		));

		d_temp_storage = spp::device_alloc<spp::byte>(d_temp_storage_bytes);

		start.record();
		cudaCheck(spp::kernel::inclusive_scan(
			d_temp_storage, d_temp_storage_bytes,
			d_data_in.get(), d_data_out_spp.get(), item_count
		));
		stop.record();
		stop.synchronize();

		std::cout << "[spp] test time = " << stop.elapsed_time_from(start) << std::endl;
	}



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