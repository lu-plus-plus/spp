
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>

#include <cub/device/device_radix_sort.cuh>

#include "spp/types.hpp"
#include "spp/log.hpp"
#include "spp/event.hpp"
#include "spp/device_ptr.hpp"
#include "spp/radix_sort.hpp"



using spp::u32;

int main(void) {

	u32 constexpr item_count = 1u << 22;

	std::mt19937 randu_gen(std::random_device{}());
	std::uniform_int_distribution<u32> randu_dist(0, std::numeric_limits<u32>::max());
	auto randu = [&] () -> u32 { return randu_dist(randu_gen); };

	std::vector<u32> h_data_in(item_count);
	std::generate(h_data_in.begin(), h_data_in.end(), randu);

	spp::device_ptr<u32> d_data_in = spp::device_copy_from(h_data_in.data(), h_data_in.data() + item_count);

	spp::device_ptr<u32> d_data_out_cub = spp::device_alloc<u32>(item_count);
	spp::device_ptr<u32> d_data_out_spp = spp::device_alloc<u32>(item_count);

	with_(spp::event start, stop) {
		void * d_temp_storage = nullptr;
		size_t d_temp_storage_bytes = 0;

		cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_storage_bytes, d_data_in.get(), d_data_out_cub.get(), item_count);

		cudaCheck(cudaMalloc(&d_temp_storage, d_temp_storage_bytes));
		
		start.record();
		cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_storage_bytes, d_data_in.get(), d_data_out_cub.get(), item_count);
		stop.record();
		stop.synchronize();

		std::cout << "[cub] baseline time = " << stop.elapsed_time_from(start) << std::endl;

		cudaCheck(cudaFree(d_temp_storage));
	}

	with_(spp::event start, stop) {
		spp::device_ptr<spp::byte> temp_storage = nullptr;
		spp::u32 temp_storage_bytes = 0;

		cudaCheck(spp::kernel::radix_sort_by_key(d_data_in, d_data_out_spp, item_count, temp_storage, temp_storage_bytes));

		temp_storage = spp::device_alloc<spp::byte>(temp_storage_bytes);

		start.record();
		cudaCheck(spp::kernel::radix_sort_by_key(d_data_in, d_data_out_spp, item_count, temp_storage, temp_storage_bytes));
		stop.record();
		stop.synchronize();

		std::cout << "[spp] test time = " << stop.elapsed_time_from(start) << std::endl;
	}

	std::vector<u32> h_data_out_cub(item_count);
	cudaCheck(cudaMemcpy(h_data_out_cub.data(), d_data_out_cub.get(), sizeof(u32) * item_count, cudaMemcpyDeviceToHost));

	std::vector<u32> h_data_out_spp(item_count);
	cudaCheck(cudaMemcpy(h_data_out_spp.data(), d_data_out_spp.get(), sizeof(u32) * item_count, cudaMemcpyDeviceToHost));

	u32 error_count = 0;

	for (u32 i = 0; i < item_count; ++i) {
		u32 const truth = h_data_out_cub[i];
		u32 const test = h_data_out_spp[i];
		if (truth != test) {
			if (error_count == 0) std::cout << "element " << std::hex << i << ": ground truth = " << std::hex << truth << ", test = " << test << std::endl;
			++error_count;
		}
	}

	std::cout << "total length = " << item_count << std::endl;
	std::cout << "error count = " << error_count << std::endl;

}