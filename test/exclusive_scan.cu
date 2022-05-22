
#include <vector>
#include <random>
#include <numeric>
#include <iostream>
#include <iomanip>

#include "cub/device/device_scan.cuh"

#include "spp/types.hpp"
#include "spp/device_vector.hpp"
#include "spp/tester.hpp"
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

	auto d_data_in = spp::device_vector<data_t>(h_data_in);

	auto d_data_out_cub = spp::device_vector<data_t>(item_count);
	auto d_data_out_spp = spp::device_vector<data_t>(item_count);

	spp::test::tester<size_t>(
		spp_test_functor_of(cub::DeviceScan::ExclusiveSum),
		d_data_in.data(), d_data_out_cub.data(), item_count
	).benchmark(
		spp_test_functor_of(cub::DeviceScan::ExclusiveSum),
		d_data_in.data(), d_data_out_cub.data(), item_count
	);

	spp::test::tester<spp::usize>(
		spp_test_functor_of(spp::kernel::exclusive_scan),
		d_data_in.data(), d_data_out_spp.data(), item_count
	).benchmark(
		spp_test_functor_of(spp::kernel::exclusive_scan),
		d_data_in.data(), d_data_out_spp.data(), item_count
	);

	auto h_data_out_cub = d_data_out_cub.to_host();
	auto h_data_out_spp = d_data_out_spp.to_host();

	spp::test::compare(h_data_out_cub, h_data_out_spp);

	return 0;
}