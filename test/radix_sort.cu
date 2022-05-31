
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>

#include <cub/device/device_radix_sort.cuh>

#include "spp/radix_sort.hpp"
#include "spp/device_vector.hpp"
#include "spp/tester.hpp"
#include "spp/rng.hpp"



int main(void) {

	uint32_t constexpr item_count = 1u << 22;

	spp::rng<uint32_t> rng(0u, 2u);
	auto const d_data_in{ rng.seq(item_count) };

	spp::device_vector<uint32_t> d_cub_data_out{ item_count };
	spp::device_vector<uint32_t> d_spp_data_out{ item_count };

	spp::test::tester<size_t>(
		spp_test_functor_of(cub::DeviceRadixSort::SortKeys),
		d_data_in.data(), d_cub_data_out.data(), item_count
	).benchmark(
		spp_test_functor_of(cub::DeviceRadixSort::SortKeys),
		d_data_in.data(), d_cub_data_out.data(), item_count
	);

	spp::test::tester<uint32_t>(
		spp_test_functor_of(spp::kernel::radix_sort),
		d_data_in.data(), d_spp_data_out.data(), item_count
	).benchmark(
		spp_test_functor_of(spp::kernel::radix_sort),
		d_data_in.data(), d_spp_data_out.data(), item_count
	);

	auto const h_cub_data_out{ d_cub_data_out.to_host() };
	auto const h_spp_data_out{ d_spp_data_out.to_host() };
	
	spp::test::compare(h_cub_data_out, h_spp_data_out);

	return 0;

}