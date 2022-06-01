
#include <vector>

#include <cub/device/device_radix_sort.cuh>

#include "spp/radix_sort.hpp"
#include "spp/device_vector.hpp"
#include "spp/tester.hpp"
#include "spp/rng.hpp"



int main(void) {

	uint32_t constexpr item_count = 1u << 22;

	auto const d_keys_in{ spp::rng<uint32_t>(0u, 2u).seq(item_count) };
	auto const d_values_in{ spp::rng<float>(0.0f, 1024.0f).seq(item_count) };

	spp::device_vector<uint32_t> d_cub_keys_out{ item_count };
	spp::device_vector<float> d_cub_values_out{ item_count };

	spp::device_vector<uint32_t> d_spp_keys_out{ item_count };
	spp::device_vector<float> d_spp_values_out{ item_count };

	spp::test::tester<size_t>(
		spp_test_functor_of(cub::DeviceRadixSort::SortPairs),
		d_keys_in.data(), d_cub_keys_out.data(),
		d_values_in.data(), d_cub_values_out.data(),
		item_count
	).benchmark(
		spp_test_functor_of(cub::DeviceRadixSort::SortPairs),
		d_keys_in.data(), d_cub_keys_out.data(),
		d_values_in.data(), d_cub_values_out.data(),
		item_count
	);

	spp::test::tester<uint32_t>(
		spp_test_functor_of(spp::kernel::radix_sort_by_key),
		d_keys_in.data(), d_spp_keys_out.data(),
		d_values_in.data(), d_spp_values_out.data(),
		item_count
	).benchmark(
		spp_test_functor_of(spp::kernel::radix_sort_by_key),
		d_keys_in.data(), d_spp_keys_out.data(),
		d_values_in.data(), d_spp_values_out.data(),
		item_count
	);

	auto const h_cub_keys_out{ d_spp_keys_out.to_host() };
	auto const h_spp_keys_out{ d_spp_keys_out.to_host() };
	spp::test::compare(h_cub_keys_out, h_spp_keys_out);

	auto const h_cub_values_out{ d_cub_values_out.to_host() };
	auto const h_spp_values_out{ d_spp_values_out.to_host() };
	spp::test::compare(h_cub_values_out, h_spp_values_out);

	return 0;

}