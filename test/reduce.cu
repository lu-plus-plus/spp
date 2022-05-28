
#include <cstdint>

#include <cub/device/device_reduce.cuh>

#include "spp/reduce.hpp"
#include "spp/device_vector.hpp"
#include "spp/rng.hpp"
#include "spp/tester.hpp"



int main(void) {

	uint32_t constexpr test_size{ 1u << 22 };

	spp::rng<uint32_t> rng(0u, 2u);
	auto const d_data_in{ rng.seq(test_size) };
	
	spp::device_vector<uint32_t> d_cub_data_out{ 1 };
	spp::device_vector<uint32_t> d_spp_data_out{ 1 };

	spp::test::tester<size_t>(
		spp_test_functor_of(cub::DeviceReduce::Sum),
		d_data_in.data(), d_cub_data_out.data(), test_size
	).benchmark(
		spp_test_functor_of(cub::DeviceReduce::Sum),
		d_data_in.data(), d_cub_data_out.data(), test_size	
	);

	spp::test::tester<uint32_t>(
		spp_test_functor_of(spp::kernel::reduce),
		d_data_in.data(), d_spp_data_out.data(), test_size
	).benchmark(
		spp_test_functor_of(spp::kernel::reduce),
		d_data_in.data(), d_spp_data_out.data(), test_size
	);

	auto const h_cub_data_out{ d_cub_data_out.to_host() };
	auto const h_spp_data_out{ d_spp_data_out.to_host() };

	if (spp::test::compare(h_cub_data_out, h_spp_data_out)) {
		std::cout << "[info] correct reduction result: " << h_spp_data_out.front() << "\n";
	}

	return 0;

}