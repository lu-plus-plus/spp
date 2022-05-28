#ifndef SPP_PIPELINED_FOR_HPP
#define SPP_PIPELINED_FOR_HPP

#include <cstdint>
#include <utility>



namespace spp::device {

	template <uint32_t Size, uint32_t Step, typename Load, typename Process>
	__device__
	void pipelined_for(Load && load, Process && process) {
		uint32_t constexpr Iterations = Size / Step;

		for (uint32_t i_load = 0; i_load < Step; ++i_load) {
			std::forward<Load>(load)(i_load);
		}

		for (uint32_t i_iter = 0; i_iter < Iterations; ++i_iter) {
			if (i_iter + 1 != Iterations) {
				for (uint32_t j_load = (i_iter + 1) * Step; j_load < (i_iter + 2) * Step; ++j_load) {
					std::forward<Load>(load)(j_load);
				}
			}

			for (uint32_t j_process = i_iter * Step; j_process < (i_iter + 1) * Step; ++j_process) {
				std::forward<Process>(process)(j_process);
			}
		}
	}

}



#endif // SPP_PIPELINED_FOR_HPP