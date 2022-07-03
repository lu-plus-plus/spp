#ifndef SPP_BARRIER_HPP
#define SPP_BARRIER_HPP

#include <cstdint>



namespace spp {

	#define ptx_barrier_arrive(id, count) asm volatile ("barrier.arrive %0, %1;" : : "n"(id), "n"(count))
	#define ptx_barrier_sync(id, count) asm volatile ("barrier.sync %0, %1;" : : "n"(id), "n"(count))

	template <uint32_t Id, uint32_t WarpCount>
	struct barrier {
		__device__
		void wait() const noexcept {
			ptx_barrier_sync(Id, WarpCount * 32);
		}

		__device__
		void arrive() const noexcept {
			ptx_barrier_arrive(Id, WarpCount * 32);
		}
	};

}



#endif // SPP_BARRIER_HPP