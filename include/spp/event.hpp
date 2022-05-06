#ifndef SPP_EVENT_HPP
#define SPP_EVENT_HPP

#include "types.hpp"
#include "log.hpp"



namespace spp {

	class event {

		cudaEvent_t m_event;

	public:

		event() : m_event() { cudaCheck(cudaEventCreate(&m_event)); }

		~event() { cudaCheck(cudaEventDestroy(m_event)); }

		event(event const &) = delete;
		event operator=(event const &) = delete;
		event(event &&) = delete;
		event operator=(event &&) = delete;

		operator cudaEvent_t () const { return m_event; }

		void record() const { cudaCheck(cudaEventRecord(m_event)); }

		void synchronize() const { cudaCheck(cudaEventSynchronize(m_event)); }

		float elapsed_time_from(cudaEvent_t start) const {
			float ms = 0.0f;
			cudaCheck(cudaEventElapsedTime(&ms, start, m_event));
			return ms;
		}

	};

} // namespace spp



#endif // SPP_EVENT_HPP