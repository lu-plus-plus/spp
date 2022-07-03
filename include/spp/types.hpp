#ifndef SPP_TYPES_HPP
#define SPP_TYPES_HPP

#include <cstdint>



namespace spp {

	using i8	= int8_t;
	using i16	= int16_t;
	using i32	= int32_t;
	using i64	= int64_t;

	using u8	= uint8_t;
	using u16	= uint16_t;
	using u32	= uint32_t;
	using u64	= uint64_t;

	using f32	= float;
	using f64	= double;

	using usize	= u32;
	using isize = i32;



	#define user_defined_literal(ret_type, suffix, literal_type)	\
		inline constexpr											\
		__host__ __device__											\
		ret_type operator "" _ ## suffix(literal_type literal) {	\
			return static_cast<ret_type>(literal);					\
		}

	user_defined_literal(i8,	i8,		unsigned long long int)
	user_defined_literal(i16,	i16,	unsigned long long int)
	user_defined_literal(i32,	i32,	unsigned long long int)
	user_defined_literal(i64,	i64,	unsigned long long int)

	user_defined_literal(u8,	u8,		unsigned long long int)
	user_defined_literal(u16,	u16,	unsigned long long int)
	user_defined_literal(u32,	u32,	unsigned long long int)
	user_defined_literal(u64,	u64,	unsigned long long int)

	user_defined_literal(f32,	f32,	long double)
	user_defined_literal(f64,	f64,	long double)

	user_defined_literal(usize,	us,		unsigned long long int)
	user_defined_literal(isize, is,		unsigned long long int)

	#undef user_defined_literal

} // namespace spp



#endif // SPP_TYPES_HPP