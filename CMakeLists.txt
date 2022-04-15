cmake_minimum_required(VERSION 3.0)

project(spp_test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD_REQUIRED true)
set(CMAKE_CUDA_STANDARD 17)

set(SPP_INCLUDE_DIR			${PROJECT_SOURCE_DIR}/include)
set(SPP_TEST_DIR			${PROJECT_SOURCE_DIR}/test)

add_executable(inclusive_scan
	${SPP_TEST_DIR}/inclusive_scan.cu
)

target_include_directories(inclusive_scan
	PRIVATE ${SPP_INCLUDE_DIR}
)

set_target_properties(inclusive_scan
	PROPERTIES CUDA_ARCHITECTURES OFF
)