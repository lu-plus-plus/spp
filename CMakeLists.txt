cmake_minimum_required(VERSION 3.0)

project(spp_test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD_REQUIRED true)
set(CMAKE_CUDA_STANDARD 17)

set(SPP_INCLUDE_DIR			${PROJECT_SOURCE_DIR}/include)
set(SPP_TEST_DIR			${PROJECT_SOURCE_DIR}/test)



function(single_file_test filename_wo_ext)
	add_executable(${filename_wo_ext}
		${SPP_TEST_DIR}/${filename_wo_ext}.cu
	)

	target_include_directories(${filename_wo_ext}
		PRIVATE ${SPP_INCLUDE_DIR}
	)

	set_target_properties(${filename_wo_ext} PROPERTIES
		CUDA_ARCHITECTURES OFF
		CUDA_SEPARABLE_COMPILATION ON
	)
endfunction()



single_file_test(inclusive_scan)
single_file_test(exclusive_scan)
single_file_test(radix_sort)
single_file_test(reduce)


