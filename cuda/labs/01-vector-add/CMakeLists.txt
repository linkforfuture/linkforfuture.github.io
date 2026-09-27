cmake_minimum_required(VERSION 3.24)

# Detect the visible GPU by default; allow -DCMAKE_CUDA_ARCHITECTURES=80 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_vector_add LANGUAGES CXX CUDA)
add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
