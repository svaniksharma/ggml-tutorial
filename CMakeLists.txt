cmake_minimum_required(VERSION 3.15..4.0)
project(tutorial VERSION 0.1 DESCRIPTION "A little GGML tutorial" LANGUAGES CXX)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()

add_executable(tutorial src/tutorial.cpp)
target_include_directories(tutorial PUBLIC include extern/ggml/include)
target_compile_features(tutorial PUBLIC cxx_std_17)
set(cxx_flags
  -Wpedantic
  -Wunused-variable
  -Wno-unused-function
  -Wno-multichar
  -g
)
target_compile_options(tutorial PUBLIC ${cxx_flags})

if (EMSCRIPTEN)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  set_target_properties(tutorial PROPERTIES LINK_FLAGS "-s ASSERTIONS=1 -s ALLOW_MEMORY_GROWTH=1 -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE_STRICT=0 -s SINGLE_FILE")
endif()

add_subdirectory(extern/ggml)
target_link_libraries(tutorial PRIVATE ggml)


if (GGML_CUDA)
  add_compile_definitions(GGML_USE_CUDA)
endif()
