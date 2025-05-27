cmake_minimum_required(VERSION 3.15..4.0)
project(tutorial VERSION 0.1 DESCRIPTION "A little GGML tutorial" LANGUAGES CXX)

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
add_subdirectory(extern/ggml)
target_link_libraries(tutorial PRIVATE ggml)
