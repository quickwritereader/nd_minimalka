# CMakeList.txt : CMake project for batch_gemm, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

option(AURORA_BUILD "AURORA_BUILD")

set(CMAKE_VERBOSE_MAKEFILE ON) 
list(APPEND EXTRAS ${CMAKE_CURRENT_SOURCE_DIR}/Environment.cpp)
list(APPEND EXTRAS ${CMAKE_CURRENT_SOURCE_DIR}/Threads.cpp) 
SET(CMAKE_CXX_FLAGS  "-fopenmp")
if (AURORA_BUILD)
    message("Aurora build in process") 
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions  -fdiag-inline=2 -fdiag-parallel=2 -fdiag-vector=2 -DSD_AURORA_BUILD=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -DSD_AURORA_BUILD=true")
endif(AURORA_BUILD)

    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")
project ("minimal_nd"  VERSION ${VERSION} LANGUAGES C CXX)
add_executable (minimal_nd "main.cpp" "user_codes.cpp"  ${EXTRAS}   "reference.cpp")
target_include_directories(minimal_nd PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )
  
add_executable (minimal_nd2 "main.cpp" "user_codes2.cpp"  ${EXTRAS} "reference.cpp"   )
target_include_directories(minimal_nd2 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )
# TODO: Add tests and install targets if needed.
add_executable (minimal_nd3 "main.cpp"  ${EXTRAS}   "user_codes3.cpp"  "reference.cpp")
target_include_directories(minimal_nd3 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )

add_executable (minimal_nd5 "main.cpp"  ${EXTRAS}   "user_codes5.cpp"  "reference.cpp")
target_include_directories(minimal_nd5 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )

add_executable (minimal_orig "main.cpp"  ${EXTRAS}   "original.cpp"  "reference.cpp")
target_include_directories(minimal_orig PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )