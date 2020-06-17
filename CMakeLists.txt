# CMakeList.txt : CMake project for batch_gemm, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

option(AURORA_BUILD "AURORA_BUILD")
option(USE_ARMCOMPUTE "AURORA_BUILD")

project ("minimal_nd"  VERSION ${VERSION} LANGUAGES C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON) 

list(APPEND EXTRAS ${CMAKE_CURRENT_SOURCE_DIR}/Environment.cpp)
list(APPEND EXTRAS ${CMAKE_CURRENT_SOURCE_DIR}/Threads.cpp) 

set(CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/")

SET(CMAKE_CXX_FLAGS  "-fopenmp")
if (AURORA_BUILD)
    message("Aurora build in process") 
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions  -fdiag-inline=2 -fdiag-parallel=2 -fdiag-vector=2 -DSD_AURORA_BUILD=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -DSD_AURORA_BUILD=true")
endif(AURORA_BUILD)

if(USE_ARMCOMPUTE)
     find_package(ARMCOMPUTE REQUIRED)

     if(ARMCOMPUTE_FOUND)
        message("Found ARMCOMPUTE: ${ARMCOMPUTE_LIBRARIES}")
        set(HAVE_ARMCOMPUTE 1)
        # Add preprocessor definition for ARM Compute NEON 
        add_definitions(-DARMCOMPUTENEON_ENABLED)
        #build our library with neon support
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
        include_directories(${ARMCOMPUTE_INCLUDE})
        message("----${ARMCOMPUTE_INCLUDE}---")
     endif()

else()
     set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp -mavx2")
endif()


    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g")

 

#add_executable (minimal_nd4 "main.cpp"  ${EXTRAS}   "user_codes4.cpp"  "reference.cpp")
#target_include_directories(minimal_nd4 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 


#set_source_files_properties(user_codes6.cpp PROPERTIES COMPILE_FLAGS  $<$<CONFIG:Release>:-ffast-math>) 
#add_executable (minimal_nd6 "main.cpp"  ${EXTRAS}   "user_codes6.cpp"  "reference.cpp")
#target_include_directories(minimal_nd6 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 

 

#add_executable (minimal_orig "main.cpp"  ${EXTRAS}   "original.cpp"  "reference.cpp")
#target_include_directories(minimal_orig PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )


add_executable (arm_examples "armcompute_main.cpp"  ${EXTRAS}    "armcomputeUtils.cpp" "armcomputeUtils.h")
target_include_directories(arm_examples PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ${ARMCOMPUTE_INCLUDE})
target_link_libraries(arm_examples  ${ARMCOMPUTE_LIBRARIES}  -latomic)