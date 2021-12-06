# CMakeList.txt : CMake project for batch_gemm, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.15)

option(AURORA_BUILD "AURORA_BUILD")
option(USE_ARMCOMPUTE "AURORA_BUILD")
set(VERSION_TXT 0.01)
project ("minimal_nd"  VERSION ${VERSION_TXT} LANGUAGES C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON) 
set (CMAKE_CXX_STANDARD 14)

option(SD_NEW  "new checks" ON)

list(APPEND EXTRAS ${CMAKE_CURRENT_SOURCE_DIR}/Environment.cpp)
list(APPEND EXTRAS ${CMAKE_CURRENT_SOURCE_DIR}/Threads.cpp) 

set(CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/")

SET(CMAKE_CXX_FLAGS  "-fopenmp")
if (AURORA_BUILD)
    message("Aurora build in process") 
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions  -fdiag-inline=2 -fdiag-parallel=2 -fdiag-vector=2 -DSD_AURORA_BUILD=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -DSD_AURORA_BUILD=true")
endif(AURORA_BUILD)



if(SD_NEW)

    
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp -mavx2 ")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g3")
    message("Adds Vednn into dependency projects using ExternalProject_Add")
    set(HAVE_VEDNN 1)
    include(ExternalProject)
    get_filename_component(MY_RESULT_ABSOLUTE_PATH_VAR
                       "${CMAKE_CURRENT_BINARY_DIR}/vednn-install"
                       ABSOLUTE)
    set(VEDNN_PREFIX "${MY_RESULT_ABSOLUTE_PATH_VAR}")
    message(${VEDNN_PREFIX})
    set(VEDNN_STATIC_LIB
        "${VEDNN_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}dnnl${CMAKE_STATIC_LIBRARY_SUFFIX}")
    message(${VEDNN_STATIC_LIB})
    if(SD_AURORA)
        set (BASH_CMD_VE  bash build.sh -aq$<$<CONFIG:Debug>:dddd>)
    else()
        set (BASH_CMD_VE  bash build.sh -vjq$<$<CONFIG:Debug>:dddd>)
    endif()
    set(VEDNN_BUILD_COMMAND NLC_HOME=$ENV{NLC_ROOT} VEDNN_INSTALL_DIR=${VEDNN_PREFIX} ${BASH_CMD_VE})
    set(VEDNN_PATCH_COMMAND "patch" "-Np0" < ${CMAKE_SOURCE_DIR}/vednn.patch "||" true)
    ExternalProject_Add(vednn_project
        GIT_REPOSITORY    https://github.com/necla-ml/gen-dnn
        GIT_TAG           ve-v1.4
        UPDATE_DISCONNECTED 1
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/vednn-src"
        PATCH_COMMAND     ${VEDNN_PATCH_COMMAND}
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE   1
        BUILD_COMMAND     ${VEDNN_BUILD_COMMAND}
        BUILD_BYPRODUCTS  ${VEDNN_STATIC_LIB}
        INSTALL_COMMAND cmake -E echo "Skipping install step as its already inside build."
        )
 
    set(VEDNN_INCLUDE "${VEDNN_PREFIX}/include")
    list(APPEND EXTERNAL_DEPENDENCY_PROJECTS vednn_project)
    list(APPEND EXTERNAL_DEPENDENCY_LIBS  ${VEDNN_STATIC_LIB} )
    include_directories((${VEDNN_INCLUDE}))
 
    add_executable (vednn_check "vednn_new.cpp"  ${EXTRAS}  )
    message( ${EXTERNAL_DEPENDENCY_LIBS})
    target_include_directories(vednn_check PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )
    foreach(external_dependency ${EXTERNAL_DEPENDENCY_PROJECTS})
    message("@external ${external_dependency}")
    add_dependencies(vednn_check ${external_dependency})
   endforeach()
    target_link_libraries(vednn_check ${EXTERNAL_DEPENDENCY_LIBS})
 

else(SD_NEW)


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
else(USE_ARMCOMPUTE)
     set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp -mavx2 -std=c++14 ")
endif(USE_ARMCOMPUTE)


    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g3 -std=c++14")
 

#add_executable (minimal_nd4 "main.cpp"  ${EXTRAS}   "user_codes4.cpp"  "reference.cpp")
#target_include_directories(minimal_nd4 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 


#set_source_files_properties(user_codes6.cpp PROPERTIES COMPILE_FLAGS  $<$<CONFIG:Release>:-ffast-math>) 
#add_executable (minimal_nd6 "main.cpp"  ${EXTRAS}   "user_codes6.cpp"  "reference.cpp")
#target_include_directories(minimal_nd6 PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 

 

#add_executable (minimal_orig "main.cpp"  ${EXTRAS}   "original.cpp"  "reference.cpp")
#target_include_directories(minimal_orig PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )

if(USE_ARMCOMPUTE)



#add_executable (arm_examples "armcompute_main.cpp"  ${EXTRAS}    "armcomputeUtils.cpp" "armcomputeUtils.h")
#target_include_directories(arm_examples PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ${ARMCOMPUTE_INCLUDE})
#target_link_libraries(arm_examples  ${ARMCOMPUTE_LIBRARIES}  -latomic)

add_executable (arm_plus "armcompute_new.cpp"  ${EXTRAS}    "armcomputeUtils.cpp" "armcomputeUtils.h")
target_include_directories(arm_plus PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ${ARMCOMPUTE_INCLUDE})
target_link_libraries(arm_plus  ${ARMCOMPUTE_LIBRARIES}  -latomic)
else(USE_ARMCOMPUTE)

add_executable (minimal_ctc "ctc_loss.cpp"  ${EXTRAS}    "vednn_new.cpp")
target_include_directories(minimal_ctc PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 

add_executable (minimal_bm ${EXTRAS}   "beamsearch.cpp")
target_include_directories(minimal_bm PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 
add_executable (minimal_extra    "heap.cpp")
target_include_directories(minimal_bm PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 

add_executable (minimal_new    "new_additions.cpp")
target_include_directories(minimal_new PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> ) 

add_library (minimalbm SHARED ${EXTRAS}   "beamsearch.cpp")
target_include_directories(minimalbm PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> )  
endif(USE_ARMCOMPUTE)
 

endif(SD_NEW)