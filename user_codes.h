#pragma once
#include <NdArrayMinimal.h>
#include <LoopCoordsHelper.h>
#include <Threads.h>

template <typename T1, typename T2, typename T3>
 void batchedGemm3(const sd::NDArray* vA, const sd::NDArray* vB, sd::NDArray* vC,
    const T1 alpha, const T3 beta,char out_order ,const bool transA=false, const bool transB=false);
 
 
 sd::NDArray* mmulMxM(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C, const double alpha, const double beta, const char outOrder, const bool transA = false, const bool transB = false);