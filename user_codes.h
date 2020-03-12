#pragma once
#include <NdArrayMinimal.h>
#include <LoopCoordsHelper.h>
#include <Threads.h>

template <typename T1, typename T2, typename T3>
 void batchedGemm3(const sd::NDArray* vA, const sd::NDArray* vB, sd::NDArray* vC,
    const T1 alpha, const T3 beta,char out_order);
 template <typename T1, typename T2, typename T3>
 void   ff (uint64_t uui, int64_t start, int64_t stop, int64_t increment, const sd::NDArray* vA, const sd::NDArray* vB, sd::NDArray* vC, const T1 alpha, const T3 beta);

 sd::NDArray* mmulMxM(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C, const double alpha, const double beta, const char outOrder);