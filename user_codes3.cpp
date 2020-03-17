#include <user_codes.h>
#include <algorithm>
using namespace sd;

template <typename T1, typename T2, typename T3>
static void inner_gemm_no_checks(const Nd4jLong M, const Nd4jLong N, const Nd4jLong K, const T3 alpha, const T1* __restrict A, const Nd4jLong aStride_M, const Nd4jLong aStride_K, const T2* __restrict B, Nd4jLong bStride_K, Nd4jLong bStride_N, T3* __restrict C, const Nd4jLong cStride_M) {
#if 0
    nd4j_printf(" M %ld   , N %ld  , K %ld , alpha %lf, aStride_M %ld  , aStride_K %ld,    bStride_K %ld  bStride_N %ld,   cStride_M %ld   \n",
        M, N, K, alpha, aStride_M, aStride_K, bStride_K, bStride_N, cStride_M);
#endif
    Nd4jLong M_L = M & -4;;
    Nd4jLong K_L = K & -4;
    // Nd4jLong aStride_Mx2 = aStride_M *2;
    if (aStride_K == 1 && bStride_N == 1) {
        for (Nd4jLong k = 0; k < K_L; k += 4) {

            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T2* __restrict BB1 = &(B[k * bStride_K + bStride_K]);
            const T2* __restrict BB2 = &(B[k * bStride_K + 2 * bStride_K]);
            const T2* __restrict BB3 = &(B[k * bStride_K + 3 * bStride_K]);

            const T1* __restrict AA = &(A[k * 1]);
            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA0_1 = alpha * AA[m * aStride_M + 1];
                T1 AA0_2 = alpha * AA[m * aStride_M + 2 * 1];
                T1 AA0_3 = alpha * AA[m * aStride_M + 3 * 1];

                T1 AA1 = alpha * AA[m * aStride_M + aStride_M];
                T1 AA1_1 = alpha * AA[m * aStride_M + aStride_M + 1];
                T1 AA1_2 = alpha * AA[m * aStride_M + aStride_M + 2 * 1];
                T1 AA1_3 = alpha * AA[m * aStride_M + aStride_M + 3 * 1];

                T1 AA2 = alpha * AA[m * aStride_M + 2 * aStride_M];
                T1 AA2_1 = alpha * AA[m * aStride_M + 2 * aStride_M + 1];
                T1 AA2_2 = alpha * AA[m * aStride_M + 2 * aStride_M + 2 * 1];
                T1 AA2_3 = alpha * AA[m * aStride_M + 2 * aStride_M + 3 * 1];

                T1 AA3 = alpha * AA[m * aStride_M + 3 * aStride_M];
                T1 AA3_1 = alpha * AA[m * aStride_M + 3 * aStride_M + 1];
                T1 AA3_2 = alpha * AA[m * aStride_M + 3 * aStride_M + 2 * 1];
                T1 AA3_3 = alpha * AA[m * aStride_M + 3 * aStride_M + 3 * 1];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n]);
                        CC0[n] += AA0 * BB0[n] + AA0_1 * BB1[n] + AA0_2 * BB2[n] + AA0_3 * BB3[n];
                        CC1[n] += AA1 * BB0[n] + AA1_1 * BB1[n] + AA1_2 * BB2[n] + AA1_3 * BB3[n];
                        CC2[n] += AA2 * BB0[n] + AA2_1 * BB1[n] + AA2_2 * BB2[n] + AA2_3 * BB3[n];
                        CC3[n] += AA3 * BB0[n] + AA3_1 * BB1[n] + AA3_2 * BB2[n] + AA3_3 * BB3[n];
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA0_1 = alpha * AA[m * aStride_M + 1];
                T1 AA0_2 = alpha * AA[m * aStride_M + 2 * 1];
                T1 AA0_3 = alpha * AA[m * aStride_M + 3 * 1];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {

                        CC[n] += AA0 * BB0[n] + AA0_1 * BB1[n] + AA0_2 * BB2[n] + AA0_3 * BB3[n];
                    }//N
            }//M
        }//K
        for (Nd4jLong k = K_L; k < K; k++) {
            const T2* __restrict BB = &(B[k * bStride_K]);
            const T1* __restrict AA = &(A[k * 1]);

            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA1 = alpha * AA[m * aStride_M + aStride_M];
                T1 AA2 = alpha * AA[m * aStride_M + 2 * aStride_M];
                T1 AA3 = alpha * AA[m * aStride_M + 3 * aStride_M];
                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n]);
                        CC0[n] += AA0 * BB[n];
                        CC1[n] += AA1 * BB[n];
                        CC2[n] += AA2 * BB[n];
                        CC3[n] += AA3 * BB[n];
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {
                T1 AA0 = alpha * AA[m * aStride_M];
                T3* __restrict CC = &(C[m * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n]);
                        CC[n] += AA0 * BB[n];
                    }//N
            }//M
        }//K
    }
    else {
        for (Nd4jLong k = 0; k < K_L; k += 4) {

            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T2* __restrict BB1 = &(B[k * bStride_K + bStride_K]);
            const T2* __restrict BB2 = &(B[k * bStride_K + 2 * bStride_K]);
            const T2* __restrict BB3 = &(B[k * bStride_K + 3 * bStride_K]);

            const T1* __restrict AA = &(A[k * aStride_K]);
            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA0_1 = alpha * AA[m * aStride_M + aStride_K];
                T1 AA0_2 = alpha * AA[m * aStride_M + 2 * aStride_K];
                T1 AA0_3 = alpha * AA[m * aStride_M + 3 * aStride_K];

                T1 AA1 = alpha * AA[m * aStride_M + aStride_M];
                T1 AA1_1 = alpha * AA[m * aStride_M + aStride_M + aStride_K];
                T1 AA1_2 = alpha * AA[m * aStride_M + aStride_M + 2 * aStride_K];
                T1 AA1_3 = alpha * AA[m * aStride_M + aStride_M + 3 * aStride_K];

                T1 AA2 = alpha * AA[m * aStride_M + 2 * aStride_M];
                T1 AA2_1 = alpha * AA[m * aStride_M + 2 * aStride_M + aStride_K];
                T1 AA2_2 = alpha * AA[m * aStride_M + 2 * aStride_M + 2 * aStride_K];
                T1 AA2_3 = alpha * AA[m * aStride_M + 2 * aStride_M + 3 * aStride_K];

                T1 AA3 = alpha * AA[m * aStride_M + 3 * aStride_M];
                T1 AA3_1 = alpha * AA[m * aStride_M + 3 * aStride_M + aStride_K];
                T1 AA3_2 = alpha * AA[m * aStride_M + 3 * aStride_M + 2 * aStride_K];
                T1 AA3_3 = alpha * AA[m * aStride_M + 3 * aStride_M + 3 * aStride_K];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC0[n] += AA0 * BB0[n * bStride_N] + AA0_1 * BB1[n * bStride_N] + AA0_2 * BB2[n * bStride_N] + AA0_3 * BB3[n * bStride_N];
                        CC1[n] += AA1 * BB0[n * bStride_N] + AA1_1 * BB1[n * bStride_N] + AA1_2 * BB2[n * bStride_N] + AA1_3 * BB3[n * bStride_N];
                        CC2[n] += AA2 * BB0[n * bStride_N] + AA2_1 * BB1[n * bStride_N] + AA2_2 * BB2[n * bStride_N] + AA2_3 * BB3[n * bStride_N];
                        CC3[n] += AA3 * BB0[n * bStride_N] + AA3_1 * BB1[n * bStride_N] + AA3_2 * BB2[n * bStride_N] + AA3_3 * BB3[n * bStride_N];
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA0_1 = alpha * AA[m * aStride_M + aStride_K];
                T1 AA0_2 = alpha * AA[m * aStride_M + 2 * aStride_K];
                T1 AA0_3 = alpha * AA[m * aStride_M + 3 * aStride_K];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC[n] += AA0 * BB0[n * bStride_N] + AA0_1 * BB1[n * bStride_N] + AA0_2 * BB2[n * bStride_N] + AA0_3 * BB3[n * bStride_N];
                    }//N
            }//M
        }//K
        for (Nd4jLong k = K_L; k < K; k++) {
            const T2* __restrict BB = &(B[k * bStride_K]);
            const T1* __restrict AA = &(A[k * aStride_K]);

            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA1 = alpha * AA[m * aStride_M + aStride_M];
                T1 AA2 = alpha * AA[m * aStride_M + 2 * aStride_M];
                T1 AA3 = alpha * AA[m * aStride_M + 3 * aStride_M];
                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC0[n] += AA0 * BB[n * bStride_N];
                        CC1[n] += AA1 * BB[n * bStride_N];
                        CC2[n] += AA2 * BB[n * bStride_N];
                        CC3[n] += AA3 * BB[n * bStride_N];
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {
                T1 AA0 = alpha * AA[m * aStride_M];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC[n] += AA0 * BB[n * bStride_N];
                    }//N
            }//M
        }//K

    }
}


template <typename T1, typename T2, typename T3>
static void parallel_batchedGemm3(const NDArray* vA, const NDArray* vB, NDArray* vC, 
    const double alpha, const double beta, Nd4jLong start, Nd4jLong stop) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
    T3* C = vC->bufferAsT<T3>();


    Nd4jLong zero_strides[MAX_RANK] = {}; //zero strides


    const T3 alphaZ = alpha;
    const T3 betaZ = beta;

    const bool betaPersent = beta;


    const Nd4jLong* cShapeInfo = vC->getShapeInfo();

    const Nd4jLong* bases = &(cShapeInfo[1]);

    Nd4jLong* aStrides = vA->stridesOf();
    Nd4jLong* bStrides = vB->stridesOf();
    const Nd4jLong* cStrides = vC->stridesOf();

    const char output_order = vC->ordering();

    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();
    if (aRank == 2) {
        aStrides = (Nd4jLong*)&zero_strides;
    }
    if (bRank == 2) {
        bStrides = (Nd4jLong*)&zero_strides;
    }

    int max_rank = aRank > bRank ? aRank : bRank;
    max_rank = max_rank > cRank ? max_rank : cRank;

    //Nd4jLong batch_len = 1;
    //for (int i = 0; i < max_rank - 2; i++) {
    //    batch_len *= bases[i];
    //}

    const int M = vA->sizeAt(aRank - 2);
    const int K = vA->sizeAt(aRank - 1);
    const int N = vC->sizeAt(cRank - 1);
    Nd4jLong aStride_M = aStrides[aRank - 2];
    Nd4jLong aStride_K = aStrides[aRank - 1];
    Nd4jLong bStride_K = bStrides[bRank - 2];
    Nd4jLong bStride_N = bStrides[bRank - 1];
    Nd4jLong cStride_M = cStrides[cRank - 2];
    Nd4jLong cStride_N = cStrides[cRank - 1];

    Nd4jLong coords[MAX_RANK] = {};
    Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
    sd::index2coords_C(start, max_rank - 2, bases, ptr_coords);
    //offset
    sd::triple_size_t offset = sd::offset_from_coords(aStrides, bStrides, cStrides, ptr_coords, max_rank - 2);
#if 0
    nd4j_printf("start:%d stop:%d \na: %d b:%d v:%d    \n", start, stop, offset.first, offset.second, offset.third);
#endif
    Nd4jLong loop = stop - start;
    bool packed_C = false;
    T3* __restrict C_PTR_ORIG = C;
    bool out_order_f = cStride_M < cStride_N;
    if (out_order_f || cStride_N != 1) {
        C_PTR_ORIG = new T3[M * N];
        memset(C_PTR_ORIG, 0, sizeof(T3) * M * N);
        packed_C = true;

    }

    if (packed_C) {

        for (Nd4jLong i = 0; i < loop; i++) {
            T3* __restrict C_PTR = C_PTR_ORIG;
            // memset(C_PTR, 0, sizeof(T3) * M * N);

            inner_gemm_no_checks(M, N, K, alphaZ, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N, C_PTR, N);

            T3* __restrict CX = &(C[offset.third]);
            if (out_order_f) {

                if (betaPersent) {
                    for (Nd4jLong n = 0; n < N; n++) {
                        T3* __restrict C_SOURCE = &(C_PTR[n]);
                        for (Nd4jLong m = 0; m < M; m++) {
                            CX[m * cStride_M] = beta * CX[m * cStride_M] + C_SOURCE[0];
                            C_SOURCE[0] = 0;
                            C_SOURCE += N;
                        }//N
                        CX += cStride_N;
                    }//M 


                }
                else {
                    for (Nd4jLong n = 0; n < N; n++) {
                        T3* __restrict C_SOURCE = &(C_PTR[n]);
                        for (Nd4jLong m = 0; m < M; m++) {
                            CX[m * cStride_M] = C_SOURCE[0];
                            C_SOURCE[0] = 0;
                            C_SOURCE += N;
                        }//N
                        CX += cStride_N;
                    }//M  

                }

            }
            else {
                if (betaPersent) {
                    for (Nd4jLong m = 0; m < M; m++) {

                        for (Nd4jLong n = 0; n < N; n++) {
                            CX[n * cStride_N] = beta * CX[n * cStride_N] + C_PTR[n];
                            C_PTR[n] = 0;
                        }//M

                        C_PTR += N;
                        CX += cStride_M;
                    }//N
                }
                else {
                    for (Nd4jLong m = 0; m < M; m++) {

                        for (Nd4jLong n = 0; n < N; n++) {
                            CX[n * cStride_N] = C_PTR[n];
                            C_PTR[n] = 0;
                        }//M

                        C_PTR += N;
                        CX += cStride_M;
                    }//N
                }
            }


            offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
        }
        delete[] C_PTR_ORIG;

    }
    else {

        for (Nd4jLong i = 0; i < loop; i++) {
#if 0
            nd4j_printf("i:%d \na: %d b:%d v:%d    \n", i, offset.first, offset.second, offset.third);
#endif
            T3* __restrict CX = &(C[offset.third]);
            if (betaZ != 0) {

                for (Nd4jLong m = 0; m < M; m++) {
                    T3* __restrict CCX = &(CX[m * N]);
                    PRAGMA_OMP_SIMD
                        for (Nd4jLong n = 0; n < N; n++) {

                            CCX[n] = betaZ * CCX[n];
                        }//N
                }//M
#if 0
                nd4j_printf("%lf\n", betaZ);
#endif
            }


            inner_gemm_no_checks(M, N, K, alphaZ, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N, &(C[offset.third]), cStride_M);


            offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
        }
    }

}


template <typename T1, typename T2, typename T3>
void batchedGemm3(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const T1 alpha, const T3 beta, char out_order) {

    const Nd4jLong* cShapeInfo = vC->getShapeInfo();

    const Nd4jLong* bases = &(cShapeInfo[1]);


    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();

    int max_rank = aRank > bRank ? aRank : bRank;
    max_rank = max_rank > cRank ? max_rank : cRank;

    Nd4jLong batch_len = 1;
    for (int i = 0; i < max_rank - 2; i++) {
        batch_len *= bases[i];
    }



    auto func = [vA, vB, vC, alpha, beta](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
#if 0
        auto timeStart = std::chrono::system_clock::now();
#endif
        parallel_batchedGemm3<T1, T2, T3>(vA, vB, vC, alpha, beta, start, stop);
#if 0
        auto timeEnd = std::chrono::system_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
        nd4j_printf("inner-time:: %lli\n", elapsed_time);
#endif
    };

    // samediff::Threads::parallel_tad(func, 0, batch_len);
    samediff::Threads::parallel_aligned_increment(func, 0, batch_len, 1, false);


}

template void batchedGemm3<float, float, float>(const NDArray* vA, const NDArray* vB, NDArray* vC, const float alpha, const float beta, char out_order);
template void batchedGemm3<double, double, double>(const NDArray* vA, const NDArray* vB, NDArray* vC, const double alpha, const double beta, char out_order);
