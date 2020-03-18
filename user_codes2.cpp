#include <user_codes.h>
#include <algorithm>
using namespace sd;


template <typename T1, typename T2, typename T3>
static void inner_gemm_no_checks(const Nd4jLong M, const Nd4jLong N, const Nd4jLong K,T3 alphaZ, const T1* __restrict A, const Nd4jLong aStride_M, const Nd4jLong aStride_K, const T2* __restrict B, Nd4jLong bStride_K, Nd4jLong bStride_N, T3 betaZ,T3* __restrict C, const Nd4jLong cStride_M, const Nd4jLong cStride_N ) {
#if 0
    nd4j_printf(" M %ld   , N %ld  , K %ld , alpha %lf, aStride_M %ld  , aStride_K %ld,    bStride_K %ld  bStride_N %ld,   cStride_M %ld   \n",
        M, N, K, alpha, aStride_M, aStride_K, bStride_K, bStride_N, cStride_M);
#endif
  
   // T3* __restrict C_BUFFER = BUFFER;
    //aurora vec_len 
    T2 C_BUF[256];
    T2 B_BUFF[256];
    T1 A_BUF0[256];   
    // Nd4jLong aStride_Mx2 = aStride_M *2;
    if (aStride_K == 1 && bStride_N == 1) {

        
        //Nd4jLong K_L = K & -8;// 8;
        //Nd4jLong M_L = M & -256;
        
        for (int n = 0; n < N; n += 256) {
            int inner_N = (int)std::min((Nd4jLong)256, N-n);
            T3* CCX = &(C[n*cStride_N ]);
            
                for (int jj_n = 0; jj_n < inner_N; jj_n++) {
                    CCX[jj_n * 1] = betaZ * CCX[jj_n];

                }//M 
            const T3* __restrict BBX = &(B[n * cStride_N]);
            for (Nd4jLong k = 0; k < K; k++) {
                const T2* __restrict BB0 = &(BBX[k * bStride_K]);
                const T1* __restrict AA = &(A[k * 1]);

                for (Nd4jLong jj = 0; jj < inner_N; jj++) {
                    B_BUFF[jj] = BB0[jj];
                }//N

                for (Nd4jLong m = 0; m < M; m++) {

                    Nd4jLong OFF_A = m * aStride_M;
                    T1 AA0 = AA[OFF_A];
                    


                    for (int jj_n = 0; jj_n < 256; jj_n++) {
                        C_BUF[jj_n] = AA0 * B_BUFF[jj_n];
                    }//N

                    T3* __restrict CX = &(CCX[m * cStride_M]);
                    //store
                    
                        if (cStride_N == 1) {
                            for (int jj_n = 0; jj_n < inner_N; jj_n++) {
                                CX[jj_n * 1] += alphaZ * C_BUF[jj_n];
                                
                            }//M

                        }
                        else {
                            for (int jj_n = 0; jj_n < inner_N; jj_n++) {
                                CX[jj_n * cStride_N] += alphaZ * C_BUF[jj_n]; 
                            }//M 
                        }
                  



                }//M
                 



            }//K

        }//N
    }
 
}


template <typename T1, typename T2, typename T3>
static void parallel_batchedGemm3(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const double alpha, const double beta, Nd4jLong start, Nd4jLong stop) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
    T3* C = vC->bufferAsT<T3>();


    Nd4jLong zero_strides[MAX_RANK] = {}; //zero strides


    const T3 alphaZ = (T3)alpha;
    const T3 betaZ = (T3)beta;

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

    int max_rank = aRank > bRank ? aRank : bRank;
    max_rank = max_rank > cRank ? max_rank : cRank;

    const int M = vA->sizeAt(aRank - 2);
    const int K = vA->sizeAt(aRank - 1);
    const int N = vC->sizeAt(cRank - 1);
    Nd4jLong aStride_M = aStrides[aRank - 2];
    Nd4jLong aStride_K = aStrides[aRank - 1];
    Nd4jLong bStride_K = bStrides[bRank - 2];
    Nd4jLong bStride_N = bStrides[bRank - 1];
    Nd4jLong cStride_M = cStrides[cRank - 2];
    Nd4jLong cStride_N = cStrides[cRank - 1];

    if (aRank == 2) {
        aStrides = (Nd4jLong*)&zero_strides;
    }
    if (bRank == 2) {
        bStrides = (Nd4jLong*)&zero_strides;
    }


    Nd4jLong coords[MAX_RANK] = {};
    Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
    sd::index2coords_C(start, max_rank - 2, bases, ptr_coords);
    //offset
    sd::triple_size_t offset = sd::offset_from_coords(aStrides, bStrides, cStrides, ptr_coords, max_rank - 2);
#if 0
    nd4j_printf("start:%d stop:%d \na: %d b:%d v:%d    \n", start, stop, offset.first, offset.second, offset.third);
#endif
    Nd4jLong loop = stop - start; 
 


    for (Nd4jLong i = 0; i < loop; i++) {
        
        // memset(C_PTR, 0, sizeof(T3) * M * N);

        inner_gemm_no_checks(M, N, K, alphaZ, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N,betaZ, &(C[offset.third]), cStride_M, cStride_N );
         

        offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
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
