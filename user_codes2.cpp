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
    Nd4jLong bStride_K = bStrides[aRank - 2];
    Nd4jLong bStride_N = bStrides[aRank - 1];
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


template <typename T1, typename T2, typename T3>
void usualGemm(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
    const double alpha, const double beta) {


    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
    T3* C = vC->bufferAsT<T3>();

    const T3 alphaZ = (T3)alpha;
    const T3 betaZ = (T3)beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->getShapeInfo();
    const Nd4jLong* bShapeInfo = vB->getShapeInfo();
    const Nd4jLong* cShapeInfo = vC->getShapeInfo();

    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();

    const Nd4jLong cLen = vC->lengthOf();

    const int K = vA->sizeAt(aKaxis);

    auto func = PRAGMA_THREADS_FOR{

        std::vector<Nd4jLong> aCoords(2), bCoords(2), cCoords(2);

        for (auto i = start; i < stop; ++i) {

            // evaluate C coordinates
            shape::index2coords(i, cShapeInfo, cCoords.data());

            // evaluate A coordinates
            aCoords[aMaxis] = cCoords[cMaxis];
            aCoords[aKaxis] = 0;

            // evaluate B coordinates
            bCoords[bKaxis] = 0;
            bCoords[bNaxis] = cCoords[cNaxis];

            auto aOffset = shape::getOffset(aShapeInfo, aCoords.data());
            auto bOffset = shape::getOffset(bShapeInfo, bCoords.data());

            T3 val = A[aOffset] * B[bOffset];                       // first iteration

            for (int j = 1; j < K; ++j) {                          // rest iterations
                aOffset += shape::stride(aShapeInfo)[aKaxis];
                bOffset += shape::stride(bShapeInfo)[bKaxis];
                val = val + A[aOffset] * B[bOffset];
            }

            auto cOffset = shape::getOffset(cShapeInfo, cCoords.data());

            if (betaPersent)
                C[cOffset] = alphaZ * val + betaZ * C[cOffset];
            else
                C[cOffset] = alphaZ * val;
        }
    };

    samediff::Threads::parallel_tad(func, 0, cLen);
}



NDArray* mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    if (A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if (B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");

    const auto M = A->sizeAt(0);
    const auto K = A->sizeAt(1);
    const auto N = B->sizeAt(1);

    if (C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");
    if (B->sizeAt(0) != K)
        throw std::runtime_error("MmulHelper::mmulMxM: B array has wrong number of rows !");
    if (C != nullptr && C->sizeAt(0) != M)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of rows !");
    if (C != nullptr && C->sizeAt(1) != N)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of columns !");
    const auto aType = A->dataType();
    const auto bType = B->dataType();


    const bool AB(aType == bType);

    const bool typeDouble = AB && aType == DataType::DOUBLE;
    const bool typeFloat = AB && aType == DataType::FLOAT32;

    if (!C) {
        C = new NDArray();
        if (typeFloat) {
            //  fprintf(stdout, "--float--\n");
            *C = NDArrayFactory::create<float>(outOrder, { M,N });
        }
        else {
            //  fprintf(stdout, "--double--\n");
            *C = NDArrayFactory::create<double>(outOrder, { M,N });
        }

    }
    if (C->isEmpty())
        return C;


    if (typeFloat) {
        //fprintf(stdout, "-gemm-float--\n");
        usualGemm<float, float, float>(A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta);
    }
    else {
        //fprintf(stdout, "-gemm-double--\n");
        usualGemm<double, double, double>(A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta);
    }

    return C;
}
