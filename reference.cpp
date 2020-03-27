
#include <user_codes.h>
#include <algorithm>
using namespace sd;


template <typename T1, typename T2, typename T3>
void usualGemm(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
    const double alpha, const double beta) {


    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
    T3* C = vC->bufferAsT<T3>();

    const T3 alphaZ = (T3)alpha;
    const T3 betaZ  = (T3)beta;

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



NDArray* mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder, const bool transA  , const bool transB  ) {

    if (A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if (B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");

    const int aMaxis = transA ? 1 : 0;
    const int aKaxis = transA ? 0 : 1;
    const int bKaxis = transB ? 1 : 0;
    const int bNaxis = transB ? 0 : 1;
    const int cMaxis = 0;
    const int cNaxis = 1;

    const auto M = A->sizeAt(aMaxis);
    const auto K = A->sizeAt(aKaxis);
    const auto N = B->sizeAt(bNaxis);

    if (C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");
    if (B->sizeAt(bKaxis) != K)
        throw std::runtime_error("MmulHelper::mmulMxM: B array has wrong number of rows !");
    if (C != nullptr && C->sizeAt(cMaxis) != M)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of rows !");
    if (C != nullptr && C->sizeAt(cNaxis) != N)
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
        usualGemm<float, float, float>(A, B, C , aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis ,alpha, beta);
    }
    else {
        //fprintf(stdout, "-gemm-double--\n");
        usualGemm<double, double, double>(A, B, C, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta);
    }

 

    return C;
}
