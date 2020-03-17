#include <user_codes.h>
using namespace sd;



template <typename T1, typename T2, typename T3>
void batchedGemm3(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const T1 alpha, const T3 beta, char out_order) {


    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf(); 

    const int aMaxis(aRank - 2), aKaxis(aRank - 1), bKaxis(bRank - 2), bNaxis(bRank - 1), cMaxis(cRank - 2), cNaxis(cRank - 1);

    std::vector<int> aBatchDimsV, bBatchDimsV, cBatchDimsV;

    if (aRank > 2)
        aBatchDimsV = ShapeUtils::evalDimsToExclude(aRank, { aMaxis, aKaxis });
    if (bRank > 2)
        bBatchDimsV = ShapeUtils::evalDimsToExclude(bRank, { bKaxis, bNaxis });
    if (cRank > 2)
        cBatchDimsV = ShapeUtils::evalDimsToExclude(cRank, { cMaxis, cNaxis });



    const int* aBatchDims = aBatchDimsV.data();
    const int* bBatchDims = bBatchDimsV.data();
    const int* cBatchDims = cBatchDimsV.data();

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
    T3* C = vC->bufferAsT<T3>();

    const T3 alphaZ = alpha;
    const T3 betaZ = beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->getShapeInfo();
    const Nd4jLong* bShapeInfo = vB->getShapeInfo();
    const Nd4jLong* cShapeInfo = vC->getShapeInfo();



    const Nd4jLong cLen = vC->lengthOf();

    const int K = vA->sizeAt(aKaxis);

    auto func = PRAGMA_THREADS_FOR{

        std::vector<Nd4jLong> aCoords(aRank), bCoords(bRank), cCoords(cRank);

        for (auto i = start; i < stop; ++i) {

            // evaluate C coordinates
            shape::index2coords(i, cShapeInfo, cCoords.data());

            // calculate index of current batch
            Nd4jLong batchInd;
            if (cRank > 2)
                batchInd = shape::coords2index(cShapeInfo, cCoords.data(), cRank - 2, cBatchDims);

            // evaluate A coordinates
            if (aRank > 2)
                shape::index2coords(batchInd, aShapeInfo, aCoords.data(), aRank - 2, aBatchDims);
            aCoords[aMaxis] = cCoords[cMaxis];
            aCoords[aKaxis] = 0;

            // evaluate B coordinates
            if (bRank > 2)
                shape::index2coords(batchInd, bShapeInfo, bCoords.data(), bRank - 2, bBatchDims);
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



template void batchedGemm3<float, float, float>(const NDArray* vA, const NDArray* vB, NDArray* vC, const float alpha, const float beta, char out_order);
template void batchedGemm3<double, double, double>(const NDArray* vA, const NDArray* vB, NDArray* vC, const double alpha, const double beta, char out_order);
