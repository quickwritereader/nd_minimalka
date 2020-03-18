#include <user_codes.h>
using namespace sd;


template <typename T1, typename T2, typename T3>
static void inner_gemm_no_checks(const Nd4jLong M, const Nd4jLong N, const Nd4jLong K,const T1 alpha,const T1* __restrict A, const Nd4jLong aStride_M, const Nd4jLong aStride_K, const T2* __restrict B, Nd4jLong bStride_K, Nd4jLong bStride_N, T3* __restrict C, const Nd4jLong cStride_M) {
#if 0
    nd4j_printf(" M %ld   , N %ld  , K %ld , alpha %lf, aStride_M %ld  , aStride_K %ld,    bStride_K %ld  bStride_N %ld,   cStride_M %ld   \n",
        M, N, K, alpha, aStride_M, aStride_K, bStride_K, bStride_N, cStride_M);
#endif

    // Nd4jLong aStride_Mx2 = aStride_M *2;

    if ( aStride_K == 1 && bStride_N == 1) {
        //cStride_N ==1
        Nd4jLong M_L =   M & -16;;
        Nd4jLong K_L = K & -8;// 8;

        for (Nd4jLong k = 0; k < K_L; k += 8) {

            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T2* __restrict BB1 = &(B[k * bStride_K + bStride_K]);
            const T2* __restrict BB2 = &(B[k * bStride_K + 2 * bStride_K]);
            const T2* __restrict BB3 = &(B[k * bStride_K + 3 * bStride_K]);
            const T2* __restrict BB4 = &(B[k * bStride_K + 4 * bStride_K]);
            const T2* __restrict BB5 = &(B[k * bStride_K + 5 * bStride_K]);
            const T2* __restrict BB6 = &(B[k * bStride_K + 6 * bStride_K]);
            const T2* __restrict BB7 = &(B[k * bStride_K + 7 * bStride_K]);
            const T1* __restrict AA = &(A[k * 1]);
            for (Nd4jLong m = 0; m < M_L; m += 16) {
                Nd4jLong OFF_A = m * aStride_M;
                const T1* __restrict PTR_A0 = &(AA[OFF_A]);
                const T1* __restrict PTR_A1 = &(AA[OFF_A + aStride_M]);
                const T1* __restrict PTR_A2 = &(AA[OFF_A + 2 * aStride_M]);
                const T1* __restrict PTR_A3 = &(AA[OFF_A + 3 * aStride_M]);
                const T1* __restrict PTR_A4 = &(AA[OFF_A + 4 * aStride_M]);
                const T1* __restrict PTR_A5 = &(AA[OFF_A + 5 * aStride_M]);
                const T1* __restrict PTR_A6 = &(AA[OFF_A + 6 * aStride_M]);
                const T1* __restrict PTR_A7 = &(AA[OFF_A + 7 * aStride_M]);
                const T1* __restrict PTR_A8 = &(AA[OFF_A + 8 * aStride_M]);
                const T1* __restrict PTR_A9 = &(AA[OFF_A + 9 * aStride_M]);
                const T1* __restrict PTR_A10 = &(AA[OFF_A + 10 * aStride_M]);
                const T1* __restrict PTR_A11 = &(AA[OFF_A + 11 * aStride_M]);
                const T1* __restrict PTR_A12 = &(AA[OFF_A + 12 * aStride_M]);
                const T1* __restrict PTR_A13 = &(AA[OFF_A + 13 * aStride_M]);
                const T1* __restrict PTR_A14 = &(AA[OFF_A + 14 * aStride_M]);
                const T1* __restrict PTR_A15 = &(AA[OFF_A + 15 * aStride_M]);

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                T3* __restrict CC4 = &(C[m * cStride_M + 4 * cStride_M]);
                T3* __restrict CC5 = &(C[m * cStride_M + 5 * cStride_M]);
                T3* __restrict CC6 = &(C[m * cStride_M + 6 * cStride_M]);
                T3* __restrict CC7 = &(C[m * cStride_M + 7 * cStride_M]);

                T3* __restrict CC8 = &(C[m * cStride_M + 8 * cStride_M]);
                T3* __restrict CC9 = &(C[m * cStride_M + 9 * cStride_M]);
                T3* __restrict CC10 = &(C[m * cStride_M + 10 * cStride_M]);
                T3* __restrict CC11 = &(C[m * cStride_M + 11 * cStride_M]);

                T3* __restrict CC12 = &(C[m * cStride_M + 12 * cStride_M]);
                T3* __restrict CC13 = &(C[m * cStride_M + 13 * cStride_M]);
                T3* __restrict CC14 = &(C[m * cStride_M + 14 * cStride_M]);
                T3* __restrict CC15 = &(C[m * cStride_M + 15 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {

                        CC0[n] += alpha * (*(PTR_A0)*BB0[n] + *(PTR_A0 + 1) * BB1[n] + *(PTR_A0 + 2) * BB2[n] + *(PTR_A0 + 3) * BB3[n] + *(PTR_A0 + 4) * BB4[n] + *(PTR_A0 + 5) * BB5[n] + *(PTR_A0 + 6) * BB6[n] + *(PTR_A0 + 7) * BB7[n]);
                        CC1[n] += alpha * (*(PTR_A1)*BB0[n] + *(PTR_A1 + 1) * BB1[n] + *(PTR_A1 + 2) * BB2[n] + *(PTR_A1 + 3) * BB3[n] + *(PTR_A1 + 4) * BB4[n] + *(PTR_A1 + 5) * BB5[n] + *(PTR_A1 + 6) * BB6[n] + *(PTR_A1 + 7) * BB7[n]);
                        CC2[n] += alpha * (*(PTR_A2)*BB0[n] + *(PTR_A2 + 1) * BB1[n] + *(PTR_A2 + 2) * BB2[n] + *(PTR_A2 + 3) * BB3[n] + *(PTR_A2 + 4) * BB4[n] + *(PTR_A2 + 5) * BB5[n] + *(PTR_A2 + 6) * BB6[n] + *(PTR_A2 + 7) * BB7[n]);
                        CC3[n] += alpha * (*(PTR_A3)*BB0[n] + *(PTR_A3 + 1) * BB1[n] + *(PTR_A3 + 2) * BB2[n] + *(PTR_A3 + 3) * BB3[n] + *(PTR_A3 + 4) * BB4[n] + *(PTR_A3 + 5) * BB5[n] + *(PTR_A3 + 6) * BB6[n] + *(PTR_A3 + 7) * BB7[n]);

                        CC4[n] += alpha * (*(PTR_A4)*BB0[n] + *(PTR_A4 + 1) * BB1[n] + *(PTR_A4 + 2) * BB2[n] + *(PTR_A4 + 3) * BB3[n] + *(PTR_A4 + 4) * BB4[n] + *(PTR_A4 + 5) * BB5[n] + *(PTR_A4 + 6) * BB6[n] + *(PTR_A4 + 7) * BB7[n]);
                        CC5[n] += alpha * (*(PTR_A5)*BB0[n] + *(PTR_A5 + 1) * BB1[n] + *(PTR_A5 + 2) * BB2[n] + *(PTR_A5 + 3) * BB3[n] + *(PTR_A5 + 4) * BB4[n] + *(PTR_A5 + 5) * BB5[n] + *(PTR_A5 + 6) * BB6[n] + *(PTR_A5 + 7) * BB7[n]);
                        CC6[n] += alpha * (*(PTR_A6)*BB0[n] + *(PTR_A6 + 1) * BB1[n] + *(PTR_A6 + 2) * BB2[n] + *(PTR_A6 + 3) * BB3[n] + *(PTR_A6 + 4) * BB4[n] + *(PTR_A6 + 5) * BB5[n] + *(PTR_A6 + 6) * BB6[n] + *(PTR_A6 + 7) * BB7[n]);
                        CC7[n] += alpha * (*(PTR_A7)*BB0[n] + *(PTR_A7 + 1) * BB1[n] + *(PTR_A7 + 2) * BB2[n] + *(PTR_A7 + 3) * BB3[n] + *(PTR_A7 + 4) * BB4[n] + *(PTR_A7 + 5) * BB5[n] + *(PTR_A7 + 6) * BB6[n] + *(PTR_A7 + 7) * BB7[n]);

                        CC8[n] += alpha * (*(PTR_A8)*BB0[n] + *(PTR_A8 + 1) * BB1[n] + *(PTR_A8 + 2) * BB2[n] + *(PTR_A8 + 3) * BB3[n] + *(PTR_A8 + 4) * BB4[n] + *(PTR_A8 + 5) * BB5[n] + *(PTR_A8 + 6) * BB6[n] + *(PTR_A8 + 7) * BB7[n]);
                        CC9[n] += alpha * (*(PTR_A9)*BB0[n] + *(PTR_A9 + 1) * BB1[n] + *(PTR_A9 + 2) * BB2[n] + *(PTR_A9 + 3) * BB3[n] + *(PTR_A9 + 4) * BB4[n] + *(PTR_A9 + 5) * BB5[n] + *(PTR_A9 + 6) * BB6[n] + *(PTR_A9 + 7) * BB7[n]);
                        CC10[n] += alpha * (*(PTR_A10)*BB0[n] + *(PTR_A10 + 1) * BB1[n] + *(PTR_A10 + 2) * BB2[n] + *(PTR_A10 + 3) * BB3[n] + *(PTR_A10 + 4) * BB4[n] + *(PTR_A10 + 5) * BB5[n] + *(PTR_A10 + 6) * BB6[n] + *(PTR_A10 + 7) * BB7[n]);
                        CC11[n] += alpha * (*(PTR_A11)*BB0[n] + *(PTR_A11 + 1) * BB1[n] + *(PTR_A11 + 2) * BB2[n] + *(PTR_A11 + 3) * BB3[n] + *(PTR_A11 + 4) * BB4[n] + *(PTR_A11 + 5) * BB5[n] + *(PTR_A11 + 6) * BB6[n] + *(PTR_A11 + 7) * BB7[n]);

                        CC12[n] += alpha * (*(PTR_A12)*BB0[n] + *(PTR_A12 + 1) * BB1[n] + *(PTR_A12 + 2) * BB2[n] + *(PTR_A12 + 3) * BB3[n] + *(PTR_A12 + 4) * BB4[n] + *(PTR_A12 + 5) * BB5[n] + *(PTR_A12 + 6) * BB6[n] + *(PTR_A12 + 7) * BB7[n]);
                        CC13[n] += alpha * (*(PTR_A13)*BB0[n] + *(PTR_A13 + 1) * BB1[n] + *(PTR_A13 + 2) * BB2[n] + *(PTR_A13 + 3) * BB3[n] + *(PTR_A13 + 4) * BB4[n] + *(PTR_A13 + 5) * BB5[n] + *(PTR_A13 + 6) * BB6[n] + *(PTR_A13 + 7) * BB7[n]);
                        CC14[n] += alpha * (*(PTR_A14)*BB0[n] + *(PTR_A14 + 1) * BB1[n] + *(PTR_A14 + 2) * BB2[n] + *(PTR_A14 + 3) * BB3[n] + *(PTR_A14 + 4) * BB4[n] + *(PTR_A14 + 5) * BB5[n] + *(PTR_A14 + 6) * BB6[n] + *(PTR_A14 + 7) * BB7[n]);
                        CC15[n] += alpha * (*(PTR_A15)*BB0[n] + *(PTR_A15 + 1) * BB1[n] + *(PTR_A15 + 2) * BB2[n] + *(PTR_A15 + 3) * BB3[n] + *(PTR_A15 + 4) * BB4[n] + *(PTR_A15 + 5) * BB5[n] + *(PTR_A15 + 6) * BB6[n] + *(PTR_A15 + 7) * BB7[n]);


                    }//N             
            }//M
            for (Nd4jLong m = M_L; m < M; m++) {
                Nd4jLong OFF_A = m * aStride_M;
                const T1* __restrict PTR_A0 = &(AA[OFF_A]);
                T3* __restrict CC0 = &(C[m * cStride_M]);

                PRAGMA_OMP_SIMD
                for (Nd4jLong n = 0; n < N; n++) {
                        CC0[n] += alpha * (*(PTR_A0)*BB0[n] + *(PTR_A0 + 1) * BB1[n] + *(PTR_A0 + 2) * BB2[n] + *(PTR_A0 + 3) * BB3[n] + *(PTR_A0 + 4) * BB4[n] + *(PTR_A0 + 5) * BB5[n] + *(PTR_A0 + 6) * BB6[n] + *(PTR_A0 + 7) * BB7[n]);
                }//N             
            }//M

        }//K


        for (Nd4jLong k = K_L; k < K; k++) {
            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T1* __restrict AA = &(A[k * 1]);

            for (Nd4jLong m = 0; m < M_L; m += 16) {
                Nd4jLong OFF_A = m * aStride_M;
                T1 A0 = alpha * AA[OFF_A];
                T1 A1 = alpha * AA[OFF_A + aStride_M];
                T1 A2 = alpha * AA[OFF_A + 2 * aStride_M];
                T1 A3 = alpha * AA[OFF_A + 3 * aStride_M];
                T1 A4 = alpha * AA[OFF_A + 4 * aStride_M];
                T1 A5 = alpha * AA[OFF_A + 5 * aStride_M];
                T1 A6 = alpha * AA[OFF_A + 6 * aStride_M];
                T1 A7 = alpha * AA[OFF_A + 7 * aStride_M];
                T1 A8 = alpha * AA[OFF_A + 8 * aStride_M];
                T1 A9 = alpha * AA[OFF_A + 9 * aStride_M];
                T1 A10 = alpha * AA[OFF_A + 10 * aStride_M];
                T1 A11 = alpha * AA[OFF_A + 11 * aStride_M];
                T1 A12 = alpha * AA[OFF_A + 12 * aStride_M];
                T1 A13 = alpha * AA[OFF_A + 13 * aStride_M];
                T1 A14 = alpha * AA[OFF_A + 14 * aStride_M];
                T1 A15 = alpha * AA[OFF_A + 15 * aStride_M];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                T3* __restrict CC4 = &(C[m * cStride_M + 4 * cStride_M]);
                T3* __restrict CC5 = &(C[m * cStride_M + 5 * cStride_M]);
                T3* __restrict CC6 = &(C[m * cStride_M + 6 * cStride_M]);
                T3* __restrict CC7 = &(C[m * cStride_M + 7 * cStride_M]);

                T3* __restrict CC8 = &(C[m * cStride_M + 8 * cStride_M]);
                T3* __restrict CC9 = &(C[m * cStride_M + 9 * cStride_M]);
                T3* __restrict CC10 = &(C[m * cStride_M + 10 * cStride_M]);
                T3* __restrict CC11 = &(C[m * cStride_M + 11 * cStride_M]);

                T3* __restrict CC12 = &(C[m * cStride_M + 12 * cStride_M]);
                T3* __restrict CC13 = &(C[m * cStride_M + 13 * cStride_M]);
                T3* __restrict CC14 = &(C[m * cStride_M + 14 * cStride_M]);
                T3* __restrict CC15 = &(C[m * cStride_M + 15 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {

                        CC0[n] += A0 * BB0[n];
                        CC1[n] += A1 * BB0[n];
                        CC2[n] += A2 * BB0[n];
                        CC3[n] += A3 * BB0[n];
                        CC4[n] += A4 * BB0[n];
                        CC5[n] += A5 * BB0[n];
                        CC6[n] += A6 * BB0[n];
                        CC7[n] += A7 * BB0[n];
                        CC8[n] += A8 * BB0[n];
                        CC9[n] += A9 * BB0[n];
                        CC10[n] += A10 * BB0[n];
                        CC11[n] += A11 * BB0[n];
                        CC12[n] += A12 * BB0[n];
                        CC13[n] += A13 * BB0[n];
                        CC14[n] += A14 * BB0[n];
                        CC15[n] += A15 * BB0[n];

                    }//N             
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {

                Nd4jLong OFF_A = m * aStride_M;
                T1 AA0 = alpha * AA[OFF_A];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                for (Nd4jLong n = 0; n < N; n++) {
                        CC[n] += AA0 * BB0[n];
                }//N
            }//M
        }//K
    }
    else {
    Nd4jLong M_L = M & -4;;
    Nd4jLong K_L = K & -4;
        
        for (Nd4jLong k = 0; k < K_L; k += 4) {

            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T2* __restrict BB1 = &(B[k * bStride_K + bStride_K]);
            const T2* __restrict BB2 = &(B[k * bStride_K + 2 * bStride_K]);
            const T2* __restrict BB3 = &(B[k * bStride_K + 3 * bStride_K]);

            const T1* __restrict AA = &(A[k * aStride_K]);
            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 =  AA[m * aStride_M];
                T1 AA0_1 =  AA[m * aStride_M + aStride_K];
                T1 AA0_2 =  AA[m * aStride_M + 2 * aStride_K];
                T1 AA0_3 =  AA[m * aStride_M + 3 * aStride_K];

                T1 AA1 =  AA[m * aStride_M + aStride_M];
                T1 AA1_1 =  AA[m * aStride_M + aStride_M + aStride_K];
                T1 AA1_2 =  AA[m * aStride_M + aStride_M + 2 * aStride_K];
                T1 AA1_3 =  AA[m * aStride_M + aStride_M + 3 * aStride_K];

                T1 AA2 =  AA[m * aStride_M + 2 * aStride_M];
                T1 AA2_1 =  AA[m * aStride_M + 2 * aStride_M + aStride_K];
                T1 AA2_2 =  AA[m * aStride_M + 2 * aStride_M + 2 * aStride_K];
                T1 AA2_3 =  AA[m * aStride_M + 2 * aStride_M + 3 * aStride_K];

                T1 AA3 =  AA[m * aStride_M + 3 * aStride_M];
                T1 AA3_1 =  AA[m * aStride_M + 3 * aStride_M + aStride_K];
                T1 AA3_2 =  AA[m * aStride_M + 3 * aStride_M + 2 * aStride_K];
                T1 AA3_3 =  AA[m * aStride_M + 3 * aStride_M + 3 * aStride_K];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC0[n] += alpha * (AA0 * BB0[n * bStride_N] + AA0_1 * BB1[n * bStride_N] + AA0_2 * BB2[n * bStride_N] + AA0_3 * BB3[n * bStride_N]);
                        CC1[n] += alpha * (AA1 * BB0[n * bStride_N] + AA1_1 * BB1[n * bStride_N] + AA1_2 * BB2[n * bStride_N] + AA1_3 * BB3[n * bStride_N]);
                        CC2[n] += alpha * (AA2 * BB0[n * bStride_N] + AA2_1 * BB1[n * bStride_N] + AA2_2 * BB2[n * bStride_N] + AA2_3 * BB3[n * bStride_N]);
                        CC3[n] += alpha * (AA3 * BB0[n * bStride_N] + AA3_1 * BB1[n * bStride_N] + AA3_2 * BB2[n * bStride_N] + AA3_3 * BB3[n * bStride_N]);
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {

                T1 AA0 =  AA[m * aStride_M];
                T1 AA0_1 =  AA[m * aStride_M + aStride_K];
                T1 AA0_2 =  AA[m * aStride_M + 2 * aStride_K];
                T1 AA0_3 =  AA[m * aStride_M + 3 * aStride_K];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC[n] += alpha * (AA0 * BB0[n * bStride_N] + AA0_1 * BB1[n * bStride_N] + AA0_2 * BB2[n * bStride_N] + AA0_3 * BB3[n * bStride_N]);
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

template<typename T3>
static FORCEINLINE void zero_buffer(T3* C_PTR, int M, int N) {
    int M_8 = M & (-8);

    for (Nd4jLong m = 0; m < M_8; m += 8) {
        T3* C_PTR1 = &(C_PTR[N]);
        T3* C_PTR2 = &(C_PTR[2 * N]);
        T3* C_PTR3 = &(C_PTR[3 * N]);
        T3* C_PTR4 = &(C_PTR[4 * N]);
        T3* C_PTR5 = &(C_PTR[5 * N]);
        T3* C_PTR6 = &(C_PTR[6 * N]);
        T3* C_PTR7 = &(C_PTR[7 * N]);

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = 0;
            C_PTR1[n] = 0;
            C_PTR2[n] = 0;
            C_PTR3[n] = 0;
            C_PTR4[n] = 0;
            C_PTR5[n] = 0;
            C_PTR6[n] = 0;
            C_PTR7[n] = 0;
        }//M

        C_PTR += 8 * N;
    }//N

    for (Nd4jLong m = M_8; m < M; m++) {

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = 0;
        }//M 
        C_PTR += N;
    }//N

}

template<typename T3>
static FORCEINLINE void scal_buffer(const T3 beta, T3* C_PTR, int M, int N) {
    int M_8 = M & (-8);

    for (Nd4jLong m = 0; m < M_8; m += 8) {
        T3* C_PTR1 = &(C_PTR[N]);
        T3* C_PTR2 = &(C_PTR[2 * N]);
        T3* C_PTR3 = &(C_PTR[3 * N]);
        T3* C_PTR4 = &(C_PTR[4 * N]);
        T3* C_PTR5 = &(C_PTR[5 * N]);
        T3* C_PTR6 = &(C_PTR[6 * N]);
        T3* C_PTR7 = &(C_PTR[7 * N]);

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = beta * C_PTR[n];
            C_PTR1[n] = beta * C_PTR1[n];
            C_PTR2[n] = beta * C_PTR2[n];
            C_PTR3[n] = beta * C_PTR3[n];
            C_PTR4[n] = beta * C_PTR4[n];
            C_PTR5[n] = beta * C_PTR5[n];
            C_PTR6[n] = beta * C_PTR6[n];
            C_PTR7[n] = beta * C_PTR7[n];
        }//M

        C_PTR += 8 * N;
    }//N

    for (Nd4jLong m = M_8; m < M; m++) {

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = beta * C_PTR[n];
        }//M 
        C_PTR += N;
    }//N

}

template<typename T3>
void copy_buffer(T3* dest, T3* source,  T3 betaZ, int M, int N, Nd4jLong dest_stride_m, Nd4jLong dest_stride_n) {
    if (dest_stride_n == 1 && dest_stride_m == N) {
        int M_8 = M & (-8);

        if ((bool)betaZ) {

            for (Nd4jLong m = 0; m < M_8; m += 8) {

                T3* dest1 = &(dest[1 * N]);
                T3* dest2 = &(dest[2 * N]);
                T3* dest3 = &(dest[3 * N]);
                T3* dest4 = &(dest[4 * N]);
                T3* dest5 = &(dest[5 * N]);
                T3* dest6 = &(dest[6 * N]);
                T3* dest7 = &(dest[7 * N]);

                T3* source1 = &(source[1 * N]);
                T3* source2 = &(source[2 * N]);
                T3* source3 = &(source[3 * N]);
                T3* source4 = &(source[4 * N]);
                T3* source5 = &(source[5 * N]);
                T3* source6 = &(source[6 * N]);
                T3* source7 = &(source[7 * N]);

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1]  = betaZ * dest[n * 1]  +  source[n];
                    dest1[n * 1] = betaZ * dest1[n * 1] +  source1[n];
                    dest2[n * 1] = betaZ * dest2[n * 1] +  source2[n];
                    dest3[n * 1] = betaZ * dest3[n * 1] +  source3[n];
                    dest4[n * 1] = betaZ * dest4[n * 1] +  source4[n];
                    dest5[n * 1] = betaZ * dest5[n * 1] +  source5[n];
                    dest6[n * 1] = betaZ * dest6[n * 1] +  source6[n];
                    dest7[n * 1] = betaZ * dest7[n * 1] +  source7[n];
                }//M


                source += 8 * N;
                dest += 8 * N;
            }//N

            for (Nd4jLong m = M_8; m < M; m++) {

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1] = betaZ * dest[n * 1] + source[n];
                    source[n] = 0;
                }//M 
                source += N;
                dest += N;
            }//N
        }
        else {

            for (Nd4jLong m = 0; m < M_8; m += 8) {

                T3* dest1 = &(dest[N]);
                T3* dest2 = &(dest[2 * N]);
                T3* dest3 = &(dest[3 * N]);
                T3* dest4 = &(dest[4 * N]);
                T3* dest5 = &(dest[5 * N]);
                T3* dest6 = &(dest[6 * N]);
                T3* dest7 = &(dest[7 * N]);

                T3* source1 = &(source[N]);
                T3* source2 = &(source[2 * N]);
                T3* source3 = &(source[3 * N]);
                T3* source4 = &(source[4 * N]);
                T3* source5 = &(source[5 * N]);
                T3* source6 = &(source[6 * N]);
                T3* source7 = &(source[7 * N]);

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1]  = source[n];
                    dest1[n * 1] = source1[n];
                    dest2[n * 1] = source2[n];
                    dest3[n * 1] = source3[n];
                    dest4[n * 1] = source4[n];
                    dest5[n * 1] = source5[n];
                    dest6[n * 1] = source6[n];
                    dest7[n * 1] = source7[n];
                }//M


                source += 8 * N;
                dest += 8 * N;
            }//N

            for (Nd4jLong m = M_8; m < M; m++) {

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1] = source[n];
                    source[n] = 0;
                }//M 
                source += N;
                dest += N;
            }//N

        }//betaZ

    }
    else if (dest_stride_m < dest_stride_n) {

        if ((bool)betaZ) {
            for (Nd4jLong n = 0; n < N; n++) {
                T3* __restrict source_0 = &(source[n]);
                for (Nd4jLong m = 0; m < M; m++) {
                    dest[m * dest_stride_m] = betaZ * dest[m * dest_stride_m] + (*source_0);
                    source_0 += N;
                }//N
                dest += dest_stride_n;
            }//M

        }
        else {
            for (Nd4jLong n = 0; n < N; n++) {
                T3* __restrict source_0 = &(source[n]);
                for (Nd4jLong m = 0; m < M; m++) {
                    dest[m * dest_stride_m] = (*source_0);
                    source_0 += N;
                }//N
                dest += dest_stride_n;
            }//M  

        }

    }
    else {
        for (Nd4jLong m = 0; m < M; m++) {
            for (Nd4jLong n = 0; n < N; n++) {
                dest[n * dest_stride_n] = betaZ * dest[n * dest_stride_n] + source[n];
            }//M

            source += N;
            dest += dest_stride_m;
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
    const T1 alphaA = (T1)alpha;
    const T3 betaZ = (T3)beta; 

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
    bool out_order_f = cStride_M < cStride_N;
    T3* __restrict C_PTR_ORIG = C;
    bool allocate_buffer = out_order_f || cStride_N != 1;

    if (allocate_buffer) {
        C_PTR_ORIG = new T3[M * N];
    }
#if 0
    allocate_buffer = false;
#endif
    if (allocate_buffer) {

        for (Nd4jLong i = 0; i < loop; i++) {
            zero_buffer(C_PTR_ORIG, M, N);
            inner_gemm_no_checks(M, N, K, alphaA, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N, C_PTR_ORIG, N);
            T3* __restrict CX = &(C[offset.third]);
            copy_buffer(CX, C_PTR_ORIG, betaZ, M, N, cStride_M, cStride_N);
            offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
        }
    }
    else {
        for (Nd4jLong i = 0; i < loop; i++) {
            T3* __restrict CX = &(C[offset.third]);
            scal_buffer(betaZ, CX, M, N);
            inner_gemm_no_checks(M, N, K, alphaA, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N, CX, cStride_M);
            
            offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
        }

    }
     
    if (allocate_buffer) {
        delete[] C_PTR_ORIG;
    }
}

 
template <typename T1, typename T2, typename T3>
 void batchedGemm3(const NDArray* vA, const NDArray* vB, NDArray* vC,  
    const T1 alpha, const T3 beta,char out_order ) {

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

 

    auto func = [vA, vB, vC,   alpha, beta](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
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

 template void batchedGemm3<float, float, float>(const NDArray* vA, const NDArray* vB, NDArray* vC,  const float alpha, const float beta, char out_order);
 template void batchedGemm3<double, double, double>(const NDArray* vA, const NDArray* vB, NDArray* vC, const double alpha, const double beta, char out_order);

