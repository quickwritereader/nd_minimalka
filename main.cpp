

using namespace std;

 
  
#include <chrono>

#include <array>

#include <algorithm>
#include <numeric> 
#include <random> 
#include <iostream>
#include <utils.h>
#include <user_codes.h>
using namespace sd;
 
#if 0
constexpr int outter_loops = 1;// 10;
constexpr int inner_loops = 1;// 10;
#else
constexpr int outter_loops = 2;
constexpr int inner_loops = 1;
#endif
#define CHECK_CORRECTNESS 1
int bnch_cases[][5] = {
	 {1,1,4,5,2},
	 {1,3,4,5,2}, 
	{1,1,32,64,64}, 
	{1,1,64,64,1024},
	{1,1,7,7,7},
	{1,1,31,31,31}, 
#if 1
	{4,12,128,128,64},
	{4 * 32,128,16,16,16},
	{1,4 * 32 * 128,16,16,16},
	{32,128,64,64,64},
//	{1,4096,64,64,64},
	{32,128,64,128,64},
	{1,4096,64,128,64},
//	{32,128,128,64,128},
//	{1,4096,128,64,128},
	{32,128,128,128,128},
	{1,4096,128,128,128},
//	{32,128,64,256,64},
//	{1,4096,64,256,64},
	{32,128,32,64,64},
//	{1,4096,32,64,64},
	{4,32,512,512,512},
//	{1,4*32,512,512,512}
#endif
};
 
template <typename T>
int test() {

	char out_orders[] = { 'c' };// , 'f' };
	double alpha = 1.0;
	double betax[] = { 0.0,1.0 };


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		int batch_k1 = bnch_cases[k][0];
		int batch_k2 = bnch_cases[k][1];
		int M = bnch_cases[k][2];
		int K = bnch_cases[k][3];
		int N = bnch_cases[k][4];
		for (double beta : betax) {
			for (auto out_order : out_orders) {
				auto a = NDArrayFactory::create<T>('c', { batch_k1,batch_k2, M,K });
				auto b = NDArrayFactory::create<T>('c', { batch_k1,batch_k2, K,N });
				auto c_ref = NDArrayFactory::create<T>(out_order, { batch_k1,batch_k2, M,N });
				auto c_actual = NDArrayFactory::create<T>(out_order, { batch_k1,batch_k2, M,N });
				fill_matrice_lastC<T>(a, nullptr);
				fill_matrice_lastC<T>(b, nullptr);
				fill_matrice_lastC<T>(c_actual, nullptr, true);
				auto a_mk = a.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });
				auto b_kn = b.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });

				a_mk.reshapei({ M,K });
				b_kn.reshapei({ K,N });

				auto ptr_c_mn = mmulMxM(&a_mk, &b_kn, nullptr, alpha, beta, out_order);
				fill_matrice_lastC<T>(c_ref, ptr_c_mn);


				//check performance
				double total_FlOps = batch_k1 * batch_k2 * 2.0 * M * K * N;
				nd4j_printf("beta %f out_order %c batch_1 %d batch_2 %d M %d K %d N %d:::: ",beta, out_order, bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3], bnch_cases[k][4]);
				time_it<outter_loops, inner_loops>(batchedGemm3<T, T, T>, total_FlOps, &a, &b, &c_actual, alpha, beta, out_order);


				//check_correctness
				fill_matrice_lastC<T>(c_actual, nullptr, true);

				//MmulHelper::mmulNxN_3(&a, &b, &c_actual, alpha, beta, out_order);

				batchedGemm3<T, T, T>(&a, &b, &c_actual, alpha, beta, out_order);
#if 0

				a_mk.printShapeInfo("smallA_MK");
				a_mk.printShapeInfo("smallB_KN");
				ptr_c_mn->printShapeInfo("smallC_MN");
				a_mk.printIndexedBuffer("smallA_MK");
				b_kn.printIndexedBuffer("smallB_KN");
				ptr_c_mn->printIndexedBuffer("smallC_MN");
				a.printShapeInfo("A");
				b.printShapeInfo("B");
				c_ref.printShapeInfo("C REFERENCE");
				c_actual.printShapeInfo("C");
				a.printIndexedBuffer("A");
				b.printIndexedBuffer("B");
				c_ref.printIndexedBuffer("C REFERENCE");
				c_actual.printIndexedBuffer("C ACTUAL");


#endif
				check_eq<T>(c_actual, c_ref);
				delete ptr_c_mn;


			}//out_order
		}//beta
	}

	return 0;
}
 

int main()
{
	test<float>();
	test<double>();
#if 0
	int x;
	std::cin >> x;
#endif
	return 0;
}
