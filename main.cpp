

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
 
#if defined(INNER_DEBUG)
constexpr int outter_loops = 1;// 10;
constexpr int inner_loops = 1;// 10;
#else
constexpr int outter_loops = 20;
constexpr int inner_loops  = 50;
#endif
#define CHECK_CORRECTNESS 1
int bnch_cases[][8] = {

#if   defined(INNER_DEBUG)
	 {1,1,31,49,49,31, 0,0},
	 {1,1,49,31,49,31, 1,0},
     {1,1,31,31,49,31, 1,1},
 	 {1,1,24,31,49,31, 0,1},

	 //{1,1,128,128,128,64, 0,0},
	 //{1,1,128,128,128,64, 1,0},
	 //{1,1,128,128,64,128, 1,1},
	 //{1,1,128,128,64,128, 0,1},
#else
		 {1,1,31,49,49,31, 0,0},
	 {1,1,49,31,49,31, 1,0},
	 {1,1,31,31,49,31, 1,1},
	 {1,1,24,31,49,31, 0,1},
	 {4,12,128,128,128,64, 0,0},
	  {4,12,128,128,128,64, 1,0},
	  {4,12,128,128,64,128, 1,1},
	  {4,12,128,128,64,128, 0,1},
#endif
};
 
template <typename T>
int test() {

	// { 'c', 'f' };
	double alpha = 1.0;
#if  defined(INNER_DEBUG)
	char orders[] = { 'c' };
	double betax[] =   {   0.0, 1.0, 0, 0.8 };
	bool strided[] = { false  };
#else	
	char orders[] = { 'c', 'f' };
	double betax[] = { 0.0, 1.0 };
	bool strided[] = { true,false };
#endif
 
	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		int batch_k1 = bnch_cases[k][0];
		int batch_k2 = bnch_cases[k][1];
		int M = bnch_cases[k][2];
		int K = bnch_cases[k][3];
		int B_K = bnch_cases[k][4];
		int N = bnch_cases[k][5];
		bool transA = bnch_cases[k][6] != 0;
		bool transB = bnch_cases[k][7] != 0;
		int K_K = transA ? M : K;
		int C_M = transA ? K : M;
		int C_N = transB ? B_K : N;

		for(auto b_order :orders)
		for (auto a_order : orders)
					for (auto out_order : orders)
						for (auto st : strided){ 
							for (double beta : betax){
						auto a = st? NDArrayFactory::create<T>(a_order, { batch_k1,batch_k2, M,K ,5})   : NDArrayFactory::create<T>(a_order, { batch_k1,batch_k2, M,K });
						auto b = st? NDArrayFactory::create<T>(b_order, { batch_k1,batch_k2, B_K,N ,5}) : NDArrayFactory::create<T>(b_order, { batch_k1,batch_k2, B_K,N });
						auto c_ref = NDArrayFactory::create<T>(out_order, { batch_k1,batch_k2, C_M,C_N });
						auto c_actual =st? NDArrayFactory::create<T>(out_order, { batch_k1,batch_k2, C_M,C_N ,5}): NDArrayFactory::create<T>(out_order, { batch_k1,batch_k2, C_M,C_N });


						if (st) {
							//subarray
							a = a.subarray({ NDIndex::all(), NDIndex::all(),NDIndex::all(), NDIndex::all(), NDIndex::point(0) });
							b = b.subarray({ NDIndex::all(), NDIndex::all(),NDIndex::all(), NDIndex::all(), NDIndex::point(0) });
#if  defined(INNER_DEBUG)
							c_actual.printShapeInfo("C");
#endif
							c_actual = c_actual.subarray({ NDIndex::all(), NDIndex::all(),NDIndex::all(), NDIndex::all(), NDIndex::point(0) });
#if  defined(INNER_DEBUG)							
							c_actual.printShapeInfo("C--");
#endif
							a.reshapei({ batch_k1,batch_k2, M,K },a_order);
							b.reshapei({ batch_k1,batch_k2, B_K,N }, b_order);
							c_actual.reshapei({ batch_k1,batch_k2, C_M,C_N },out_order);

#if  defined(INNER_DEBUG)
							a.printShapeInfo("A");
							b.printShapeInfo("B"); 
							c_actual.printShapeInfo("C");

#endif
						}

						fill_matrice_lastC<T>(a, nullptr);
						fill_matrice_lastC<T>(b, nullptr);
						fill_matrice_lastC<T>(c_actual, nullptr, true);

						auto a_mk = a.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });
						auto b_kn = b.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });

						a_mk.reshapei({ M,K },a_order);
						b_kn.reshapei({ B_K,N },b_order);


						auto ptr_c_mn = mmulMxM(&a_mk, &b_kn, nullptr, alpha, beta, out_order,transA,transB);
						fill_matrice_lastC<T>(c_ref, ptr_c_mn);


						//check performance
						double total_FlOps = batch_k1 * batch_k2 * 2.0 * C_M * C_N * K_K;
						nd4j_printf("strided %d beta %f A_order %c B_order %c out_order %c batch_1 %d batch_2 %d M %d K %d  B_K %d N %d:: transA %d transB %d:: ", st, beta, a_order, b_order, out_order, batch_k1, batch_k2, M, K, B_K, N, transA, transB);

#if   !defined(INNER_DEBUG)			
						time_it<outter_loops, inner_loops>(batchedGemm3<T, T, T>, total_FlOps, &a, &b, &c_actual, alpha, beta, out_order, transA, transB);
#else
						nd4j_printf("\n");
#endif

						//check_correctness
						fill_matrice_lastC<T>(c_actual, nullptr, true);

						//MmulHelper::mmulNxN_3(&a, &b, &c_actual, alpha, beta, out_order);

						batchedGemm3<T, T, T>(&a, &b, &c_actual, alpha, beta, out_order, transA, transB);
#if  defined(INNER_DEBUG_2)

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
 
 
int test2() {

	const int bS = 4;
	const int K = 3;
	const int N = 4;
	double alpha = 1.0;
	double beta = 0.0;
	auto input = NDArrayFactory::create<double>('c', { bS,  K, N });
	auto weights = NDArrayFactory::create<double>('c', { 3 * K, K });
	auto c_ref = NDArrayFactory::create<double>('c', { bS,  3 * K, N });

	auto c_actual = NDArrayFactory::create<double>('c', { bS,3*K,N });
	fill_matrice_lastC<double>(input, nullptr);
	fill_matrice_lastC<double>(weights, nullptr);
	fill_matrice_lastC<double>(c_actual, nullptr, true);
	auto inputs_kn = input.subarray({ NDIndex::point(0) , NDIndex::all(), NDIndex::all() }); 
 
	inputs_kn.reshapei({ K,N });

	auto ptr_c_mn = mmulMxM(&weights, &inputs_kn, nullptr, alpha, beta, 'c');
	fill_matrice_lastC<double>(c_ref, ptr_c_mn);
	 
	batchedGemm3<double, double, double>(&weights, &input, &c_actual, alpha, beta, 'c');
 
	weights.printIndexedBuffer("weights");
	input.printIndexedBuffer("input");
	//result->printShapeInfo("result shape");
	c_ref.printIndexedBuffer("result buffer");
	c_actual.printIndexedBuffer("actual buffer"); 

	delete ptr_c_mn;
 
	return 0;
}


int main()
{
#if  defined(INNER_DEBUG)
	test2();
#else
#if 1
	test<float>();
#else
	test<double>();
#endif
#endif
#if  defined(INNER_DEBUG)
	int x;
	std::cin >> x;
#endif
	return 0;
}
