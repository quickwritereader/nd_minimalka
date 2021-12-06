

using namespace std;

 
  
#include <chrono>

#include <array>

#include <algorithm>
#include <numeric> 
#include <random> 
#include <iostream>
#include <utils.h>
#include <fstream>
#include <vector>
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

#include <assert.h>

void testYY() {


	constexpr int n = 2;
	constexpr int c = 3;
	constexpr int h = 4;
	constexpr int w = 5;
	constexpr int n_pad = 2;
	constexpr int c_pad = 3;
	constexpr int h_pad = 4;
	constexpr int w_pad = 5;
	char orders[] = { 'c', 'f' };
	for (auto& order : orders) {
		std::cout << "---------------" << std::endl;
		auto shapeDesc1 = ShapeDescriptor::paddedBufferDescriptor(DataType::FLOAT32, order, { n, c, h, w }, { n_pad, c_pad, h_pad, w_pad });
		auto shapeDesc2 = ShapeDescriptor(DataType::FLOAT32, order, { n + n_pad, c + c_pad, h + h_pad, w + w_pad });
		auto shapeDesc3 = ShapeDescriptor::paddedBufferDescriptor(DataType::FLOAT32, order, { n, c, h, w }, { n_pad, c_pad });
		auto shapeDesc4 = ShapeDescriptor(DataType::FLOAT32, order, { n + n_pad, c + c_pad, h, w });
		auto shapeDesc5 = ShapeDescriptor::paddedBufferDescriptor(DataType::FLOAT32, order, { n, c, h, w }, { 0, 0, h_pad, w_pad });
		auto shapeDesc6 = ShapeDescriptor(DataType::FLOAT32, order, { n, c , h + h_pad, w + w_pad });

		//std::cout << "------offfst-------" << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ 1,1,1,w_pad  }) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ 1,1,1,w_pad + 1 }) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ 1,1,h_pad,1}) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ 1,1,h_pad+1,1 }) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ 1,c_pad,1,1 }) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ 1,c_pad+1,1,1 }) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ n_pad,1,1,1 }) << std::endl;
		//std::cout << shapeDesc1.offsetInFull({ n_pad+1,1,1,1 }) << std::endl;
		//std::cout << "------offfst-------" << std::endl;

		std::cout << shapeDesc1.validate() << " " << shapeDesc1.allocLength() << " " << shapeDesc1.fullAllocLength() << std::endl;
		std::cout << shapeDesc2.validate() << " " << shapeDesc2.allocLength() << " " << shapeDesc2.fullAllocLength()<<  std::endl;
		std::cout << shapeDesc3.validate() << " " << shapeDesc3.allocLength() << " " << shapeDesc3.fullAllocLength() << std::endl;
		std::cout << shapeDesc4.validate() << " " << shapeDesc4.allocLength() << " " << shapeDesc4.fullAllocLength() << std::endl;
		std::cout << shapeDesc5.validate() << " " << shapeDesc5.allocLength() << " " << shapeDesc5.fullAllocLength() << std::endl;
		std::cout << shapeDesc6.validate() << " " << shapeDesc6.allocLength() << " " << shapeDesc6.fullAllocLength() << std::endl;
		assert(shapeDesc1.validate() == SHAPE_DESC_OK);
		assert(shapeDesc2.validate() == SHAPE_DESC_OK);
		assert(shapeDesc3.validate() == SHAPE_DESC_OK);
		assert(shapeDesc4.validate() == SHAPE_DESC_OK);
		assert(shapeDesc5.validate() == SHAPE_DESC_OK);
		assert(shapeDesc6.validate() == SHAPE_DESC_OK);

		assert(shapeDesc1.fullAllocLength() == shapeDesc2.fullAllocLength());
        assert(shapeDesc3.fullAllocLength()==shapeDesc4.fullAllocLength());
        assert(shapeDesc5.fullAllocLength()==shapeDesc6.fullAllocLength());

		const auto& v1 = shapeDesc1.strides();
		const auto& v2 = shapeDesc2.strides();
		const auto& v3 = shapeDesc3.strides();
		const auto& v4 = shapeDesc4.strides();
		const auto& v5 = shapeDesc5.strides();
		const auto& v6 = shapeDesc6.strides();
		for (int i = 0; i < v1.size(); i++) {
			assert(v1[i] == v2[i]);
			std::cout << "," << v1[i];
		}
		std::cout << std::endl;
		for (int i = 0; i < v3.size(); i++) {
			assert(v3[i] == v4[i]);
		}
		for (int i = 0; i < v5.size(); i++) {
			assert(v5[i] == v6[i]);
		}
	}

}

void testXX() {
	 
		//for c order
		std::vector<Nd4jLong> shape{ 2,3,4,5 };
		std::vector<Nd4jLong> incorrectStride1{ 20,20,5,1 };
		std::vector<Nd4jLong> incorrectStride2{ 60,20,5,5 };
		std::vector<Nd4jLong> correctStride1{ 60,20,5,1 };
		std::vector<Nd4jLong> correctStride2{ 300,100,25,5 };
		std::vector<Nd4jLong> correctStride3{ 800, 200, 40, 5 }; 

		auto shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, incorrectStride1, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_INCORRECT_STRIDES);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, correctStride1, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_OK);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, incorrectStride2, 1);
		assert(shapeDesc.validate() == (SHAPE_DESC_INCORRECT_STRIDES | SHAPE_DESC_INCORRECT_EWS));
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, correctStride2, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, correctStride2, 5);
		assert(shapeDesc.validate() == SHAPE_DESC_OK);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, correctStride3, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'c', shape, correctStride3, 0);
		assert(shapeDesc.validate() == SHAPE_DESC_OK);
	
		//order f
		std::reverse(std::begin(shape), std::end(shape));
		std::reverse(std::begin(incorrectStride1), std::end(incorrectStride1));
		std::reverse(std::begin(incorrectStride2), std::end(incorrectStride2));
		std::reverse(std::begin(correctStride1), std::end(correctStride1));
		std::reverse(std::begin(correctStride2), std::end(correctStride2));
		std::reverse(std::begin(correctStride3), std::end(correctStride3));

		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, incorrectStride1, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_INCORRECT_STRIDES);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, correctStride1, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_OK);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, incorrectStride2, 1);
		assert(shapeDesc.validate() == (SHAPE_DESC_INCORRECT_STRIDES | SHAPE_DESC_INCORRECT_EWS));
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, correctStride2, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, correctStride2, 5);
		assert(shapeDesc.validate() == SHAPE_DESC_OK);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, correctStride3, 1);
		assert(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
		shapeDesc = ShapeDescriptor(DataType::FLOAT32, 'f', shape, correctStride3, 0);
		assert(shapeDesc.validate() == SHAPE_DESC_OK);

}


int main()
{
#if 0
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

//#else
	const int bS = 5;
	const int K = 3;
	const int N = 4;
	double alpha = 1.0;
	double beta = 0.0;

	auto input0 = NDArrayFactory::create<float>('c', { bS,  K, N } );
	auto input = NDArrayFactory::createWithStrides('c', { bS,  K, N }, { 1,K * N,N }, sd::DataType::FLOAT32);
	input0.printShapeInfo("0000 shapeINfo strides");
	input.printShapeInfo("shapeINfo strides");

	auto input10 = NDArrayFactory::create<float>('f', { bS,  K, N });
	auto input11 = NDArrayFactory::createWithStrides('f', { bS,  K, N }, { 1,K*N,N}, sd::DataType::FLOAT32);
	input10.printShapeInfo("0010 shapeINfo strides");
	input11.printShapeInfo("0011  shapeINfo strides");


	auto input20 = NDArrayFactory::createWithPadding('c', { bS,  K, N }, sd::DataType::FLOAT32, 0,0,0,0);
	auto input21 = NDArrayFactory::createWithPadding('c', { bS,  K, N }, sd::DataType::FLOAT32, 1,1,1,2);
	auto input22 = NDArrayFactory::createWithPadding('c', { bS,  K, N }, sd::DataType::FLOAT32, 2,2,1,1);
	input20.printShapeInfo("0020 shapeINfo strides");
	input21.printShapeInfo("0021  shapeINfo strides");
	input22.printShapeInfo("0022 shapeINfo strides");

	auto  rank = 3;
	auto extra_pad_x = rank < 1 ? 0 : 32;
	auto pad_x = rank < 1 ? 0 : 4;
	auto pad_y = rank < 2 ? 0 : 4;

	auto auto00 = NDArrayFactory::createWithPadding('c', { bS,  K, N }, sd::DataType::FLOAT32, pad_y, pad_x + extra_pad_x, pad_y, pad_x);

	auto00.printShapeInfo("auto00 shapeINfo strides");

	auto auto01 = NDArrayFactory::createWithPadding('c', { 2,2,5,5}, sd::DataType::FLOAT32, pad_y, pad_x + extra_pad_x, pad_y, pad_x);

	auto01.printShapeInfo("auto01 shapeINfo strides");
#endif
	//testXX();
	//testYY();


	std::fstream filex("rnn.csv");
	std::vector<float> data;
	data.reserve(8 * 1024);
	if (filex.is_open())
	{
		std::string line;
		while (std::getline(filex, line, ';')) {
			data.push_back(std::stof(line));
		}

		filex.close();
	}
	else std::cout << "Unable to open file";
	ShapeDescriptor descriptor(DataTypeUtils::fromT<float>(), 'c', { 80,100 });

	if (descriptor.arrLength() != data.size()) {
		nd4j_printf("NDArrayFactory::create: data size [%li] doesn't match shape length [%lld]\n", data.size(), descriptor.arrLength());
		throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
	}

	 auto input1 = NDArrayFactory::create<float>('c', { 80,100 }, data);
	input1.printIndexedBuffer("Rnn");
	int fg;
	std::cin >> fg;
	return 0;
}
