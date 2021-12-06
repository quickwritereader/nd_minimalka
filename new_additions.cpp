
#include <fstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <numeric>
#include "NdArrayMinimal.h"
#include "LoopCoordsHelper.h"
#include "Threads.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace sd;



void perfRaw(NDArray& arrayX)
{
	Nd4jLong* shapeInfo = arrayX.getShapeInfo();
	Nd4jLong* strides = arrayX.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);

	auto strides_0 = strides[0];
	auto strides_1 = strides[1];
	auto strides_2 = strides[2];
	auto strides_3 = strides[3];
	auto strides_4 = strides[4];
	float* x = arrayX.bufferAsT<float>();
	for (int i = 0; i < bases[0]; i++)
		for (int j = 0; j < bases[1]; j++)
			for (int k = 0; k < bases[2]; k++)
				for (int l = 0; l < bases[3]; l++)
					for (int t = 0; t < bases[4]; t++)
	{
		x[i*strides_0+j*strides_1+k*strides_2+l*strides_3 +t*strides_4] = i; 
	}
}

template<int ConstRank>
void perfC(NDArray& arrayX)
{ 
	Nd4jLong* shapeInfo = arrayX.getShapeInfo();
	Nd4jLong* strides = arrayX.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	
	sd::CoordsState<ConstRank - 1> cst;

	size_t offset = sd::init_coords<ConstRank>(cst, 0, bases, strides);

	float* x = arrayX.bufferAsT<float>(); 
	for (int i = 0; i < arrayX.lengthOf(); i++)
	{
		x[offset] = i;
		offset = sd::inc_coords<ConstRank>(cst, offset);
	}
}

void perf0(NDArray& arrayX)
{
	Nd4jLong coords[MAX_RANK] = {};
	
	Nd4jLong* shapeInfo = arrayX.getShapeInfo();
	Nd4jLong* strides = arrayX.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);

	float* x = arrayX.bufferAsT<float>();

	size_t offset = 0;
	for (int i = 0; i < arrayX.lengthOf(); i++)
	{
		x[offset] = i;
		offset = sd::inc_coords(bases, strides, coords, offset, rank);
	}
}

FORCEINLINE size_t inc_coords2(const Nd4jLong* upstrides, const Nd4jLong* strides, Nd4jLong* coordsStrided, const size_t rank, const size_t skip = 0) {
	for (int i = rank - skip - 1; i >= 0; i--) {
		Nd4jLong val = coordsStrided[i] + strides[i];
		if (likely(val < upstrides[i])) {
			coordsStrided[i] = val;
			break;
		}
		else {
			coordsStrided[i] = 0;
		}
	}
	size_t last_offset = 0;
	for(int i=0;i<rank & (-4);i++)
	{
		last_offset += coordsStrided[i] + coordsStrided[i+1] + coordsStrided[i+2] + coordsStrided[i+3];
	}
	for (int i = rank & (-4); i < rank ; i++)
	{
		last_offset += coordsStrided[i];
	}
	
	return last_offset;
}

void perf1(NDArray& arrayX)
{
	Nd4jLong coords[MAX_RANK] = {};

	Nd4jLong upStrides[MAX_RANK] = {};

	Nd4jLong* shapeInfo = arrayX.getShapeInfo();
	Nd4jLong* strides = arrayX.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);

	for(int i =0; i<rank;i++)
	{
		upStrides[i] = bases[i] * strides[i];
	}

	float* x = arrayX.bufferAsT<float>();

	size_t offset = 0;
	for (int i = 0; i < arrayX.lengthOf(); i++)
	{
		x[offset] = i;
		offset = inc_coords2(upStrides, strides, coords, rank);
	}
}

int main() {
	NDArray arrayX=  NDArrayFactory::create<float>('c', { 23, 40, 100, 80, 33 });;
	perf0(arrayX);

	time_it<2, 2>(perfRaw, 0, arrayX);
	time_it<2, 2>(perfRaw, 0, arrayX);

	time_it<2, 2>(perfC<5>, 0, arrayX);
	time_it<2, 2>(perfC<5>, 0, arrayX);
	
	time_it<2,2>(perf0, 0, arrayX);

	time_it<2, 2>(perf0, 0, arrayX);

	time_it<2, 2>(perf1, 0, arrayX);
	time_it<2, 2>(perf1, 0, arrayX);
	//arrayX.printIndexedBuffer("arrayX");

	int l; std::cin >> l;

	return 0;
}
