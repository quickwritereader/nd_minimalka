#include <fstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include "common.h"
#include "NdArrayMinimal.h"
#include "LoopCoordsHelper.h"
#include "Threads.h"
#include "utils.h"
using namespace sd;

std::vector<float> loadFile(std::string filename)
{
	std::fstream filex(filename);
	std::vector<float> data;
	data.reserve(8 * 1024);
	int colSize = 0;
	bool countedCol = false;
	int countY = 0;
	if (filex.is_open())
	{
		std::string numb;
		while (std::getline(filex, numb, ';'))
		{
			auto pos = numb.find("\n");
			if (pos != std::string::npos)
			{
				//std::cout<<"___"<<numb<< "___";
				numb = numb.erase(pos, 1);
				//std::cout<<"___"<<numb<< "__"<<numb.size()<<std::endl;
				if (!countedCol) colSize = countY;
				countedCol = true;
			}
			++countY;
			//std::cout<<numb<<" ";
			if (numb.size() > 0) data.push_back(std::stof(numb));
		}

		filex.close();
	}
	else std::cout << "Unable to open file";
	std::cout << data.size() << std::endl;
	std::cout << data.data() << std::endl;
	std::cout << data.size() / colSize << " " << colSize << std::endl;
	return data;
}

template <int constRank = 2, bool IsFortran = false>
NDArray Word2LabelSeq(std::string word, std::string classes)
{
	constexpr int skip = 0;
	auto dim = std::vector<Nd4jLong>(constRank, 1); //calls ctor(count, val)
	dim[constRank - 1] = word.size();
	auto res = NDArrayFactory::create<int>(IsFortran ? 'c' : 'f', dim);
	auto x = static_cast<int*>(res.buffer());
	CoordsState<constRank - 1> cst;
	int wi = 0;
	size_t offset = sd::init_coords<constRank, skip, IsFortran>(cst, 0, dim.data(), res.stridesOf());
	size_t loop_count = 1 * word.size();
	for (size_t i = 0; i < loop_count; i++)
	{
		x[offset] = classes.find(word[wi++]);
		offset = sd::inc_coords<constRank, skip, IsFortran>(cst, offset);
	}
	return res;
}

#include <numeric>

template <typename T, int constRank = 2>
void softmax(NDArray& input, bool makeLog = false)
{
	constexpr int skip = 0;
	CoordsState<constRank> cst;
	auto x_shapeInfo = input.shapeInfo();
	T* x = input.bufferAsT<T>();
	auto bases = &(x_shapeInfo[1]);
	auto x_strides = &(x_shapeInfo[constRank + 1]);
	size_t offset = sd::init_coords<constRank, skip, false>(cst, 0, bases, x_strides);
	size_t loop_count = 1;
	for (Nd4jLong i = 0; i < constRank - 1; i++)
	{
		loop_count *= bases[i];
	}
	std::cout << loop_count << " , " << LAST_NUM(cst, 0) << " , " << STRIDE(cst, 0);
	std::cout << " , " << LAST_NUM(cst, 1) << " , " << STRIDE(cst, 1) << std::endl;
	auto el_stride = STRIDE(cst, constRank - 1);
	auto last_index = LAST_NUM(cst, constRank - 1);
	T minVal = std::numeric_limits<T>::min();
	for (size_t i = 0; i < loop_count; i++)
	{
		T* lx = x + offset;
		auto maxVal = lx[0];
		for (size_t j = 0; j <= last_index; j++)
		{
			maxVal = lx[j * el_stride] > maxVal ? lx[j * el_stride] : maxVal;
		}

		T denom_sum = 0;
		for (size_t j = 0; j <= last_index; j++)
		{
			T val = std::exp(lx[j * el_stride] - maxVal);
			denom_sum += val;
			lx[j * el_stride] = val;
		}
		if (!makeLog)
		{
			for (size_t j = 0; j <= last_index; j++)
			{
				lx[j * el_stride] = lx[j * el_stride] / denom_sum;
				if (lx[j * el_stride] < minVal) lx[j * el_stride] = minVal;


				//std::cout<<std::scientific<<lx[j*el_stride]<<", ";
			}
		}
		else
		{
			for (size_t j = 0; j <= last_index; j++)
			{
				lx[j * el_stride] = std::log(lx[j * el_stride]) - std::log(denom_sum);
				//std::cout<<std::scientific<<lx[j*el_stride]<<", ";
			}
		}
		//std::cout<<std::endl;
		offset = sd::inc_coords<constRank, skip, false>(cst, offset);
	}
}

float forwardCompute_C(const NDArray& input, const NDArray& targetLabels, int blankIndex)
{
	// lenT  - input length of T
	// lenS  - lenght of sequence
	// lenSB - length with blanks 
	auto lenS = targetLabels.shapeOf()[1];
	auto lenT = input.shapeOf()[0];
	auto lenSB = 2 * lenS + 1;
	std::cout << lenS << "," << lenSB << "," << lenT << std::endl;
	//create temp Array for holding alphaArr [lenT,lenSB]
	auto alphaArr = NDArrayFactory::create<float>('c', {lenT, lenSB});

	// get probability bufer and tagetLabels buffer
	float* p = input.bufferAsT<float>();
	int* lbl = targetLabels.bufferAsT<int>();
	float logLoss = 0;


	float* alphaPtr = alphaArr.bufferAsT<float>();

	auto incP = input.stridesOf()[0];
	auto incA = alphaArr.stridesOf()[0];

	//initialize alphas at t=0
	alphaPtr[0] = p[blankIndex];
	auto targetInd = lbl[0];
	alphaPtr[1] = p[targetInd];

	//move to the next frame
	auto alphaPrevPtr = alphaPtr;
	alphaPtr += incA;
	auto startX = lenSB - 2 * lenT;
	//process the rest
	for (auto t = 1; t < lenT; t++)
	{
		//start = max(0,L-2*(T-t))
		auto s = startX - 2 * t;
		s = s > 0 ? s : 0;
		for (; s < lenSB; s++)
		{
			auto ind = s / 2; //our real index
			//we forced blanks for even indexes
			auto currentInd = (s % 2 == 0) ? blankIndex : lbl[ind];
			// {t-1,s}
			auto alphaS = alphaPrevPtr[s];
			auto alphaS_1 = s > 0 ? alphaPrevPtr[s - 1] : 0;
			// if blank or the same as previous
			auto alphaS_2 = (s > 1 && currentInd != blankIndex && currentInd != lbl[ind - 1]) ? alphaPrevPtr[s - 2] : 0;

			alphaPtr[s] = (alphaS + alphaS_1 + alphaS_2) * p[currentInd];
		}

		//store t-1 alpha Ptr
		alphaPrevPtr = alphaPtr;
		p += incP;
		alphaPtr += incA;
	}


	logLoss = -std::log(alphaPrevPtr[lenSB - 1] + alphaPrevPtr[lenSB - 2]);

	alphaArr.printIndexedBuffer("alphas");
	return logLoss;
}

#include <tuple>

//choose ptr[index*element_stride] 
template <bool Strided, typename Type>
std::enable_if_t<Strided == true, Type&>
element(Type* ptr, int index, int element_stride)
{
	return ptr[index * element_stride];
}

//choose ptr[index] assuming element_stride is 1
template <bool Strided, typename Type>
std::enable_if_t<Strided == false, Type&>
element(Type* ptr, int index, int element_stride)
{
	return ptr[index];
}

template <bool IsLogPStrided = false, bool IsLblStrided = false, typename Type, typename IndexType>
Type forward(Type* alphaPtr, const Nd4jLong& incA, Type* logP, const Nd4jLong& incP, IndexType* lbl,
             const Nd4jLong& lenSB, const Nd4jLong& lenT, const int& blankIndex, int elwiseP = 1, int elwiseS = 1)
{
	constexpr Type negInf = -std::numeric_limits<Type>::infinity();
	//initialize alphas at t=0 
	alphaPtr[0] = element<IsLogPStrided>(logP, blankIndex, elwiseP);
	//alphaPtr[1] =logP[lbl[0]];
	alphaPtr[1] = element<IsLogPStrided>(logP, *lbl, elwiseP);
	//the rest initialization was skipped 
	//as its assumed the array already were initialized with negative infinity
	//move to the next frame
	Type* alphaPrevPtr = alphaPtr;
	alphaPtr += incA;
	logP += incP;

	auto startX = lenSB - 2 * lenT;
	//process the rest
	for (auto t = 1; t < lenT; t++)
	{
		//start = max(0,L-2*(T-t))
		auto s = startX + 2 * t;
		s = s > 0 ? s : 0;
		for (; s < lenSB; s++)
		{
			auto ind = s / 2; //our real index
			//we force blanks for even indexes
			//strided version of lbl[ind] => element<IsLblStrided>(lbl, ind, elwiseS)
			auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);
			// {t-1,s}
			Type alphaS = alphaPrevPtr[s];
			Type alphaS_1 = s > 0 ? alphaPrevPtr[s - 1] : negInf;
			Type cMax = std::max(alphaS, alphaS_1);
			//logP[currentInd] or logP[currentInd*elwiseP] 
			auto currentProb = element<IsLogPStrided>(logP, currentInd, elwiseP);
			// if blank or the same as previous
			if (s > 1 && currentInd != blankIndex && currentInd != element<IsLblStrided>(lbl, ind - 1, elwiseS))
			{
				Type alphaS_2 = alphaPrevPtr[s - 2];
				cMax = std::max(cMax, alphaS_2);
				if (cMax == negInf) cMax = 0;
				alphaPtr[s] = std::log(std::exp(alphaS - cMax) + std::exp(alphaS_1 - cMax) + std::exp(alphaS_2 - cMax))
					+ cMax + currentProb;
			}
			else
			{
				if (cMax == negInf) cMax = 0;
				alphaPtr[s] = std::log(std::exp(alphaS - cMax) + std::exp(alphaS_1 - cMax)) + cMax + currentProb;
			}
		}

		//store t-1 alpha Ptr
		alphaPrevPtr = alphaPtr;
		logP += incP;
		alphaPtr += incA;
	}
	auto logP0 = alphaPrevPtr[lenSB - 1];
	auto logP1 = alphaPrevPtr[lenSB - 2];
	auto cMax = std::max(logP0, logP1);
	return -(std::log(std::exp(logP0 - cMax) + std::exp(logP1 - cMax)) + cMax);
}

#define CALCULATE_ALL_IN_ONE_FRAME_LOOP

template <bool IsLogPStrided = false, bool IsLblStrided = false, bool isGradStrided = false, typename Type, typename
          IndexType = int>
void backwardAndGrad(Type forwardLogLoss, Type* alphaPtr, Type* bettaPtr, int incA, Type* logP, int incP, Type* gradPtr,
                     int incG, IndexType* lbl,
                     const Nd4jLong& lenS, const Nd4jLong& lenT, const Nd4jLong& lenK, const int& blankIndex,
                     int elwiseP = 1, int elwiseS = 1, int elwiseG = 1)
{
	constexpr Type negInf = -std::numeric_limits<Type>::infinity();
	Nd4jLong lenSB = 2 * lenS + 1;
	auto origBetta = bettaPtr;
	auto origLogP = logP;
	//move to the last frame
	bettaPtr += (lenT - 1) * incA;
	logP += (lenT - 1) * incP;

	//initialize bettas at t=lenT
	bettaPtr[lenSB - 1] = element<IsLogPStrided>(logP, blankIndex, elwiseP);
	auto lblIndex = element<IsLblStrided>(lbl, lenS - 1, elwiseS);
	bettaPtr[lenSB - 2] = element<IsLogPStrided>(logP, lblIndex, elwiseP); // logP[lbl[lenS - 1]];


#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
	//move to the last
	gradPtr += (lenT - 1) * incG;
	alphaPtr += (lenT - 1) * incA;
	for (auto s = lenSB - 1; s >= 0; s--)
	{
		auto ind = s / 2; //our real index
		//we forced blanks for even indexes
		auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);
		//alpha(s)*betta(s) in log scale but still store in alpha to save memory
		auto alphaBettaS = alphaPtr[s] + bettaPtr[s];

		//sum  (alpha(s)*betta(s) ) over real indexes
		auto& currentGrad = element<isGradStrided>(gradPtr, currentInd, elwiseG); // gradPtr[currentInd];
		if (currentGrad == negInf)
		{
			currentGrad = alphaBettaS;
		}
		else
		{
			Type cMax = std::max(currentGrad, alphaBettaS);
			currentGrad = std::log(std::exp(currentGrad - cMax) + std::exp(alphaBettaS - cMax)) + cMax;
		}
	}
	for (int k = 0; k < lenK; k++)
	{
		//compute the rest grad 

		// prob(t,k) - grad(k) / ((prob(t,k)*Z) )

		// p2= grad(k) / (prob(t,k)*Z )
		//in logscale . plus we have Z as -logLoss
		// auto p2 = std::exp(gradPtr[k] + forwardLogLoss - logP[k]);
		// gradPtr[k] = std::exp(logP[k]) - p2;
		auto currentProb = element<IsLogPStrided>(logP, k, elwiseP);
		auto& currentGrad = element<isGradStrided>(gradPtr, k, elwiseG);
		auto p2 = std::exp(currentGrad + forwardLogLoss - currentProb);
		currentGrad = std::exp(currentProb) - p2;
	}
	gradPtr -= incG;
	alphaPtr -= incA;
#endif

	auto bettaPrevPtr = bettaPtr;
	bettaPtr -= incA;
	logP -= incP;
	//process the rest
	for (auto t = lenT - 2; t >= 0; t--)
	{
#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
		auto end = lenSB - 1;
#else
		auto end = std::min(2 * t + 2, lenSB - 1);
#endif
		for (auto s = end; s >= 0; s--)
		{
			auto ind = s / 2; //our real index
			//we forced blanks for even indexes
			auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS); //lbl[ind];
			// {t-1,s}
			Type bettaS = bettaPrevPtr[s];
			Type bettaS_1 = s < lenSB - 1 ? bettaPrevPtr[s + 1] : negInf;
			Type cMax = std::max(bettaS, bettaS_1);
			//logP[currentInd] 
			auto currentProb = element<IsLogPStrided>(logP, currentInd, elwiseP);
			// if blank or the same as previous
			if (s < lenSB - 2 && currentInd != blankIndex && currentInd != element<IsLblStrided>(lbl, ind + 1, elwiseS))
			{
				Type bettaS_2 = bettaPrevPtr[s + 2];
				cMax = std::max(cMax, bettaS_2);
				if (cMax == negInf) cMax = 0;
				bettaPtr[s] = std::log(std::exp(bettaS - cMax) + std::exp(bettaS_1 - cMax) + std::exp(bettaS_2 - cMax))
					+ cMax + currentProb;
			}
			else
			{
				if (cMax == negInf) cMax = 0;
				bettaPtr[s] = std::log(std::exp(bettaS - cMax) + std::exp(bettaS_1 - cMax)) + cMax + currentProb;
			}

#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
			//alpha(s)*betta(s) in log scale but still store in alpha to save memory
			auto alphaBettaS = alphaPtr[s] + bettaPtr[s];

			//sum  (alpha(s)*betta(s) ) over real indexes
			auto& currentGrad = element<isGradStrided>(gradPtr, currentInd, elwiseG); // gradPtr[currentInd];
			if (currentGrad == negInf)
			{
				currentGrad = alphaBettaS;
			}
			else
			{
				Type cMax = std::max(currentGrad, alphaBettaS);
				currentGrad = std::log(std::exp(currentGrad - cMax) + std::exp(alphaBettaS - cMax)) + cMax;
			}

#endif
		}


#if defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
		for (int k = 0; k < lenK; k++)
		{
			//compute the rest grad 

			// prob(t,k) - grad(k) / ((prob(t,k)*Z) )

			// p2= grad(k) / (prob(t,k)*Z )
			//in logscale . plus we have Z as -logLoss
			// auto p2 = std::exp(gradPtr[k] + forwardLogLoss - logP[k]);
			// gradPtr[k] = std::exp(logP[k]) - p2;
			auto currentProb = element<IsLogPStrided>(logP, k, elwiseP);
			auto& currentGrad = element<isGradStrided>(gradPtr, k, elwiseG);
			auto p2 = std::exp(currentGrad + forwardLogLoss - currentProb);
			currentGrad = std::exp(currentProb) - p2;
		}
		alphaPtr -= incA;
		gradPtr -= incG;
#endif


		bettaPrevPtr = bettaPtr;
		bettaPtr -= incA;
		logP -= incP;
	}

	auto logBP0 = bettaPrevPtr[0];
	auto logBP1 = bettaPrevPtr[1];
	auto bcMax = std::max(logBP0, logBP1);
	auto blogLoss = -(std::log(std::exp(logBP0 - bcMax) + std::exp(logBP1 - bcMax)) + bcMax);

#if !defined(CALCULATE_ALL_IN_ONE_FRAME_LOOP)
	//alpha*betta 
	bettaPtr = origBetta;
	logP = origLogP;

	for (int t = 0; t < lenT; t++) {

		for (int s = 0; s < lenSB; s++) {
			auto ind = s / 2; //our real index
		   //we forced blanks for even indexes
			auto currentInd = (s % 2 == 0) ? blankIndex : element<IsLblStrided>(lbl, ind, elwiseS);//lbl[ind];
			//alpha(s)*betta(s) in log scale but still store in alpha to save memory
			auto alphaBettaS = alphaPtr[s] + bettaPtr[s];

			//sum  (alpha(s)*betta(s) ) over real indexes
			auto& currentGrad = element<isGradStrided>(gradPtr, currentInd, elwiseG);// gradPtr[currentInd];
			if (currentGrad == negInf) {
				currentGrad = alphaBettaS;
			}
			else {
				Type cMax = std::max(currentGrad, alphaBettaS);
				currentGrad = std::log(std::exp(currentGrad - cMax) + std::exp(alphaBettaS - cMax)) + cMax;
			}
			//alphaPtr[s] = alphaBettaS; 

		} 

        PRAGMA_OMP_SIMD
		for (int k = 0; k < lenK; k++) {
			//compute the rest grad 

		   // prob(t,k) - grad(k) / ((prob(t,k)*Z) )

		   // p2= grad(k) / (prob(t,k)*Z )
		   //in logscale . plus we have Z as -logLoss
			// auto p2 = std::exp(gradPtr[k] + forwardLogLoss - logP[k]);
			// gradPtr[k] = std::exp(logP[k]) - p2;
			auto currentProb = element<IsLogPStrided>(logP, k, elwiseP);
			auto& currentGrad = element<isGradStrided>(gradPtr, k, elwiseG);
			auto p2 = std::exp(currentGrad + forwardLogLoss - currentProb);
			currentGrad = std::exp(currentProb) - p2;

		}


		gradPtr += incG; 
		bettaPtr += incA;
		alphaPtr += incA; 
		logP += incP;
	}
#endif
}

/**
 * Calculates ctc loss and fills gradients
 * @param logP logits matrix(lenT,lenK) pointer (log soft max input of rnn) 
 * @param incP stride of logits for the next time frame
 * @param gradPtr gradient for output
 * @param incG  stride of the gradient for the next time frame 
 * @param lbl target label 
 * @param lenT frame length
 * @param lenK class length
 * @param lenS target label length
 * @param blankIndex index of the blank label in logit class
*/
template <bool IsLogPStrided = true, bool IsLblStrided = true, bool IsGradStrided = true, typename Type, typename
          IndexType = int>
Type unitLossAndGrad(Type* logP, int incP, Type* gradPtr, int incG, IndexType* lbl, int lenT, int lenK, int lenS,
                     int blankIndex,
                     int elwiseP = 1, int elwiseS = 1, int elwiseG = 1)
{
	auto lenSB = 2 * lenS + 1;
	//create temp Array for holding bettaArr [lenT,lenSB]
	//create temp Array for holding alphaArr [lenT,lenSB] 
	int bufferC = gradPtr ? 2 : 1;
	NDArray bufferArr = NDArrayFactory::create<Type>('c', {bufferC, lenT, lenSB});
	auto bufferPtr = bufferArr.bufferAsT<Type>();
	auto incA = bufferArr.stridesOf()[1];
	auto bettaBufferPtr = bufferPtr + bufferArr.stridesOf()[0];
	constexpr auto negInf = -std::numeric_limits<Type>::infinity();

#if 1
	if (gradPtr)
	{
		if (elwiseG == 1)
		{
			PRAGMA_OMP_SIMD
			for (int i = 0; i < lenK * lenT; i++)
			{
				gradPtr[i] = negInf;
			}
		}
		else
		{
			auto tempPtr = gradPtr;
			for (int i = 0; i < lenT; i++)
			{
				for (int j = 0; j < lenK; j++)
					element<false>(tempPtr, j, elwiseG) = negInf;
				tempPtr += incG;
			}
		}
	}
#endif

	// set all vals to neginf
	PRAGMA_OMP_SIMD
	for (int i = 0; i < bufferC * lenSB * lenT; i++)
	{
		bufferPtr[i] = negInf;
	}

	//forward 
	Type logLoss = forward<IsLogPStrided, IsLblStrided>(bufferPtr, incA, logP, incP, lbl, lenSB, lenT, blankIndex,
	                                                    elwiseP, elwiseS);
	//backward and gradient if gradptr supplied
	if (gradPtr) backwardAndGrad<IsLogPStrided, IsLblStrided, IsGradStrided>(
		logLoss, bufferPtr, bettaBufferPtr, incA, logP, incP, gradPtr, incG, lbl, lenS, lenT, lenK, blankIndex, elwiseP,
		elwiseS, elwiseG);
	return logLoss;
}

template <typename Type, typename IndexType = int>
void
ctc_loss(const NDArray& logInput, const NDArray& targetLabels, const NDArray& logInputLengths,
         const NDArray& targetLabelLengths, NDArray& logLosses, NDArray& gradients, int blankIndex)
{
	// lenT  - input length of T
	// lenS  - lenght of sequence
	// lenSB - length with blanks
	auto lenBatch = logInput.shapeOf()[0];

	auto maxLenT = logInput.shapeOf()[1];
	auto lenK = logInput.shapeOf()[2];
	auto maxLenS = targetLabels.shapeOf()[1];

	// get probability bufer and tagetLabels buffer
	auto logP = logInput.bufferAsT<Type>();
	auto lblPtr = targetLabels.bufferAsT<IndexType>();
	auto logLossPtr = logLosses.bufferAsT<Type>();

	auto lenTPtr = logInputLengths.bufferAsT<IndexType>();
	auto lenSPtr = targetLabelLengths.bufferAsT<IndexType>();

	auto batchLbl = targetLabelLengths.stridesOf()[0];
	auto batchP = logInput.stridesOf()[0];
	auto incP = logInput.stridesOf()[1];

	auto elwiseLL = logLosses.stridesOf()[0];
	auto elwiseSLen = targetLabelLengths.stridesOf()[0];
	auto elwiseT = logInputLengths.stridesOf()[0];
	auto elwiseS = targetLabels.stridesOf()[1];
	auto elwiseP = logInput.stridesOf()[2];

	//make blankIndex last class if its incorrect
	if (blankIndex > maxLenS || blankIndex < 0) blankIndex = maxLenS - 1;
	
	auto func = [logP, batchP, incP, elwiseP, lenK, lenTPtr, lenSPtr, logLossPtr, lblPtr, maxLenT, maxLenS,
			batchLbl, blankIndex, elwiseT, elwiseLL, elwiseSLen, elwiseS, &gradients]
	(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void
	{
		Type* gradPtr = nullptr;
		int batchG, incG, elwiseG;
		if (!gradients.isEmpty())
		{
			batchG = gradients.stridesOf()[0];
			incG = gradients.stridesOf()[1];
			elwiseG = gradients.stridesOf()[2];
			gradPtr = gradients.bufferAsT<Type>() + start * batchG;
		}
		auto logPtr = logP + start * batchP;
		auto tempLblPtr = lblPtr + start * batchLbl;

		if (elwiseP == 1 && elwiseS == 1 && elwiseG == 1)
		{
			//choose ews one
			for (int batchIndex = start; batchIndex < stop; batchIndex += increment)
			{
				auto lenT = lenTPtr[batchIndex * elwiseT];
				auto lenS = lenSPtr[batchIndex * elwiseSLen];
				lenT = lenT > maxLenT ? maxLenT : lenT;
				lenS = lenS > maxLenS ? maxLenS : lenS;
				if (lenS <= 0 || lenT <= 0)
				{
					logLossPtr[batchIndex * elwiseLL] = -std::numeric_limits<Type>::infinity();
				}
				else
				{
					if (lenS > lenT) lenS = lenT;
					logLossPtr[batchIndex * elwiseLL] = unitLossAndGrad<false, false, false>(
						logPtr, incP, gradPtr, incG, tempLblPtr, lenT, lenK, lenS, blankIndex);
				}
				if (gradPtr) gradPtr += batchG;
				logPtr += batchP;
				tempLblPtr += batchLbl;
			}
		}
		else
		{
			//slow strided case for all 3
			for (int batchIndex = start; batchIndex < stop; batchIndex += increment)
			{
				auto lenT = lenTPtr[batchIndex * elwiseT];
				auto lenS = lenSPtr[batchIndex * elwiseSLen];
				lenT = lenT > maxLenT ? maxLenT : lenT;
				lenS = lenS > maxLenS ? maxLenS : lenS;
				if (lenS <= 0 || lenT <= 0)
				{
					logLossPtr[batchIndex * elwiseLL] = -std::numeric_limits<Type>::infinity();
				}
				else
				{
					if (lenS > lenT) lenS = lenT;
					logLossPtr[batchIndex * elwiseLL] = unitLossAndGrad(logPtr, incP, gradPtr, incG, tempLblPtr, lenT,
					                                                    lenK, lenS, blankIndex, elwiseP, elwiseS,
					                                                    elwiseG);
				}
				if (gradPtr) gradPtr += batchG;
				logPtr += batchP;
				tempLblPtr += batchLbl;
			}
		}
	};
	samediff::Threads::parallel_for(func, 0, lenBatch, 1);
}

 

int main()
{
	std::vector<float> data = loadFile("rnn.csv");

	auto logInput = NDArrayFactory::create<float>('c', {1, 100, 80}, data);
	//softmax<float>(input1);
	softmax<float, 3>(logInput, true);
	logInput.printShapeInfo("Rnn");

	std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";

	// [72, 60, 57,  0, 58, 53, 63, 57,  0, 58, 70, 61, 57, 66, 56,  0,
	//         67, 58,  0, 72, 60, 57,  0, 58, 53, 65, 61, 64, 77, 10,  0, 64,
	//         61, 63, 57,  0, 72, 60, 57]
	std::string words = "the fake friend of the family, like the";
	auto labels = Word2LabelSeq<2>(words, classes);

	//auto resProb = forwardCompute_C(input1, labels, 79);
	//std::cout << resProb << std::endl;

	float prob;
	bool onlyForward = false;
	NDArray gradient = onlyForward ? NDArrayFactory::empty<float>() : NDArrayFactory::create<float>('c', {1, 100, 80});
	NDArray lenTArr = NDArrayFactory::create<int>('c', {1}, {100});
	NDArray lenSArr = NDArrayFactory::create<int>('c', {1}, {static_cast<int>(labels.shapeOf()[1])});
	NDArray llArr = NDArrayFactory::create<float>('c', {1});
	ctc_loss<float>(logInput, labels, lenTArr, lenSArr, llArr, gradient, 79);
	std::cout << llArr << std::endl;
	std::cout << gradient << std::endl;

	time_it<100, 100>(ctc_loss<float>, 0, logInput, labels, lenTArr, lenSArr, llArr, gradient, 79);
	int hk;
	std::cin >> hk;
	return 0;
}
