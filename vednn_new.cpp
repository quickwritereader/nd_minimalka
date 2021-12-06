

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

using namespace sd;

static inline void calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int paddingMode) {

    if (paddingMode == 0) {             // valid
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        // oW = (iW - (kW + (kW-1)*(dW-1)) + 2*pW)/sW + 1;
        oH = (iH - ((kH - 1) * dH + 1) + 2 * pH) / sH + 1;
        oW = (iW - ((kW - 1) * dW + 1) + 2 * pW) / sW + 1;
    }
    else if (paddingMode == 1) {       // same
        oH = (int)std::ceil(double(iH * 1. / sH));
        oW = (int)std::ceil(double(iW * 1. / sW));
    }
    else {                      // causal
        oH = (iH - 1) / sH + 1;     // 2*pH = (kH-1)*dH
        oW = (iW - 1) / sW + 1;
    }
}



static inline void calcPadding2D(int& pH, int& pW, int oH, int oW, int iH, int iW, int kH, int kW, int sH, int sW, int dH, int dW, const int paddingMode = 1 /* default is same mode*/) {

    if (paddingMode == 0)        // valid
        return;

    if (paddingMode == 1) {      // same

        const int eKH = (kH - 1) * dH + 1;
        const int eKW = (kW - 1) * dW + 1;

        pH = ((oH - 1) * sH + eKH - iH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
        pW = ((oW - 1) * sW + eKW - iW) / 2;
    }
    else {                      // causal
        pH = (kH - 1) * dH;
        pW = (kW - 1) * dW;
    }
}


// calculation of output height and width in 2D deconvolution procedure
static inline void calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int paddingMode) {

    if (paddingMode) {
        oH = sH * iH;
        oW = sW * iW;
    }
    else {
        const int ekH = (kH - 1) * dH + 1;
        const int ekW = (kW - 1) * dW + 1;

        oH = sH * (iH - 1) + ekH - 2 * pH;
        oW = sW * (iW - 1) + ekW - 2 * pW;
    }
}

static inline void getSizesAndIndexesConv2d(const bool isNCHW, const int wFormat, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {
    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (wFormat = 0), [oC, iC, kH, kW] (wFormat = 1), [oC, kH, kW, iC] (wFormat = 2)
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    if (0 == wFormat) {
        indWkH = 0; indWiC = 2; indWoC = 3;
    }
    else if (1 == wFormat) {
        indWkH = 2; indWiC = 1; indWoC = 0;
    }
    else {
        indWkH = 1; indWiC = 3; indWoC = 0;
    }

    if (!isNCHW) {
        indIOioC = 3; indIiH = 1; indOoH = 1;
    }
    else {
        indIOioC = 1; indIiH = 2; indOoH = 2;
    }

    bS = inShapeInfo[1];                          // batch size
    iC = inShapeInfo[indIOioC + 1];                 // input channels
    iH = inShapeInfo[indIiH + 1];                   // input height
    iW = inShapeInfo[indIiH + 2];                   // input width
    oC = outShapeInfo[indIOioC + 1];                // output channels
    oH = outShapeInfo[indOoH + 1];                  // output height
    oW = outShapeInfo[indOoH + 2];                  // output width
}


// evaluates sizes values and indexes using input and output arrays depending on data format
static inline void getSizesAndIndexesConv2d(const bool isNCHW, const int wFormat, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {
    getSizesAndIndexesConv2d(isNCHW, wFormat, input.getShapeInfo(), output.getShapeInfo(), bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
}


struct Conv {
        int kH;
        int kW;
        int sH;
        int sW;
        int pH;
        int pW;
        int dH;
        int dW;
        int paddingMode;
        bool isNCHW;
        int wFormat;

};

ostream&  operator<<(ostream& stream, const Conv& l) {
    stream <<"int kH =" << l.kH << ";\n int kW =" << l.kW <<"int sh =" << l.sH << ";\nint sW =" << l.sW << ";\nint pH =" << l.pH << ";\nint pW =" << l.pW
        << ";\n int dH = " << l.dH << ";\n int dW= " << l.dW << ";\n int paddingMode =" << l.paddingMode
        << ";\n int isNCHW = " << l.isNCHW << ";\n int wFormat= " << l.wFormat << std::endl;
    return stream;
}

void conv2d(NDArray* input,
    NDArray* weights,
    NDArray* bias,
    NDArray* output,
    const Conv &args) {
    std::cout << args << std::endl;

}



int main()
{
    //shapeInfo input : [4, 1, 2, 5, 4, 40, 20, 4, 1, 8192, 1, 99]
    //shapeInfo weights : [4, 2, 2, 2, 3, 2, 1, 4, 8, 8192, 0, 99]
    //shapeInfo output : [4, 1, 3, 5, 4, 60, 20, 4, 1, 8192, 1, 99]
    Conv c;
    c.kH = 2;
    c.kW = 2;
    c.sH = 1;
    c.sW = 1;
    c.pH = 0;
    c.pW = 0;
    c.dH = 1;
    c.dW = 1;
    c.paddingMode = 1;
    c.isNCHW = 1;
    c.wFormat = 0;


    auto input = NDArrayFactory::create<float>('c', { 1,2,5,4 });
    auto weights = NDArrayFactory::create<float>('c', { 2, 2, 2, 3 });
    auto out = NDArrayFactory::create<float>('c', { 1, 3, 5, 4 });
    fill_nd<float>(input, FILL_MODE::INC);
    fill_nd<float>(weights, FILL_MODE::INC);

	input.printIndexedBuffer("input");

    conv2d(&input, &weights, nullptr, &out, c);

	int fg;
 
	std::cin >> fg;
	return 0;
}
