

using namespace std;



#include <chrono>

#include <array>

#include <algorithm>
#include <numeric> 
#include <random> 
#include <iostream>
#include <utils.h>
#include <armcomputeUtils.h>
using namespace sd;


#define CHECK_CORRECTNESS 1

int bnch_cases[][8] = { 
	 {1,1,31,49,49,31, 0,0},
	 {1,1,49,31,49,31, 1,0},
	 {1,1,31,31,49,31, 1,1},
	 {1,1,24,31,49,31, 0,1},
	  
};


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



// evaluates sizes values and indexes using input and output arrays depending on data format
static inline void getSizesAndIndexesConv2d(const bool isNCHW, const int wFormat, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {
    getSizesAndIndexesConv2d(isNCHW, wFormat, input.getShapeInfo(), output.getShapeInfo(), bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
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

 

bool pool2d(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC=0) {
      
    bool isNCHW = !isNHWC;
    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
    input->printShapeInfo("input");
    output->printShapeInfo("output");


    arm_compute::NEPoolingLayer pool;
    auto data_layout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
    auto maxPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), data_layout, arm_compute::PadStrideInfo(sW, sH, pW, pH));
    nd4j_printf("-------maxPool---%d--\n", 0);
    auto in = getArmTensor(*input, data_layout); 
    auto out = getArmTensor(*output, data_layout); 

 
    pool.configure(&in, &out, maxPool);
    print_tensor(in, "In");
    pool.run(); // run function
    print_tensor(out, "After Run: Out");
    return true;
}



bool pool2d_b(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC = 0) {

    bool isNCHW = !isNHWC;
    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
    input->printShapeInfo("input");


    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
    auto inInfo  = getArmTensorInfo(*input, dataLayout);
    auto outInfo = getArmTensorInfo(*output, dataLayout);

    Arm_Tensor in{}; 
    Arm_Tensor out{}; 
    in.allocator()->init(inInfo);
    out.allocator()->init(outInfo);

    arm_compute::NEPoolingLayer pool;

    auto maxPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), dataLayout, arm_compute::PadStrideInfo(sW, sH, pW, pH));
  
    pool.configure(&in, &out, maxPool);

    if (in.info()->has_padding()) {
        //allocate and copy
        in.allocator()->allocate();
        //copy 
        copyToTensor(*input,in);

    }
    else {
        //import buffer
        void* buff = input->getBuffer();
        in.allocator()->import_memory(buff);
    }
    bool copyOut = false;
    if (out.info()->has_padding()) {
        //allocate and flag that we have to copy
        out.allocator()->allocate();
        copyOut = true;
    }
    else {
        //import
        void* buff = output->getBuffer();
        out.allocator()->import_memory(buff);
    }


    print_tensor(in, "In");
    pool.run(); // run function
    print_tensor(out, "After Run: Out");
    if (copyOut) {
        //copy the result
        copyFromTensor(out, *output);

    }
    output->printShapeInfo("output");
    output->printIndexedBuffer("output");
    return true;
}





bool pool2d_c(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC = 0) {

    bool isNCHW = !isNHWC;
    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
    input->printShapeInfo("input");
    input->printIndexedBuffer("input");

    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
     
    ArmFunction<arm_compute::NEPoolingLayer> pool;

    auto maxPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), dataLayout, arm_compute::PadStrideInfo(sW, sH, pW, pH));

    pool.configure(input,output, dataLayout, maxPool);
     
    pool.run(); // run function
  
    output->printShapeInfo("output");
    output->printIndexedBuffer("output");
    return true;
}


bool pool2d_auto_tensor(int iW, int iH, int oW, int  oH, int iD, int bS, int kH, int kW, int sH, int sW, int pH
    , int pW) {


    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d ", kH, kW, sH, sW, pH
        , pW);
    Arm_Tensor in{};
    const Arm_TensorShape in_shape(iW, iH, iD, bS);
    Arm_Tensor out{};
    const Arm_TensorShape out_shape(oW, oH, iD, bS);
    in.allocator()->init(Arm_TensorInfo(in_shape, 1, arm_compute::DataType::F32));
    out.allocator()->init(Arm_TensorInfo(out_shape, 1, arm_compute::DataType::F32));

    arm_compute::NEPoolingLayer pool;
    auto data_layout = in.info()->data_layout();
    auto maxPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), data_layout, arm_compute::PadStrideInfo(sW, sH, pW, pH));
    nd4j_printf("-------maxPool---%d--\n", 0);

 
    pool.configure(&in, &out, maxPool);

    in.allocator()->allocate();
    out.allocator()->allocate();
     

    print_tensor(in, "tensor In");
    pool.run(); // run function
    print_tensor(out, "tensor: Out");


    return true;
}

void test0() {
    const int bS = 2;
    const int iD = 1;
    const int iH = 24;
    const int iW = 24;
    const int kH = 3;
    const int kW = 3;
    const int sH = 1;
    const int sW = 1;
    const int pH = 0;
    const int pW = 0;
    const int dH = 1;
    const int dW = 1;
    const int oH = (iH - kH - (kH - 1) * (dH - 1) + 2 * pH) / sH + 1;     // output height
    const int oW = (iW - kW - (kW - 1) * (dW - 1) + 2 * pW) / sW + 1;     // output width

 

    pool2d_auto_tensor(iW, iH, oW, oH, iD, bS, kH, kW, sH, sW, pH, pW);
 
}

void test1() {
    const int bS = 2;
    const int iD = 1;
    const int iH = 24;
    const int iW = 24;
    const int kH = 3;
    const int kW = 3;
    const int sH = 1;
    const int sW = 1;
    const int pH = 0;
    const int pW = 0;
    const int dH = 1;
    const int dW = 1;
    const int oH = (iH - kH - (kH - 1) * (dH - 1) + 2 * pH) / sH + 1;     // output height
    const int oW = (iW - kW - (kW - 1) * (dW - 1) + 2 * pW) / sW + 1;     // output width

 
    auto x = NDArrayFactory::create<float>('c', { bS,iD,iH,iW });
    auto exp = NDArrayFactory::create<float>('c', { bS,iD,oH,oW });
    auto out = NDArrayFactory::create<float>('c', { bS,iD,oH,oW });
    fill_nd<float>(x, FILL_MODE::INC);


    //pool2d(&x, &out, kH, kW, sH, sW, pH, pW, dH, dW, 0);
    //pool2d_b(&x, &out, kH, kW, sH, sW, pH, pW, dH, dW, 0);
    pool2d_c(&x, &out, kH, kW, sH, sW, pH, pW, dH, dW, 0);
}

void test2(const NDArray &input) {
	input.printIndexedBuffer("Nd input");
    auto in = getArmTensor(input); 
    
    print_tensor(in, "In"); 
}

void test_pool() {
    auto x = NDArrayFactory::create<float>('c', { 2,2, 4, 4 });
    //auto exp = NDArrayFactory::create<float>('c', { 2, 2, 2, 2 }, { 11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f });
    auto out = NDArrayFactory::create<float>('c', { 2, 2, 2, 2 } );


    fill_nd<float>(x, FILL_MODE::INC);
     
    pool2d(&x, &out, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0);
  
    //exp.printIndexedBuffer("Expected");
    out.printIndexedBuffer("out");
 
}

void test_pool2() {
    auto x = NDArrayFactory::create<float>('c', { 2,  4, 4,2 });
    //auto exp = NDArrayFactory::create<float>('c', { 2, 2, 2, 2 }, { 11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f });
    auto out = NDArrayFactory::create<float>('c', { 2, 2, 2, 2 });


    fill_nd<float>(x, FILL_MODE::INC);



    pool2d(&x, &out, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1);

    //exp.printIndexedBuffer("Expected");
    out.printIndexedBuffer("out");
    

}

int main()
{
    const int bS = 2;
    const int iD = 1;
    const int iH = 24;
    const int iW = 24;
    Arm_Tensor in{};
    const Arm_TensorShape in_shape(iW, iH, iD, bS); 
    in.allocator()->init(Arm_TensorInfo(in_shape, 1, arm_compute::DataType::F32));
 

    //test0();
    test1();
    //test_pool();
    //test_pool2();
 
	return 0;
}
