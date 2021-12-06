

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




bool pool2d(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC = 0) {

    bool isNCHW = !isNHWC;
    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
    internal_print_nd_shape(*input,"input");
    internal_print_nd_shape(*output,"output");


    arm_compute::NEPoolingLayer pool;
    auto data_layout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
    auto poolPad = arm_compute::PadStrideInfo(sW, sH, pW, pH, arm_compute::DimensionRoundingType::CEIL);
    auto maxPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), data_layout, poolPad);
    nd4j_printf("-------maxPool---%d--\n", 0);
    auto in = getArmTensor(*input, data_layout);
    auto out = getArmTensor(*output, data_layout);


    pool.configure(&in, &out, maxPool);
    internal_print_arm_array(in, "In");
    pool.run(); // run function
    internal_print_arm_array(out, "After Run: Out");
    return true;
}



bool pool2d_b(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC = 0) {

    bool isNCHW = !isNHWC;
    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
    internal_print_nd_shape(*input,"input");


    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
    auto inInfo = getArmTensorInfo(*input, dataLayout);
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
        copyToTensor(*input, in);

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


    internal_print_arm_array(in, "In");
    pool.run(); // run function
    internal_print_arm_array(out, "After Run: Out");
    if (copyOut) {
        //copy the result
        copyFromTensor(out, *output);

    }
    internal_print_nd_shape(*output,"output");
    internal_print_nd_array(*output,"output");
    return true;
}





bool pool2d_c(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC = 0) {

    bool isNCHW = !isNHWC;
    nd4j_printf(" kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
    internal_print_nd_shape(*input,"input");
    internal_print_nd_array(*input,"input");

    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

    ArmFunction<arm_compute::NEPoolingLayer> pool;

    auto maxPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), dataLayout, arm_compute::PadStrideInfo(sW, sH, pW, pH));

    pool.configure(input, output, dataLayout, maxPool);

    pool.run(); // run function

    internal_print_nd_shape(*output,"output");
    internal_print_nd_array(*output,"output");
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


    internal_print_arm_array(in, "tensor In");
    pool.run(); // run function
    internal_print_arm_array(out, "tensor: Out");


    return true;
}

void pool_avg() {

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

void test2(const NDArray& input) {
    input.printIndexedBuffer("Nd input");
    auto in = getArmTensor(input);

    internal_print_arm_array(in, "In");
}

void test_pool() {
    auto x = NDArrayFactory::create<float>('c', { 2,2, 4, 4 });
    //auto exp = NDArrayFactory::create<float>('c', { 2, 2, 2, 2 }, { 11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f });
    auto out = NDArrayFactory::create<float>('c', { 2, 2, 2, 2 });


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


void pool2d_avg(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int extraParam0, int isNHWC = 0) {

    int isNCHW = isNHWC ? 0 : 1;
    int bS, iC, iH, iW, oC, oH, oW;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
    getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if (paddingMode)
        calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    bool exclude_padding = (extraParam0 == 0) ? true : false;

    nd4j_printf("avgpool kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d exclude pad %d \n", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0, exclude_padding ? 1 : 0);
    nd4j_printf("avgpool oH %d ow %d \n", oH, oW);

    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

    ArmFunction<arm_compute::NEPoolingLayer> pool;
    auto poolPad = arm_compute::PadStrideInfo(sW, sH, pW, pH, arm_compute::DimensionRoundingType::CEIL);
    auto avgPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, arm_compute::Size2D(kW, kH), dataLayout, poolPad, exclude_padding);

#if 0
    auto in = getArmTensor(*input, dataLayout);
    internal_print_arm_array(in, "original imported in");

#endif

    pool.configure(input, output, dataLayout, avgPool);

    pool.run(); // run function
    nd4j_printf("------------avg armcompute---%d-----\n", 0);
    return;
}


void test_pool_avg() {

    auto x = NDArrayFactory::create<float>('c', { 2, 2, 5, 5 });
    auto z = NDArrayFactory::create<float>('c', { 2, 2, 3, 3 });
    //, { 4.f, 6.f, 7.5f, 14.f, 16.f, 17.5f,  21.5f, 23.5f, 25.f, 29.f, 31.f, 32.5f, 39.f, 41.f, 42.5f, 46.5f, 48.5f, 50.f, 54.f, 56.f, 57.5f,  64.f, 66.f, 67.5f, 71.5f, 73.5f, 75.f, 79.f, 81.f, 82.5f, 89.f, 91.f, 92.5f,  96.5f, 98.5f, 100.f });

    fill_nd<float>(x, FILL_MODE::INC);

    x.printIndexedBuffer("x");

    //////////////////2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0
    pool2d_avg(&x, &z, 2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0);


    z.printIndexedBuffer("z");

}



static std::vector<Nd4jLong> expectWeightsShape(const int wFormat, const int kH, const int kW, const int iC, const int oC) {

    if (0 == wFormat)
        return std::vector<Nd4jLong>({ kH, kW, iC, oC });

    if (1 == wFormat)
        return std::vector<Nd4jLong>({ oC, iC, kH, kW });

    return std::vector<Nd4jLong>({ oC, kH, kW, iC });
}

static std::vector<Nd4jLong> expectWeightsShape(const int wFormat, const int kD, const int kH, const int kW, const int iC, const int oC) {

    if (0 == wFormat)
        return std::vector<Nd4jLong>({ kD, kH, kW, iC, oC });

    if (1 == wFormat)
        return std::vector<Nd4jLong>({ oC, iC, kD, kH, kW });

    return std::vector<Nd4jLong>({ oC, kD, kH, kW, iC });
}


void deconv2d(NDArray* input, NDArray* weights, NDArray* bias, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int isNHWC = 0, int wFormat = 0) {

    bool isNCHW = isNHWC == 0;

    // Calculate individual paddings
    // Calculate individual paddings
    unsigned int pad_left, pad_top, pad_right, pad_bottom;
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    std::vector<Nd4jLong> expectedWeightsShape = expectWeightsShape(wFormat, kH, kW, oC, iC);

    internal_print_nd_shape(*weights,"weights");
    for (auto& xx : expectedWeightsShape)
        nd4j_printf("%d ", (int)xx);
    nd4j_printf("%s \n", "");
    if (paddingMode) {
        //Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward pass
        calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);
    }
    pad_left = pW;
    pad_top = pH;
    pad_right = (iW - 1) * sW - oW + kW - pW;
    pad_bottom = (iH - 1) * sH - oH + kH - pH;
    //deconv2dMKLDNN(input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat);
#if 1
    nd4j_printf("deconv2d  bS = %d,  iH =%d, iW = %d,  oH=%d, oW=%d  kH=%d, kW=%d wformat=%d, iC =%d, , oC=%d\n",
        bS, iH, iW, oH, oW, kH, kW, wFormat, iC, oC
    );
    nd4j_printf("deconv2d kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d \n", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0);
#endif



    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
    //check weight input datalayout match
    bool dataLayoutMatch = (isNCHW && wFormat == 1) || (!isNCHW && wFormat == 2);
    arm_compute::PermutationVector permuteVector;

    //unlike in cov2d for weights iC and oC permutted : for example  {oC, iC, kH, kW}, {iC, oC, kH, kW}
    //but we need it normal way for arm
    if (!dataLayoutMatch) {
        //lets premute  
        if (wFormat == 0) {
            if (isNCHW) {
#if 1
                nd4j_printf("perm choise %d\n", 0);
#endif                    
                //reshape
                permuteVector = arm_compute::PermutationVector(2U, 3U, 0U, 1U);
            }
            else {
#if 1
                nd4j_printf("perm choise %d\n", 1);
#endif                         
                //reshape
                permuteVector = arm_compute::PermutationVector(0U, 2U, 3U, 1U);
            }
        }
        else if (wFormat == 1) {
#if 1
            nd4j_printf("perm choise %d\n", 2);
#endif                     
            permuteVector = arm_compute::PermutationVector(3U, 0U, 1U, 2U);
        }
        else {
#if 1
            nd4j_printf("perm choise %d\n", 3);
#endif                     
            permuteVector = arm_compute::PermutationVector(1U, 2U, 3U, 0U);
        }
    }
    else {
#if 1
        nd4j_printf("perm choise %d\n", 4);
#endif                 
        //set 0
        permuteVector.set_num_dimensions(0);
    }

    Arm_WeightsInfo wInfo(false, kW, kH, 1);
    //arm_compute::Size2D dilation(dW, dH);
    arm_compute::PadStrideInfo pad(sW, sH, pad_left, pad_right, pad_top, pad_bottom, arm_compute::DimensionRoundingType::FLOOR);
    ArmFunctionWeighted<arm_compute::NEDeconvolutionLayer> deconv;

    deconv.configure(input, weights, bias, output, dataLayout, permuteVector, pad);

    deconv.run(); // run function 
}

void testdeconv2d() {

    int bS = 2, oH = 4, oW = 4, oC = 5, iC = 10, kH = 2, kW = 2, sH = 1, sW = 1, pH = 0, pW = 0, dH = 1, dW = 1;
    int       iH = 3, iW = 3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat = 1;             // 1-NHWC, 0-NCHW
    int wFormat = 1;             // 0-[kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]


    auto input = NDArrayFactory::create<float>('c', { bS, iH, iW, iC, 5 });
    auto weights = NDArrayFactory::create<float>('c', { iC, oC, kH, kW });
    //auto bias = NDArrayFactory::create<float>('c', { oC } );
    auto output = NDArrayFactory::create<float>('c', { bS, oH, oW, oC });

    // fill_nd<float>(bias, FILL_MODE::INC);
    input = input.subarray({ NDIndex::all(), NDIndex::all(),NDIndex::all(), NDIndex::all(), NDIndex::point(0) });
    input.reshapei({ bS, iH, iW, iC }, 'c');

    fill_nd<float>(weights, FILL_MODE::INC);
    fill_nd<float>(input, FILL_MODE::INC);

    deconv2d(&input, &weights, nullptr, &output, kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, dataFormat, wFormat);

    output.printIndexedBuffer("output");
}






void pool2d_avg_universal(NDArray* input, NDArray* output, int kH, int kW, int sH, int sW, int pH
    , int pW, int dH, int dW, int paddingMode, int extraParam0, int isNHWC = 0) {

    int isNCHW = isNHWC ? 0 : 1;
    int bS, iC, iH, iW, oC, oH, oW;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
    getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if (paddingMode)
        calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    bool exclude_padding = (extraParam0 == 0) ? true : false;

    nd4j_printf("avgpool kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d exclude pad %d \n", kH, kW, sH, sW, pH
        , pW, dH, dW, paddingMode, isNCHW ? 1 : 0, exclude_padding ? 1 : 0);
    nd4j_printf("avgpool oH %d ow %d \n", oH, oW);

    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

    ArmFunctionUniversal<arm_compute::NEPoolingLayer> pool;
    auto poolPad = arm_compute::PadStrideInfo(sW, sH, pW, pH, arm_compute::DimensionRoundingType::CEIL);
    auto avgPool = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, arm_compute::Size2D(kW, kH), dataLayout, poolPad, exclude_padding);


    pool.configure(input, output, dataLayout, avgPool);

    pool.run(); // run function
    nd4j_printf("------------avg armcompute---%d-----\n", 0);
    return;
}


void test_pool_avg_uni(bool autoPadInput, bool autoPadOutput) {
    auto rank = 4;
    auto x = autoPadInput ? NDArrayFactory::createWithAutoPaddedStrides('c', { 2, 2, 5, 5 }, sd::DataType::FLOAT32) :
        NDArrayFactory::create('c', { 2, 2, 5, 5 }, sd::DataType::FLOAT32);
    auto z = autoPadOutput ? NDArrayFactory::createWithAutoPaddedStrides('c', { 2, 2, 3, 3 }, sd::DataType::FLOAT32) :
        NDArrayFactory::create('c', { 2, 2, 3, 3 }, sd::DataType::FLOAT32);

    fill_nd<float>(x, FILL_MODE::INC);

    x.printIndexedBuffer("x_auto");

    //////////////////2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0
    pool2d_avg_universal(&x, &z, 2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0);


    z.printIndexedBuffer("z");

}


void testXX(bool input_padded, bool output_padded) {
    auto rank = 4;
    int top, right, bottom, left;
    std::cout << "/////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////" << std::endl;
    std::tie(top, right, bottom, left) = getAutoPadding(rank);

    std::vector<Nd4jLong> input_paddings = { 0, 0, top + bottom, left + right } ;
    std::vector<Nd4jLong> input_offsets = { 0, 0, left, top } ;
    std::vector<Nd4jLong> output_paddings = { 0, 0, top + bottom, left + right } ;
    std::vector<Nd4jLong> output_offsets = { 0, 0, left, top };

    if (!input_padded) {
        input_paddings.clear();
        input_offsets.clear();
    }
    if (!output_padded) {
        output_paddings.clear();
        output_offsets.clear();
    }


    auto x = NDArrayFactory::create('c', { 2, 2, 5, 5 }, sd::DataType::FLOAT32, input_paddings, input_offsets );
    auto z = NDArrayFactory::create( 'c', { 2, 2, 3, 3 }, sd::DataType::FLOAT32, output_paddings, output_offsets);
    auto tret = getArmTensor(z);
    internal_print_arm_array(tret, "z");
    fill_nd<float>(x, FILL_MODE::INC);

    x.printIndexedBuffer("x_auto");

    //////////////////2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0
    pool2d_avg_universal(&x, &z, 2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0);


    z.printIndexedBuffer("z");
}


void test5(){

    int top, right, bottom, left;
    std::tie(top, right, bottom, left) = getAutoPadding(4);

    auto input = NDArrayFactory::create('c', { 2, 5, 5, 2 }, DataTypeUtils::fromT<float>(), { 0, 0, top + bottom, left + right }, { 0, 0, top, left });
   fill_nd<float>(input, FILL_MODE::INC);
 

}

int main()
{
    //
    int top, right, bottom, left;
    std::tie(top, right, bottom, left) = getAutoPadding(4);
    //auto x = NDArrayFactory::create('c', { 2, 2, 5, 5 }, sd::DataType::FLOAT32, { 0,0,top + bottom,left + right }, { 0,0,left,top });

    //internal_print_nd_shape(x,"x");

    //auto tret = getArmTensor(x);

    //Arm_Tensor t;
    //Arm_TensorInfo info, info1, info2, info3;
    //info.init_auto_padding(Arm_TensorShape{ 5,5,2,2 }, 1, Arm_DataType::F32);
    //t.allocator()->init(info);
    //t.allocator()->allocate();

    //internal_print_arm_array(tret, "tret");
    //internal_print_arm_array(t, "tensor");

    //testXX(false,false);
    //testXX(false, true);
    //testXX(true, false);
    //testXX(true, true);
    //auto x = NDArrayFactory::create('c', { 2, 3, 3, 2 }, DataTypeUtils::fromT<float>(), { 0, 0, top + bottom, left + right }, { 0, 0, top + bottom, left + right });
    //fill_nd<float>(x, FILL_MODE::INC);
    //auto tret = getArmTensor(x);
    //internal_print_arm_array(tret, "tret");
    test5();
    return 0;
}
