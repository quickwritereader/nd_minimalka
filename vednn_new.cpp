

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
#include <stdio.h>
#include <dnnl.hpp>

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

  static std::vector<Nd4jLong> expectWeightsShape(const int wFormat, const int kH, const int kW, const int iC,
                                                      const int oC) {
    if (0 == wFormat) return std::vector<Nd4jLong>({kH, kW, iC, oC});

    if (1 == wFormat) return std::vector<Nd4jLong>({oC, iC, kH, kW});

    return std::vector<Nd4jLong>({oC, kH, kW, iC});
  }


void setBlockStrides(const NDArray& array, dnnl::memory::desc& mklMd, const std::vector<int>& permut= {}) {
  if (array.ews() != 1 || (array.rankOf() > 3 && array.ordering() == 'f') || !permut.empty()) {
    mklMd.data.format_kind = dnnl_blocked;  // overrides format

    if (permut.empty())
      for (auto i = 0; i < array.rankOf(); ++i) mklMd.data.format_desc.blocking.strides[i] = array.stridesOf()[i];
    else {
      if (array.rankOf() != permut.size())
        throw std::invalid_argument("mkldnnUtils::setBlockStrides: size of permut vector is not equal to array rank !");
      for (auto i = 0; i < array.rankOf(); ++i) mklMd.data.format_desc.blocking.strides[i] = array.stridesOf()[(permut[i])];
    }
  }
}

dnnl::memory loadDataToMklStream(const NDArray& array, const dnnl::engine& engine, const dnnl::stream& stream,
                                 const dnnl::memory::desc& user_md, const dnnl::memory::desc& primitive_md,
                                 dnnl::memory& arg) {
  array.printIndexedBuffer("array");
  array.printShapeInfo("array");
  auto user_mem = dnnl::memory(user_md, engine, const_cast<NDArray&>(array).buffer());
  const bool bReorder = primitive_md != user_mem.get_desc();
  printf("%s %d --> %d\n",__FILE__,__LINE__,(int)bReorder);
  auto mkl_mem = bReorder ? dnnl::memory(primitive_md, engine) : user_mem;
  if (bReorder) dnnl::reorder(user_mem, mkl_mem).execute(stream, user_mem, mkl_mem);
  arg = mkl_mem;
  return user_mem;
}

dnnl::engine& getEngine(void* ptr) {
  auto eng = reinterpret_cast<dnnl::engine*>(ptr);
  return *eng;
}

static void conv2dMKLDNN(const NDArray *input, const NDArray *weights, const NDArray *bias, NDArray *output,
                         const int kH, const int kW, const int sH, const int sW, const int pH, const int pW,
                         const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {
  // mkl support weights in [oC, iC, kH, kW] format only
  printf("%s %d\n",__FILE__,__LINE__);
  int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  printf("Running conv2d onednn with strides: %d %d padding: %d %d dilation: %d %d paddingMode %d weightFormat %d\n",sH,sW,pH,pW,dH,dW,paddingMode,wFormat);
  const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2
                                                  : pW;  // dH == 1 for causal mode in conv1d

  dnnl::memory::dims strides = {sH, sW};
  dnnl::memory::dims padding = {pH, pW};
  dnnl::memory::dims padding_r = {(oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pWSame};
  dnnl::memory::dims dilation = {dH - 1, dW - 1};
  printf("%s %d\n",__FILE__,__LINE__);
  auto xzFormatMkl = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oihw;

  dnnl::memory::dims xDims = {bS, iC, iH, iW};
  dnnl::memory::dims wDims = {oC, iC, kH, kW};
  dnnl::memory::dims zDims = {bS, oC, oH, oW};
  printf("%s %d\n",__FILE__,__LINE__);
  auto type = dnnl::memory::data_type::f32;

  std::vector<int> permut;
  if (0 == wFormat)
    permut = {3, 2, 0, 1};  // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
  else if (2 == wFormat)
    permut = {0, 3, 1, 2};  // [oC, kH, kW, iC] -> [oC, iC, kH, kW]

  // memory descriptors for arrays

  printf("Creating input descriptor %d\n",0);
  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFormatMkl);
  setBlockStrides(*input, x_user_md);
  printf("%s %d\n",__FILE__,__LINE__);
  printf("Creating weight descriptor %d\n",0);

  // weights
  dnnl::memory::desc w_mkl_md = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormatMkl);
  setBlockStrides(*weights, w_user_md, permut);
  printf("%s %d\n",__FILE__,__LINE__);
  printf("Creating bias descriptor %d\n",0);

  // bias
  dnnl::memory::desc b_mkl_md;
  if (bias != nullptr) b_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);
  printf("%s %d\n",__FILE__,__LINE__);
  printf("Creating output %d\n",0);

  // output
  dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, type, xzFormatMkl);
  setBlockStrides(*output, z_user_md);
  printf("%s %d\n",__FILE__,__LINE__);
  auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);

  printf("Creating op descriptor %d\n",0);
  printf("%s %d\n",__FILE__,__LINE__);
  // operation primitive description
  dnnl::convolution_forward::desc op_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
                                          x_mkl_md, w_mkl_md, b_mkl_md, z_mkl_md, strides, dilation, padding,
                                          padding_r);
  printf("%s %d\n",__FILE__,__LINE__);
  printf("Creating prim  descriptor %d\n",0);

  dnnl::convolution_forward::primitive_desc op_prim_desc(op_desc, engine);
  printf("Created engine %d\n",0);
  printf("%s %d\n",__FILE__,__LINE__);
  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);
  printf("%s %d\n",__FILE__,__LINE__);
  // provide memory buffers and check whether reorder is required

  // input
  loadDataToMklStream(*input, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);
  printf("%s %d\n",__FILE__,__LINE__);
  // weights
  loadDataToMklStream(*weights, engine, stream, w_user_md, op_prim_desc.weights_desc(),
                                   args[DNNL_ARG_WEIGHTS]);
  printf("%s %d\n",__FILE__,__LINE__);
  // bias
  if (bias != nullptr) {
    auto b_mkl_mem = dnnl::memory(b_mkl_md, engine, const_cast<void *>(bias->buffer()));
    args[DNNL_ARG_BIAS] = b_mkl_mem;
  }
  printf("%s %d\n",__FILE__,__LINE__);
  // output
  auto z_user_mem =
      loadDataToMklStream(*output, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);
  printf("%s %d\n",__FILE__,__LINE__);
  // run calculations
  dnnl::convolution_forward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);
  printf("%s %d\n",__FILE__,__LINE__);
  stream.wait();
  // shape::printArray(z_mkl_mem.map_data<float>(),8);
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
    Conv &args) {
    std::cout << args << std::endl;

     int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  getSizesAndIndexesConv2d(args.isNCHW, args.wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  calcPadding2D(args.pH, args.pW, oH, oW, iH, iW, args.kH, args.kW, args.sH, args.sW, args.dH, args.dW, args.paddingMode);

  std::vector<Nd4jLong> expectedWeightsShape = expectWeightsShape(args.wFormat, args.kH, args.kW, iC, oC);
 

  conv2dMKLDNN(input, weights, bias, output, args.kH, args.kW, args.sH, args.sW, args.pH, args.pW, args.dH, args.dW, args.paddingMode, args.isNCHW, args.wFormat);

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
out.printIndexedBuffer("out");
	int fg;
 
	std::cin >> fg;
	return 0;
}
