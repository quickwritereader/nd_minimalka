

#include "armcomputeUtils.h"
#include <cstdint>
#include <LoopCoordsHelper.h>
namespace sd {

Arm_DataType getArmType(const DataType& dType) {
  Arm_DataType ret;
  switch (dType) {
    case HALF:
      ret = Arm_DataType::F16;
      break;
    case FLOAT32:
      ret = Arm_DataType::F32;
      break;
    case DOUBLE:
      ret = Arm_DataType::F64;
      break;
    case INT8:
      ret = Arm_DataType::S8;
      break;
    case INT16:
      ret = Arm_DataType::S16;
      break;
    case INT32:
      ret = Arm_DataType::S32;
      break;
    case INT64:
      ret = Arm_DataType::S64;
      break;
    case UINT8:
      ret = Arm_DataType::U8;
      break;
    case UINT16:
      ret = Arm_DataType::U16;
      break;
    case UINT32:
      ret = Arm_DataType::U32;
      break;
    case UINT64:
      ret = Arm_DataType::U64;
      break;
    case BFLOAT16:
      ret = Arm_DataType::BFLOAT16;
      break;
    default:
      ret = Arm_DataType::UNKNOWN;
  };

  return ret;
}

bool isArmcomputeFriendly(const NDArray& arr) {
  auto dType = getArmType(arr.dataType());
  int rank = (int)(arr.rankOf());
  Nd4jLong* arrStrides = arr.stridesOf();
  return dType != Arm_DataType::UNKNOWN && arr.ordering() == 'c' &&
         arrStrides[rank - 1] == 1;
}

Arm_TensorInfo getArmTensorInfo(int rank, Nd4jLong* bases,sd::DataType ndArrayType, arm_compute::DataLayout layout) {
    constexpr int numChannels = 1; 
    auto dType = getArmType(ndArrayType);
    if (dType == Arm_DataType::UNKNOWN) {
        throw std::runtime_error("unsupported Data Type");
    }

    if (rank > arm_compute::MAX_DIMS) {
        throw std::runtime_error("the rank of the array is higher");
    } 
    Arm_TensorShape shape;
    shape.set_num_dimensions(rank); 
    for (int i = 0, j = rank - 1; i < rank; i++, j--) {
        shape[i] = static_cast<uint32_t>(bases[j]); 
    }
    // fill the rest unused with 1
    for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
        shape[i] = 1;
    } 

    return Arm_TensorInfo(shape, numChannels, dType, layout); 
}

Arm_TensorInfo getArmTensorInfo(const sd::NDArray& arr,
                                arm_compute::DataLayout layout) {
  auto dType = getArmType(arr.dataType());
  if (dType == Arm_DataType::UNKNOWN) {
    throw std::runtime_error("unsupported Data Type");
  }

  //
  constexpr int numChannels = 1;
  int rank = (int)(arr.rankOf());
  Nd4jLong* bases = arr.shapeOf();
  Nd4jLong* arrStrides = arr.stridesOf();
  if (rank > arm_compute::MAX_DIMS) {
    throw std::runtime_error("the rank of the array is higher");
  }
  if (arr.ordering() != 'c' /*|| arrStrides[rank - 1] != 1*/) {
      throw std::runtime_error(
          "NDArray should be c order ");// and its last stride should be 1");
  }
  // https://arm-software.github.io/ComputeLibrary/v20.05/_dimensions_8h_source.xhtml
  // note: underhood it is stored as std::array<T, num_max_dimensions> _id;
  // TensorShape is derived from Dimensions<uint32_t>
  // as well as Strides : public Dimensions<uint32_t>
  Arm_TensorShape shape;
  Arm_Strides strides;
  shape.set_num_dimensions(rank);
  strides.set_num_dimensions(rank);
  size_t element_size = arm_compute::data_size_from_type(dType);
  for (int i = 0, j = rank - 1; i < rank; i++, j--) {
    shape[i] = static_cast<uint32_t>(bases[j]);
    strides[i] = static_cast<uint32_t>(arrStrides[j]) * element_size;
  }
  // fill the rest unused with 1
  for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
    shape[i] = 1;
  }
  //size_t total_size;
  //size_t size_ind = rank - 1;
  //total_size = shape[size_ind] * strides[size_ind];
  auto total_size = arr.getDataBuffer()->getLenInBytes();
  auto offset = arr.bufferOffset() * element_size;
  Arm_TensorInfo info;
  info.init(shape, numChannels, dType, strides, offset, total_size);
  info.set_data_layout(layout);

  return info;
}

Arm_Tensor getArmTensor(const NDArray& arr, arm_compute::DataLayout layout) {
  // - Ownership of the backing memory is not transferred to the tensor itself.
  // - The tensor mustn't be memory managed.
  // - Padding requirements should be accounted by the client code.
  // In other words, if padding is required by the tensor after the function
  // configuration step, then the imported backing memory should account for it.
  // Padding can be checked through the TensorInfo::padding() interface.

  // Import existing pointer as backing memory
  auto info = getArmTensorInfo(arr, layout);
  Arm_Tensor tensor;
  tensor.allocator()->init(info);

  //get without offset
  void* buff = arr.getDataBuffer()->primary();
  std::cout << "+++" << buff << std::endl;
  tensor.allocator()->import_memory(buff);
  return tensor;
}

std::tuple<int, int, int, int> getAutoPadding(int rank) {
    auto extra_pad_x = rank < 1 ? 0 : 32;
    auto pad_x = rank < 1 ? 0 : 4;
    auto pad_y = rank < 2 ? 0 : 4;
    return std::tuple<int, int, int, int>{ pad_y, pad_x + extra_pad_x, pad_y, pad_x };
}

void copyFromTensor(const Arm_Tensor& inTensor, sd::NDArray& output) {
    //only for C order
    if (output.ordering() != 'c') return;
    Nd4jLong* shapeInfo = output.getShapeInfo();
    Nd4jLong* bases = &(shapeInfo[1]);
    Nd4jLong rank = shapeInfo[0];
    Nd4jLong* strides = output.stridesOf();
    int width = bases[rank - 1];
    uint8_t* outputBuffer = (uint8_t*)output.getBuffer(); 
    size_t offset = 0;
    arm_compute::Window window;
    arm_compute::Iterator tensor_it(&inTensor, window);

    int element_size = inTensor.info()->element_size();
    window.use_tensor_dimensions(inTensor.info()->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY);

    if (output.ews() == 1) {
        auto copySize = width * element_size;
        auto dest = outputBuffer;
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates& id)
            {
                auto src = tensor_it.ptr(); 
                memcpy(dest, src, copySize);
                dest += copySize;
            },
            tensor_it);
    }
    else {
        Nd4jLong coords[MAX_RANK] = {};
        auto copySize = width * element_size;
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates& id)
            {
                auto src = tensor_it.ptr();
                auto dest = outputBuffer + offset * element_size;
                memcpy(dest, src, copySize);
                offset = sd::inc_coords(bases, strides, coords, offset, rank, 1);
            },
            tensor_it);
    }
}

void copyToTensor(const sd::NDArray& input, Arm_Tensor& outTensor) {
    //only for C order
    if (input.ordering() != 'c') return;
    Nd4jLong* shapeInfo = input.getShapeInfo();
    Nd4jLong* bases = &(shapeInfo[1]);
    Nd4jLong rank = shapeInfo[0];
    Nd4jLong* strides = input.stridesOf();
    uint8_t *inputBuffer = (uint8_t*)input.getBuffer(); 
    int width = bases[rank - 1];
    size_t offset = 0; 
    arm_compute::Window window;
    arm_compute::Iterator tensor_it(&outTensor, window);
    int element_size = outTensor.info()->element_size(); 

    window.use_tensor_dimensions(outTensor.info()->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY);
    
 if (input.ews() == 1) {

     auto copySize = width * element_size;
     auto src = inputBuffer;
     arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates& id)
         {
             auto dest = tensor_it.ptr(); 
             memcpy(dest,src, copySize);
             src += copySize;
         },
         tensor_it);
 }
 else {
     Nd4jLong coords[MAX_RANK] = {};
     auto copySize = width * element_size;
     arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates& id)
         {
             auto dest = tensor_it.ptr();
             auto src = inputBuffer + offset * element_size;
             memcpy(dest, src, copySize);
             offset = sd::inc_coords(bases, strides, coords, offset, rank, 1);
         },
         tensor_it);
 }
}


// armcompute should be built with debug option
void print_tensor(Arm_ITensor& tensor, const char* msg) {
    auto info = tensor.info();
  auto padding = info->padding();
  std::cout << msg << "\ntotal: " << info->total_size() << "\n";

  for (int i = 0; i < arm_compute::MAX_DIMS; i++) {
    std::cout << info->dimension(i) << ",";
  }
  std::cout << std::endl;
  for (int i = 0; i < arm_compute::MAX_DIMS; i++) {
    std::cout << info->strides_in_bytes()[i] << ",";
  }
  std::cout << "\npadding: l " << padding.left << ", r " << padding.right
            << ", t " << padding.top << ", b " << padding.bottom << std::endl;
  std::cout << "\noffset_first_element_in_bytes : " << info->offset_first_element_in_bytes() << std::endl;;
#ifdef ARM_COMPUTE_ASSERTS_ENABLED
  std::cout << msg << ":\n";
  tensor.print(std::cout);
  std::cout << std::endl;
#endif
}

}  // namespace sd
