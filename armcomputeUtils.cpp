

#include "armcomputeUtils.h"


namespace sd { 

            Arm_DataType getArmType(const DataType& dType) {
                Arm_DataType  ret;
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
                return dType != Arm_DataType::UNKNOWN && arr.ordering() == 'c' && arrStrides[rank - 1] == 1;
            }

            

            Arm_Tensor getArmTensor(const NDArray& arr, arm_compute::DataLayout layout) {
                // - Ownership of the backing memory is not transferred to the tensor itself.
                // - The tensor mustn't be memory managed.
                // - Padding requirements should be accounted by the client code. 
                // In other words, if padding is required by the tensor after the function configuration step,
                // then the imported backing memory should account for it. 
                // Padding can be checked through the TensorInfo::padding() interface.
                auto dType = getArmType(arr.dataType());
                if (dType == Arm_DataType::UNKNOWN) {
                    throw std::runtime_error("unsupported Data Type");
                }

                //
                int numChannels = 1;
                int rank = (int)(arr.rankOf());
                Nd4jLong* bases = arr.shapeOf();
                Nd4jLong* arrStrides = arr.stridesOf();
                if (rank > arm_compute::MAX_DIMS) {
                    throw std::runtime_error("the rank of the array is higher");
                }
                if (arr.ordering() != 'c' || arrStrides[rank - 1] != 1) {
                    throw std::runtime_error("NDArray should be c order and its last stride should be 1");
                }
                //https://arm-software.github.io/ComputeLibrary/v20.05/_dimensions_8h_source.xhtml
                //note: underhood it is stored as std::array<T, num_max_dimensions> _id;
                //TensorShape is derived from Dimensions<uint32_t>
                //as well as Strides : public Dimensions<uint32_t>
                Arm_TensorShape shape;
                Arm_Strides strides;
                shape.set_num_dimensions(rank);
                strides.set_num_dimensions(rank);
                size_t element_size = arm_compute::data_size_from_type(dType);
                for (int i = 0, j = rank - 1; i < rank; i++, j--) {
                    shape[i] = static_cast<uint32_t>(bases[j]);
                    strides[i] = static_cast<uint32_t>(arrStrides[j]) * element_size;
                }
                //fill the rest unused with 1
                for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
                    shape[i] = 1;
                }
                size_t total_size;
                size_t size_ind = rank - 1;
                total_size = shape[size_ind] * strides[size_ind];

                Arm_TensorInfo info;
                info.init(shape, numChannels, dType, strides, 0, total_size);
                info.set_data_layout(layout);
                // Import existing pointer as backing memory
                Arm_Tensor tensor;
                tensor.allocator()->init(info);
                void* buff = arr.getBuffer();
                tensor.allocator()->import_memory(buff);

                return tensor;

            }

#define ARM_COMPUTE_ASSERTS_ENABLED 1
            //armcompute should be built with debug option            
            void print_tensor(Arm_ITensor& tensor, const char* msg) {
#ifdef ARM_COMPUTE_ASSERTS_ENABLED   
                std::cout << msg << ":\n";
                tensor.print(std::cout);
                std::cout << std::endl;
#endif 
            }

 
}
