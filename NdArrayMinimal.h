#pragma once
#include "NDX.h"
#include <numeric>
namespace sd {

    struct ShapeUtils {
        static FORCEINLINE  std::string shapeAsString(const Nd4jLong* shapeInfo) {

            if (!shapeInfo)
                throw std::runtime_error("shapeAsString method: input shapeInfo must not be nullptr !");

            std::string result;

            result.append("[");
            for (int e = 0; e < shapeInfo[0]; e++) {
                result += std::to_string(shapeInfo[e + 1]);
                if (e < shapeInfo[0] - 1)
                    result.append(", ");
            }
            result.append("]");

            return result;
        }

        static FORCEINLINE    std::string shapeInfoAsString(const Nd4jLong* shapeInfo) {

            if (!shapeInfo)
                throw std::runtime_error("shapeAsString method: input shapeInfo must not be nullptr !");

            std::string result;

            int len = shape::shapeInfoLength(shapeInfo[0]);

            result.append("[");
            for (int e = 0; e < len; e++) {
                result += std::to_string(shapeInfo[e]);
                if (e < len - 1)
                    result.append(", ");
            }
            result.append("]");

            return result;
        }


        static FORCEINLINE     std::string shapeAsString(const int rank, const Nd4jLong* shapeInfo) {
            if (!shapeInfo)
                throw std::runtime_error("shapeAsString method: input shapeInfo must not be nullptr !");

            std::string result;

            result.append("[");
            for (int e = 0; e < rank; e++) {
                result += std::to_string(shapeInfo[e]);
                if (e < rank - 1)
                    result.append(", ");
            }
            result.append("]");

            return result;
        }

        //////////////////////////////////////////////////////////////////////////
        static FORCEINLINE   std::vector<Nd4jLong> shapeAsVector(const Nd4jLong* shapeInfo) {

            if (!shapeInfo)
                throw std::runtime_error("shapeAsVector method: input shapeInfo must not be nullptr !");

            std::vector<Nd4jLong> vector(shapeInfo[0]);

            for (uint e = 0; e < shapeInfo[0]; e++)
                vector[e] = shapeInfo[e + 1];

            return vector;
        }

        static FORCEINLINE  std::vector<int>  evalDimsToExclude(const int rank, const int dimsLen, const int* dimensions) {

            std::vector<int> newDimensions;
            if (dimsLen == 0) {                          // if input vector is empty then return whole shape range
                newDimensions.resize(rank);
                std::iota(newDimensions.begin(), newDimensions.end(), 0);   // fill with 0, 1, ... rank-1
            }
            else {
                bool isAbsent;
                for (int i = 0; i < rank; ++i) {
                    isAbsent = true;
                    for (int j = 0; j < dimsLen; ++j) {
                        int dim = dimensions[j] >= 0 ? dimensions[j] : dimensions[j] + rank;
                        if (i == dim) {
                            isAbsent = false;
                            break;
                        }
                    }
                    if (isAbsent)
                        newDimensions.emplace_back(i);
                }
            }

            return newDimensions;
        }

        //////////////////////////////////////////////////////////////////////////
        static FORCEINLINE   std::vector<int>  evalDimsToExclude(const int rank, const std::vector<int>& dimensions) {

            return ShapeUtils::evalDimsToExclude(rank, dimensions.size(), dimensions.data());
        }

        static FORCEINLINE  std::vector<int> evalBroadcastBackwardAxis(const Nd4jLong* operandShapeInfo, const Nd4jLong* resultShapeInfo) {
            // rRank >= oRank always  !!
            const auto oRank = shape::rank(operandShapeInfo);
            const auto rRank = shape::rank(resultShapeInfo);
            const auto diff = rRank - oRank;
            std::vector<int> axis;

            for (int i = 0; i < rRank; ++i)
                if (i < diff || shape::sizeAt(operandShapeInfo, i - diff) != shape::sizeAt(resultShapeInfo, i))
                    axis.push_back(i);

            return axis;
        }

        ////////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE  Nd4jLong* matrixProductShape(Nd4jLong* theFirstShape, Nd4jLong* theSecondShape, bool shouldTranspondFirst, bool shouldTranspondSecond, sd::DataType  dtype, sd::memory::Workspace* workspace) {

            auto inA = theFirstShape;
            auto inB = theSecondShape;
            Nd4jLong* shape;
            ALLOCATE(shape, workspace, shape::shapeInfoLength(2), Nd4jLong);

            Nd4jLong* tmpA = ShapeBuilders::copyShapeInfo(inA, true, workspace);
            Nd4jLong* tmpB = ShapeBuilders::copyShapeInfo(inB, true, workspace);

            if (shouldTranspondFirst)
                shape::transposeInplace(tmpA);

            if (shouldTranspondSecond)
                shape::transposeInplace(tmpB);


            if (shape::rank(tmpA) == 1 && shape::isMatrix(tmpB)) {
                // special case here
                shape[0] = 1;
                shape[1] = tmpB[2];
                Nd4jLong* newShape = ShapeBuilders::createShapeInfo(dtype, 'f', 2, shape, workspace);

                RELEASE(shape, workspace);
                RELEASE(tmpA, workspace);
                RELEASE(tmpB, workspace);

                return newShape;
            }
            else if (shape::isScalar(tmpA) && shape::isScalar(tmpB)) {
                // just scalar vs scalar
                shape[0] = 1;
                shape[1] = 1;
            }
            else if (shape::isMatrix(tmpA) && shape::isVector(tmpB)) {
                // gemv case
                if (shape::rank(tmpB) == 2) {
                    shape[0] = tmpA[1];
                    shape[1] = tmpB[2];
                }
                else {
                    // we have new 1D shape here
                    auto newShape = ShapeBuilders::createVectorShapeInfo(dtype, tmpA[1], workspace);

                    RELEASE(shape, workspace);
                    RELEASE(tmpA, workspace);
                    RELEASE(tmpB, workspace);

                    return newShape;
                }
            }
            else if ((shape::isMatrix(tmpA) && shape::isMatrix(tmpB)) ||
                (shape::isVector(tmpA) && shape::isMatrix(tmpB)) ||
                (shape::isColumnVector(tmpA) && shape::isVector(tmpB))) {
                // gemm case
                shape[0] = tmpA[1];
                shape[1] = tmpB[2];
            }
            else if ((shape::isVector(tmpA) && shape::isScalar(tmpB)) ||
                (shape::isScalar(tmpA) && shape::isVector(tmpB))) {
                // element-wise
                shape[0] = 1;
                shape[1] = (int)std::max(shape::length(tmpA), shape::length(tmpB));
            }
            else if (shape::isRowVector(tmpA) && shape::isRowVector(tmpB)) {
                // dot case
                shape[0] = 1;
                shape[1] = 1;
            }
            else if (shape::isRowVector(tmpA) && shape::isColumnVector(tmpB)) {
                // dot case
                shape[0] = 1;
                shape[1] = 1;
            }

            Nd4jLong* newShape = ShapeBuilders::createShapeInfo(dtype, 'f', 2, shape, workspace);

            RELEASE(shape, workspace);

            RELEASE(tmpA, workspace);
            RELEASE(tmpB, workspace);
            return newShape;
        }

        ////////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE  std::vector<int> evalPermutFromTo(const std::vector<Nd4jLong>& shapeFrom, const std::vector<Nd4jLong>& shapeTo) {
            auto rank = shapeFrom.size();
            if (rank != shapeTo.size())
                throw std::runtime_error("evalPermutFromTo static method: the input shapes are not suitable for mutual permutation !");

            if (std::equal(begin(shapeFrom), end(shapeFrom), begin(shapeTo)))       // if shapes are identical (permutation is unnecessary) then return empty vector
                return std::vector<int>();

            std::vector<int> permutation(rank, -2);                                 // vector to be returned
            std::vector<Nd4jLong> shapeTo2(shapeTo);                                     // make copy of const vector since we will change the content of shapeTo

            for (int i = 0; i < rank; ++i)
                for (int j = 0; j < rank; ++j)
                    if (shapeFrom[i] == shapeTo2[j]) {
                        permutation[j] = i;
                        shapeTo2[j] = -2;                                           // mark coincidence as -2 in order to not account index of shapeTo twice
                        break;
                    }

            if (std::find(begin(permutation), end(permutation), -2) != end(permutation))      // if -2 is still present in vector then permutation is impossible
                throw std::runtime_error("evalPermutFromTo static method: the input shapes are not suitable for mutual permutation !");

            return permutation;
        }


        ////////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE  std::vector<Nd4jLong> composeShapeUsingDimsAndIdx(const std::vector<int>& dimsAndIdx) {
            auto size = dimsAndIdx.size();
            if (size % 2 != 0)
                throw std::runtime_error("composeShapeUsingDimsAndIdx static method: the size of input vector must be even !");

            size /= 2;

            std::vector<Nd4jLong> shape(size);
            int index;

            for (int i = 0; i < size; ++i) {
                index = dimsAndIdx[i + size];
                if (index > size - 1)
                    throw std::runtime_error("composeShapeUsingDimsAndIdx static method: input index is too large !");
                shape[index] = dimsAndIdx[i];
            }

            return shape;
        }


        ////////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE  std::vector<Nd4jLong> evalShapeForMatmul(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const bool transX, const bool transY) {

            const auto xRank = xShapeInfo[0];
            const auto yRank = yShapeInfo[0];

            const Nd4jLong x0Dim = transX ? xShapeInfo[xRank] : xShapeInfo[xRank - 1];
            const Nd4jLong y0Dim = transY ? yShapeInfo[yRank] : yShapeInfo[yRank - 1];
            const Nd4jLong x1Dim = transX ? xShapeInfo[xRank - 1] : xShapeInfo[xRank];
            const Nd4jLong y1Dim = transY ? yShapeInfo[yRank - 1] : yShapeInfo[yRank];


            if (xRank == 1 && yRank == 1) {   // dot case, output is scalar
                if (xShapeInfo[1] != yShapeInfo[1]) {
                    nd4j_printf("evalShapeForMatmul method: since input arrays are vectors they must have the same length, but got x length = %lli, y length = %lli !", xShapeInfo[1], yShapeInfo[1]);
                    throw std::invalid_argument("");
                }
                return std::vector<Nd4jLong>({});
            }


            if (xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
                if (xShapeInfo[1] != y0Dim) {
                    nd4j_printf("evalShapeForMatmul method: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !", shapeAsString(xShapeInfo).c_str(), shapeAsString(yShapeInfo).c_str());
                    throw std::invalid_argument("");
                }
                return std::vector<Nd4jLong>({ y1Dim });
            }


            if (xRank == 2 && yRank == 1) {  // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
                if (x1Dim != yShapeInfo[1]) {
                    nd4j_printf("evalShapeForMatmul method: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !", shapeAsString(xShapeInfo).c_str(), shapeAsString(yShapeInfo).c_str());
                    throw std::invalid_argument("");
                }
                return std::vector<Nd4jLong>({ x0Dim });
            }


            // rest cases - usual 2Dx2D or batched mmul
            if (xRank != yRank) {
                nd4j_printf("evalShapeForMatmul static method: the ranks of arrays must be the same, but got xRank = %lli and yRank = %lli ! \n", xRank, yRank);
                throw std::invalid_argument("");
            }

            if (x1Dim != y0Dim) {
                nd4j_printf("evalShapeForMatmul static method: input shapes are inconsistent: xDim %lli != yDim %lli \n", x1Dim, y0Dim);
                throw std::invalid_argument("");
            }

            for (int i = 0; i < xRank - 2; ++i)
                if (xShapeInfo[i + 1] != yShapeInfo[i + 1]) {
                    nd4j_printf("evalShapeForMatmul static method: input shapes are inconsistent: xShape = %s, yShape = %s ! \n", shapeAsString(xShapeInfo).c_str(), shapeAsString(yShapeInfo).c_str());
                    throw std::invalid_argument("");
                }

            std::vector<Nd4jLong> cShape(xRank);

            // copy batch part of shape (if present)
            for (int i = 0; i < xRank - 2; ++i)
                cShape[i] = xShapeInfo[i + 1];
            // copy rest part of shape (two dims: multiplication part)
            cShape[xRank - 2] = x0Dim;
            cShape[xRank - 1] = y1Dim;

            return cShape;
        }

        ////////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE  Nd4jLong getNumOfSubArrs(const Nd4jLong* shapeInfo, const std::vector<int>& dimsToExclude) {

            Nd4jLong numOfSubArrs = 1;

            if (dimsToExclude.size() == shape::rank(shapeInfo) || dimsToExclude.size() == 0)     // means there is only one sub-array and it coincides with whole array
                return numOfSubArrs;

            for (const auto& dim : dimsToExclude)
                numOfSubArrs *= shapeInfo[dim + 1];

            return numOfSubArrs;
        }

        ////////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE  std::vector<Nd4jLong> evalDimsWithoutUnities(const Nd4jLong* shapeInfo) {

            std::vector<Nd4jLong> result;
            for (int i = 1; i <= shapeInfo[0]; ++i)
                if (shapeInfo[i] != 1)
                    result.push_back(shapeInfo[i]);

            return result;
        }
    };

    class ND4J_EXPORT NDIndex {
    protected:
        std::vector<Nd4jLong> _indices;
        Nd4jLong _stride = 1;
    public:
        NDIndex() = default;
        ~NDIndex() = default;

        bool isAll() {
            return _indices.size() == 1 && _indices.at(0) == -1;
        }
        bool isPoint() {
            return _indices.size() == 1 && _indices.at(0) >= 0;
        }
        virtual bool isInterval() {
            return false;
        }

        std::vector<Nd4jLong>& getIndices() {
            return _indices;
        }
        Nd4jLong stride() {
            return _stride;
        }

        static NDIndex* all();
        static NDIndex* point(Nd4jLong pt);
        static NDIndex* interval(Nd4jLong start, Nd4jLong end, Nd4jLong stride = 1);

    };

    class ND4J_EXPORT NDIndexAll : public NDIndex {
    public:
        NDIndexAll() {
            _indices.push_back(-1);
        }
        virtual bool isInterval() {
            return false;
        }
        ~NDIndexAll() = default;
    };


    class ND4J_EXPORT NDIndexPoint : public NDIndex {
    public:
        NDIndexPoint(Nd4jLong point) {
            this->_indices.push_back(point);
        }
        virtual bool isInterval() {
            return false;
        }
        ~NDIndexPoint() = default;
    };

    class ND4J_EXPORT NDIndexInterval : public NDIndex {
    public:
        NDIndexInterval(Nd4jLong start, Nd4jLong end, Nd4jLong stride = 1) {
            this->_stride = stride;
            for (int e = start; e < end; e += stride)
                this->_indices.push_back(e);
        }
        virtual bool isInterval() {
            return true;
        }
        ~NDIndexInterval() = default;
    };




    FORCEINLINE NDIndex* NDIndex::all() {
        return new NDIndexAll();
    }
    FORCEINLINE NDIndex* NDIndex::point(Nd4jLong pt) {
        return new NDIndexPoint(pt);
    }
    FORCEINLINE NDIndex* NDIndex::interval(Nd4jLong start, Nd4jLong end, Nd4jLong stride) {
        return new NDIndexInterval(start, end, stride);
    }
}


namespace sd {

  
    struct DataTypeUtils {
        static FORCEINLINE _CUDA_HD size_t sizeOfElement(sd::DataType type) {
            switch (type) {
            case sd::DataType::UINT8:
            case sd::DataType::INT8:
            case sd::DataType::FLOAT8:
            case sd::DataType::QINT8:
            case sd::DataType::BOOL: return (size_t)1;

            case sd::DataType::BFLOAT16:
            case sd::DataType::HALF:
            case sd::DataType::INT16:
            case sd::DataType::QINT16:
            case sd::DataType::UINT16: return (size_t)2;

            case sd::DataType::UTF8:
            case sd::DataType::UTF16:
            case sd::DataType::UTF32:
            case sd::DataType::INT32:
            case sd::DataType::UINT32:
            case sd::DataType::HALF2:
            case sd::DataType::FLOAT32: return (size_t)4;

            case sd::DataType::UINT64:
            case sd::DataType::INT64:
            case sd::DataType::DOUBLE: return (size_t)8;

            default: { 
#ifndef __CUDA_ARCH__
                throw std::runtime_error("Unknown DataType requested");
#endif
            }
            }
        }

        template <typename T>
        static FORCEINLINE _CUDA_HD sd::DataType  fromT() {
            if (std::is_same<T, bool>::value) {
                return sd::DataType::BOOL;
            }
            else if (std::is_same<T, std::string>::value) {
                return sd::DataType::UTF8;
            }
            else if (std::is_same<T, std::u16string>::value) {
                return sd::DataType::UTF16;
            }
            else if (std::is_same<T, std::u32string>::value) {
                return sd::DataType::UTF32;
            }
            else if (std::is_same<T, float>::value) {
                return sd::DataType::FLOAT32;
            }
            //else if (std::is_same<T, float16>::value) {
            //    return sd::DataType::HALF;
            //}
            //else if (std::is_same<T, bfloat16>::value) {
            //    return sd::DataType::BFLOAT16;
            //}
            else if (std::is_same<T, double>::value) {
                return sd::DataType::DOUBLE;
            }
            else if (std::is_same<T, int8_t >::value) {
                return sd::DataType::INT8;
            }
            else if (std::is_same<T, int16_t >::value) {
                return sd::DataType::INT16;
            }
            else if (std::is_same<T, int>::value) {
                return sd::DataType::INT32;
            }
            else if (std::is_same<T, Nd4jLong>::value) {
                return sd::DataType::INT64;
            }
            else if (std::is_same<T, uint8_t>::value) {
                return sd::DataType::UINT8;
            }
            else if (std::is_same<T, uint16_t>::value) {
                return sd::DataType::UINT16;
            }
            else if (std::is_same<T, uint32_t>::value) {
                return sd::DataType::UINT32;
            }
            else if (std::is_same<T, Nd4jULong>::value) {
                return sd::DataType::UINT64;
            }
            else {
                return sd::DataType::INHERIT;
            }
        }

        static FORCEINLINE size_t sizeOf(DataType type) {
            return sizeOfElement(type);
        }

        ///////////////////////////////////////////////////////////////////
        static FORCEINLINE size_t  sizeOf(const Nd4jLong* shapeInfo) {
            return sizeOfElement(ArrayOptions::dataType(shapeInfo));
        }
    };


    class ND4J_EXPORT DataBuffer {

    private:

        void* _primaryBuffer = nullptr;
        void* _specialBuffer = nullptr;
        size_t _lenInBytes = 0;
        DataType _dataType;
        memory::Workspace* _workspace = nullptr;
        bool _isOwnerPrimary;
        bool _isOwnerSpecial;

    public:
        // default constructor
        DataBuffer() {

            _primaryBuffer = nullptr;
            _specialBuffer = nullptr;
            _lenInBytes = 0;
            _dataType = INT8;
            _workspace = nullptr;
            _isOwnerPrimary = false;
            _isOwnerSpecial = false;
         


        }

        ////////////////////////////////////////////////////////////////////////
        // copy constructor
        DataBuffer(const DataBuffer& other) {

            throw std::runtime_error("DataBuffer copy constructor: we don't expect using of this constructor!");

            _lenInBytes = other._lenInBytes;
            _dataType = other._dataType;
            _workspace = other._workspace;

            _primaryBuffer = nullptr;
            _specialBuffer = nullptr;
 

            allocateBuffers();
            copyBufferFrom(other);
        }

        ////////////////////////////////////////////////////////////////////////
        DataBuffer(void* primary, void* special,
            const size_t lenInBytes, const DataType dataType,
            const bool isOwnerPrimary, const bool isOwnerSpecial,
            memory::Workspace* workspace=nullptr) {

            if (primary == nullptr && special == nullptr)
                throw std::runtime_error("DataBuffer constructor: can't be initialized with both nullptr buffers !");

            _primaryBuffer = primary;
            _specialBuffer = special;
            _lenInBytes = lenInBytes;
            _dataType = dataType;
            _workspace = workspace;
            _isOwnerPrimary = isOwnerPrimary;
            _isOwnerSpecial = isOwnerSpecial;
            
        }

        ////////////////////////////////////////////////////////////////////////
        DataBuffer(void* primary, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary, memory::Workspace* workspace=nullptr) :
            DataBuffer(primary, nullptr, lenInBytes, dataType, isOwnerPrimary, false, workspace) {
             
        }

 

        ////////////////////////////////////////////////////////////////////////
        DataBuffer(const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace=nullptr, const bool allocBoth=false) {

            _dataType = dataType;
            _workspace = workspace;
            _lenInBytes = lenInBytes;

            _primaryBuffer = nullptr;
            _specialBuffer = nullptr;
             
            if (lenInBytes != 0) {
                allocateBuffers(allocBoth); 
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // move constructor
        DataBuffer(DataBuffer&& other) {

            _primaryBuffer = other._primaryBuffer;
            _specialBuffer = other._specialBuffer;
            _lenInBytes = other._lenInBytes;
            _dataType = other._dataType;
            _workspace = other._workspace;
            _isOwnerPrimary = other._isOwnerPrimary;
            _isOwnerSpecial = other._isOwnerSpecial;
            

            other._primaryBuffer = other._specialBuffer = nullptr;
            other.setAllocFlags(false, false);
            other._lenInBytes = 0;
        }

        ////////////////////////////////////////////////////////////////////////
        // assignment operator
        DataBuffer& operator=(const DataBuffer& other) {

            if (this == &other)
                return *this;

            deleteBuffers();

            _lenInBytes = other._lenInBytes;
            _dataType = other._dataType;
            _workspace = other._workspace;

            allocateBuffers();
            copyBufferFrom(other);

            return *this;
        }

        ////////////////////////////////////////////////////////////////////////
        // move assignment operator
        DataBuffer& operator=(DataBuffer&& other) noexcept {

            if (this == &other)
                return *this;

            deleteBuffers();

            _primaryBuffer = other._primaryBuffer;
            _specialBuffer = other._specialBuffer;
            _lenInBytes = other._lenInBytes;
            _dataType = other._dataType;
            _workspace = other._workspace;
            _isOwnerPrimary = other._isOwnerPrimary;
            _isOwnerSpecial = other._isOwnerSpecial;



            other._primaryBuffer = other._specialBuffer = nullptr;
            other.setAllocFlags(false, false);
            other._lenInBytes = 0;

            return *this;
        }

        ////////////////////////////////////////////////////////////////////////
        void* primary() {
            return _primaryBuffer;
        }

        ////////////////////////////////////////////////////////////////////////
        void* special() {
            return _specialBuffer;
        }

        ////////////////////////////////////////////////////////////////////////
        DataType getDataType() {
            return _dataType;
        }

        ////////////////////////////////////////////////////////////////////////
        size_t getLenInBytes() const {
            return _lenInBytes;
        }


        ////////////////////////////////////////////////////////////////////////
        void allocatePrimary() {

            if (_primaryBuffer == nullptr && getLenInBytes() > 0) {

                ALLOCATE(_primaryBuffer, _workspace, getLenInBytes(), int8_t);
                _isOwnerPrimary = true;


            }
        }

        ////////////////////////////////////////////////////////////////////////
        void setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial) {
            _isOwnerPrimary = isOwnerPrimary;
            _isOwnerSpecial = isOwnerSpecial;
        }

        ////////////////////////////////////////////////////////////////////////
        void deletePrimary() {

            if (_isOwnerPrimary && _primaryBuffer != nullptr && getLenInBytes() != 0) {
                auto p = reinterpret_cast<int8_t*>(_primaryBuffer);
                RELEASE(p, _workspace);
                _primaryBuffer = nullptr;
                _isOwnerPrimary = false;

            }
        }

        ////////////////////////////////////////////////////////////////////////
        void deleteBuffers() {

            deletePrimary();
             
            _lenInBytes = 0;
        }

        ////////////////////////////////////////////////////////////////////////
        ~DataBuffer() {

            deleteBuffers();
        }

        void setPrimaryBuffer(void* buffer, size_t length) {
            if (_primaryBuffer != nullptr && _isOwnerPrimary) {
                deletePrimary();
            }

            _primaryBuffer = buffer;
            _isOwnerPrimary = false;
            _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
        }

 

        void setDataType(DataType dataType) {
            _dataType = dataType;
        }

   
        void close() {
            this->deleteBuffers();
        }

 
        ////////////////////////////////////////////////////////////////////
        void allocateBuffers(const bool allocBoth = false) {    // always allocate primary buffer only (cpu case)

            allocatePrimary();
        }

        ////////////////////////////////////////////////////////////////////////
        void copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes = 0, const Nd4jLong offsetThis = 0, const Nd4jLong offsetOther = 0) {

            if (sizeToCopyinBytes == 0)
                sizeToCopyinBytes = other.getLenInBytes();
            if (sizeToCopyinBytes == 0)
                return;

            if (other._primaryBuffer != nullptr)
                std::memcpy(static_cast<int8_t*>(_primaryBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType), static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType), sizeToCopyinBytes);
        }





        ////////////////////////////////////////////////////////////////////////
        void setToZeroBuffers(const bool both=false) {

            memset(primary(), 0, getLenInBytes());
        }

        /////////////////////////
        void memcpy(const DataBuffer& dst, const DataBuffer& src) {
            if (src._lenInBytes > dst._lenInBytes)
                throw std::runtime_error("memcpy: Source data buffer is larger than destination");

            std::memcpy(dst._primaryBuffer, src._primaryBuffer, src._lenInBytes);
             
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        T*  primaryAsT() {
            return reinterpret_cast<T*>(_primaryBuffer);
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        T*  specialAsT() {
            return reinterpret_cast<T*>(_specialBuffer);
        }

    };
 

     
    class ND4J_EXPORT NDArray {
    private:
 

        /**
        *  if true then array doesn't own buffer and simply points to another's buffer
        */
        bool _isView = false;

        /**
        *  pointer on DataBuffer buffers in cpu/device memory
        */
        std::shared_ptr<DataBuffer> _buffer = std::make_shared<DataBuffer>();

        /**
        *  buffers offset, it is the same both for cpu and device buffers
        */
        Nd4jLong _offset = 0L;

        /**
        *  contains shape info:  matrix rank, numbers of elements per each dimension, dimensions strides, element-wise-stride, c-like or fortan-like order
        */
        Nd4jLong* _shapeInfo = nullptr;
        Nd4jLong* _shapeInfoD = nullptr;

        /**
        *  pointer on device launch context (with all data needed there).
        */
         sd::memory::Workspace* _context = nullptr;

        // indicates if array's buffer is within workspace
        bool _isAttached = false;

        /**
         * Field to store cached length
         */
        Nd4jLong _length = -1L;

        /**
        *  type of array elements
        */
        sd::DataType _dataType = FLOAT32;

        /**
         * deviceID where this NDArray belongs to
         */
        int _deviceId = 0;
         
    public:

        NDArray() {

        }

        NDArray(const NDArray* other, const bool copyStrides, sd::memory::Workspace* workspace) {

            _context = workspace;
            _offset = 0;
            _isAttached = workspace != nullptr;

            if (copyStrides)
                setShapeInfo(ShapeDescriptor(other->_shapeInfo));
            else
                setShapeInfo(ShapeDescriptor(other->dataType(), other->ordering(), other->shapeOf(), other->rankOf()));

            if (!isEmpty())
                _buffer = std::make_shared<DataBuffer>(lengthOf() * sizeOfT(), dataType(), workspace);
        }

        ////////////////////////////////////////////////////////////////////////
        NDArray(void* buffer, const char order, const std::vector<Nd4jLong>& shape, sd::DataType dtype, sd::memory::Workspace* workspace, const bool isBuffAlloc) {

            if (shape.empty())
                throw std::runtime_error("NDArray constructor: input shape is empty !");

            if ((int)shape.size() > MAX_RANK)
                throw std::invalid_argument("Rank of NDArray can't exceed 32");

            _context = workspace;
            _offset = 0;
            _isAttached = workspace != nullptr;

            setShapeInfo(ShapeDescriptor(dtype, order, shape));

            _buffer = std::make_shared<DataBuffer>(buffer, lengthOf() * sizeOfT(), dataType(), isBuffAlloc, workspace);
        }

        ////////////////////////////////////////////////////////////////////////
        // creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
        NDArray(Nd4jLong* shapeInfo, const sd::DataType dtype, const bool copyStrides, sd::memory::Workspace* workspace) {

            if (shapeInfo == nullptr)
                throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo");

            if ((int)shapeInfo[0] > MAX_RANK)
                throw std::invalid_argument("Rank of NDArray can't exceed 32");

            _context = workspace;
            _offset = 0;

            if (copyStrides)
                setShapeInfo(ShapeDescriptor(shapeInfo, dtype));
            else
                setShapeInfo(ShapeDescriptor(dtype, shape::order(shapeInfo), shape::shapeOf(shapeInfo), shape::rank(shapeInfo)));

            if (!isEmpty()) {
                _buffer = std::make_shared<DataBuffer>(lengthOf() * sizeOfT(), dtype, workspace);
                _buffer->setToZeroBuffers();
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // scalar constructor
        NDArray(sd::DataType dtype, sd::memory::Workspace* workspace, const bool isScalar) {

            _context = workspace;
            _offset = 0;
            _isAttached = workspace != nullptr;

            if (isScalar) {
                setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));
                _buffer = std::make_shared<DataBuffer>(sizeOfT(), dtype, workspace);
                _buffer->setToZeroBuffers();
            }
            else
                setShapeInfo(ConstantShapeHelper::getInstance()->emptyShapeInfo(dtype));
        }

        //////////////////////////////////////////////////////////////////////////
        // move constructor
        NDArray(NDArray&& other) noexcept {

            _isView = other._isView;
            _buffer = other._buffer;
            _shapeInfo = other._shapeInfo;
            _shapeInfoD = other._shapeInfoD;
            _context = other._context;
            _dataType = other._dataType;
            _length = other._length;
            _offset = other._offset;

            other._buffer = std::make_shared<DataBuffer>();
            other._shapeInfo = other._shapeInfoD = nullptr;
            other._length = 0;
        }

        // move assignment operator
        NDArray&  operator=(NDArray&& other) noexcept {
            if (this == &other)
                return *this;

            _isView = other._isView;
            _buffer = other._buffer;
            _shapeInfo = other._shapeInfo;
            _shapeInfoD = other._shapeInfoD;
            _context = other._context;
            _dataType = other._dataType;
            _length = other._length;
            _offset = other._offset;

            other._buffer = std::make_shared<DataBuffer>();
            other._shapeInfo = other._shapeInfoD = nullptr;
            other._length = 0;

            return *this;
        }


        ////////////////////////////////////////////////////////////////////////
        //constructor, create empty array at given workspace
        NDArray(sd::memory::Workspace* workspace) {
            _buffer = std::make_shared<DataBuffer>();
            _shapeInfo = nullptr;
            _shapeInfoD = nullptr;
            _offset = 0;
            _context = workspace;
            _length = 0;
        }
 

        ////////////////////////////////////////////////////////////////////////
        // do not allocate memory, memory for array is passed from outside
        NDArray(void* buffer, Nd4jLong* shapeInfo, sd::memory::Workspace* workspace, const bool isBuffAlloc) {

            if (buffer == nullptr && ArrayOptions::arrayType(shapeInfo) != ArrayType::EMPTY)
                throw std::runtime_error("NDArray constructor: can't be initalized with nullptr buffer !");

            if (shapeInfo == nullptr)
                throw std::runtime_error("NDArray constructor: can't be initalized without shapeinfo !");

            if ((int)shapeInfo[0] > MAX_RANK)
                throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32 !");

            _context = workspace;
            _isAttached = workspace != nullptr;
            _offset = 0;

            setShapeInfo(ShapeDescriptor(shapeInfo));

            if (this->isEmpty()) {
                tickReadDevice();
                tickReadHost();
            }
            else {
                _buffer = std::make_shared<DataBuffer>(buffer, lengthOf() * sizeOfT(), dataType(), isBuffAlloc, workspace);
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // do not allocate memory, memory for array is passed from outside
        // we suppose the content of both (device and host) buffers is identical
        NDArray(void* buffer, void* bufferD, Nd4jLong* shapeInfo, sd::memory::Workspace* workspace, const bool isBuffAlloc, const bool isBuffDAlloc) {

            if (shapeInfo == nullptr)
                throw std::runtime_error("NDArray constructor cuda: can't be initalized without shapeinfo");

            if ((int)shapeInfo[0] > MAX_RANK)
                throw std::invalid_argument("NDArray constructor cuda: rank of NDArray can't exceed 32");

            _context = workspace;
            _offset = 0;

            setShapeInfo(ShapeDescriptor(shapeInfo));

            if (!isEmpty())
                _buffer = std::make_shared<DataBuffer>(buffer, bufferD, lengthOf() * sizeOfT(), dataType(), isBuffAlloc, isBuffDAlloc, workspace);
        }

        //////////////////////////////////////////////////////////////////////////
        NDArray(std::shared_ptr<DataBuffer> buffer, const char order, const std::vector<Nd4jLong>& shape, sd::memory::Workspace* workspace=nullptr) {

            if (shape.empty())
                throw std::runtime_error("NDArray constructor: input shape is empty !");

            if ((int)shape.size() > MAX_RANK)
                throw std::invalid_argument("NDArray constructor: rank of NDArray can't exceed 32");

            _context = workspace;
            _offset = 0;

            setShapeInfo(ShapeDescriptor(buffer->getDataType(), order, shape));

            _buffer = buffer;

            _isView = _length * DataTypeUtils::sizeOf(_dataType) < buffer->getLenInBytes();
        }

        NDArray(Nd4jLong* shapeInfo, const bool copyStrides, sd::memory::Workspace* workspace) :
            NDArray(shapeInfo, ArrayOptions::dataType(shapeInfo), copyStrides, workspace) {
        }

        ////////////////////////////////////////////////////////////////////////
        NDArray(std::shared_ptr<DataBuffer> buffer, const ShapeDescriptor& descriptor, sd::memory::Workspace* workspace=nullptr, const Nd4jLong offset=0) {

            _context = workspace;
            _offset = offset;

            setShapeInfo(descriptor);

            _buffer = buffer;

            _isView = offset > 0 || _length * DataTypeUtils::sizeOf(_dataType) < buffer->getLenInBytes();
        }

        //////////////////////////////////////////////////////////////////////////
        ///// IMLEMENTATION OF INLINE METHODS /////
        //////////////////////////////////////////////////////////////////////////
        bool isAttached() {
            return false;
        }
        template <typename T>
        T* bufferAsT() const { 

            return reinterpret_cast<T*>(getBuffer());
        }
        Nd4jLong  sizeAt(const int dim) const {

            if (dim >= this->rankOf() || dim < -this->rankOf())
                throw std::runtime_error("Bad size index requested");

            if (dim >= 0)
                return shape::shapeOf(_shapeInfo)[dim];
            else
                return shape::shapeOf(_shapeInfo)[this->rankOf() + dim];
        }

        NDArray  operator()(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape=false)  const {

            std::vector<Nd4jLong> idxRanges(2 * rankOf());

            const auto rank = rankOf();
            const auto subArrRank = static_cast<int>(dimsToExclude.size());

            if (subArrRank > rank)
                throw std::invalid_argument("NDArray::operator(const Nd4jLong subArrIdx, const std::vector<int>& dimsToExclude, bool keepUnitiesInShape): static method: dimsToExclude is empty or has size > rank of array !");

            memset(idxRanges.data(), 0, 2 * rank * sizeof(Nd4jLong));

            // subArrRank == 0 means whole array, idxRanges should contain zeros only

            if (subArrRank != 0) {

                std::vector<Nd4jLong> shapeOfSubArr(subArrRank), indexes(subArrRank);
                for (int i = 0; i < subArrRank; ++i)
                    shapeOfSubArr[i] = sizeAt(dimsToExclude[i]);

                shape::index2coords(subArrIdx, subArrRank, shapeOfSubArr.data(), indexes.data());

                for (int i = 0; i < subArrRank; ++i) {
                    int currIdx = 2 * dimsToExclude[i];
                    idxRanges[currIdx] = indexes[i];
                    idxRanges[currIdx + 1] = indexes[i] + 1;
                }
            }

            return (*this)(idxRanges, keepUnitiesInShape);
        }

 
        NDArray  operator()(const std::vector<Nd4jLong>& idx, const bool keepUnitiesInShape=false, const bool isStrided=false)  const {
            if (isEmpty())
                throw std::invalid_argument("NDArray::operator(sub-arrays): array is empty !");

            // Nd4jLong *outShapeInfo = nullptr;
            //     ALLOCATE(outShapeInfo, workspace, shape::shapeInfoLength(inShapeInfo), Nd4jLong);

            int numOfUntiesInSubArrShape = 0;

            Nd4jLong* subArrShapeInfo = nullptr;

            if (!keepUnitiesInShape) {

                int n(isStrided ? 3 : 2), first, last;

                // calculate the number of unities in shape
                for (uint d = 0; d < rankOf(); ++d) {

                    if (idx[n * d] != idx[n * d + 1]) {

                        first = idx[n * d] >= 0 ? idx[n * d] : idx[n * d] + sizeAt(d) + 1;
                        last = idx[n * d + 1] >= 0 ? idx[n * d + 1] : idx[n * d + 1] + sizeAt(d) + 1;
                        if (last - first == 1)
                            ++numOfUntiesInSubArrShape;
                    }
                }
            }

            ALLOCATE(subArrShapeInfo,nullptr, shape::shapeInfoLength(rankOf() - numOfUntiesInSubArrShape), Nd4jLong);

            Nd4jLong offset;

            shape::calcSubArrShapeInfoAndOffset(idx.data(), getShapeInfo(), subArrShapeInfo, offset, keepUnitiesInShape, isStrided, numOfUntiesInSubArrShape);

            NDArray result(_buffer, ShapeDescriptor(subArrShapeInfo), nullptr, offset + getBufferOffset());
            result._isView = true;

            RELEASE(subArrShapeInfo,nullptr);

            return result;
        }

        ////////////////////////////////////////////////////////////////////////
        NDArray  subarray(const std::initializer_list<NDIndex*>& idx) const {

            const int idxSize = idx.size();
            if (idxSize != this->rankOf())
                throw std::runtime_error("NDArray::subarray: number of indices should match the array rank");

            std::vector<Nd4jLong> indexes(3 * idxSize);

            // convert NDIndex to vector
            int d = 0;
            for (const auto& item : idx) {

                if (item->isAll()) {
                    indexes[3 * d] = 0;                             // first
                    indexes[3 * d + 1] = 0;                             // last
                    indexes[3 * d + 2] = 1;                             // stride
                }
                else if (item->isPoint()) {
                    indexes[3 * d] = item->getIndices().at(0);      // first
                    indexes[3 * d + 1] = indexes[3 * d] + 1;            // last
                    indexes[3 * d + 2] = 1;                             // stride
                }
                else if (item->isInterval()) {
                    indexes[3 * d] = item->getIndices().at(0);      // first
                    indexes[3 * d + 1] = item->getIndices().size();     // last
                    indexes[3 * d + 2] = item->stride();                // stride
                }
                else {
                    indexes[3 * d] = item->getIndices().at(0);      // first
                    indexes[3 * d + 1] = item->getIndices().at(1);      // last
                    indexes[3 * d + 2] = item->getIndices().at(2);      // stride
                }
                ++d;
            }

            // release NDIndices
            for (auto i : idx)
                delete i;

            return NDArray((*this)(indexes, true, true));
        }

        void*  platformBuffer() { return buffer(); }
        void*  getPlatformBuffer() const { return getBuffer(); }

        Nd4jLong*  getPlatformShapeInfo() const { return getShapeInfo(); }
        Nd4jLong* platformShapeInfo() { return shapeInfo(); }


        bool isC() const {
            // TODO: this method must be implemented once we add support for complex numbers
            return false;
        }

        //////////////////////////////////////////////////////////////////////////
        bool isS() const {
            return (dataType() == DataType::UTF8 ||
                dataType() == DataType::UTF16 ||
                dataType() == DataType::UTF32);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isR() const {
            auto xType = ArrayOptions::dataType(this->_shapeInfo);
            return xType == FLOAT32 || xType == HALF || xType == DOUBLE || xType == FLOAT8 || xType == BFLOAT16;
        }

        //////////////////////////////////////////////////////////////////////////
        bool isZ() const {
            // TODO: decide if we really want to exclude Bool here
            return !isC() && !isR() && !isB() && !isS();
        }

        //////////////////////////////////////////////////////////////////////////
        bool isB() const {
            return ArrayOptions::dataType(this->_shapeInfo) == BOOL;
        }

        //////////////////////////////////////////////////////////////////////////
        template<typename T>
        std::string toStringValue(T value) {
            std::ostringstream os;
            //throw the value into the string stream
            os << value;
            os << value;
            //convert the string stream into a string and return
            return os.str();
        }

        template <typename T>
        T  e(const Nd4jLong i, const Nd4jLong j) const {

            if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
                throw std::invalid_argument("NDArray::e(i,j): one of input indexes is out of array length or rank!=2 !");

            const Nd4jLong coords[2] = { i, j };
            const auto rp = shape::getOffset(getShapeInfo(), coords);
            switch (dataType()) {
            case sd::DataType::BOOL:
                return templatedGet<bool, T>(getBuffer(), rp);
                break; 
            case sd::DataType::FLOAT32:
                return templatedGet<float, T>(getBuffer(), rp);
                break;
            case sd::DataType::DOUBLE:
                return templatedGet<double, T>(getBuffer(), rp);
                break;
            case sd::DataType::INT8:
                return templatedGet<int8_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::INT16:
                return templatedGet<int16_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::INT32:
                return templatedGet<int32_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::INT64:
                return templatedGet<int64_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::UINT8:
                return templatedGet<uint8_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::UINT16:
                return templatedGet<uint16_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::UINT32:
                return templatedGet<uint32_t, T>(getBuffer(), rp);
                break;
            case sd::DataType::UINT64:
                return templatedGet<uint64_t, T>(getBuffer(), rp);
                break;

            default:
#ifndef __CUDA_ARCH__
                throw std::runtime_error("Can't set unknown data type");
#else
                printf("Can't set unknown data type");
#endif
            }
            return  (T)0;
        }

        template <typename T>
        T  e(const Nd4jLong i) const {

            const auto rp = getOffset(i); 
            switch (dataType()) {
                 case sd::DataType::BOOL:
                     return templatedGet<bool, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::FLOAT32:
                     return templatedGet<float, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::DOUBLE:
                     return templatedGet<double, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::INT8:
                     return templatedGet<int8_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::INT16:
                     return templatedGet<int16_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::INT32:
                     return templatedGet<int32_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::INT64:
                     return templatedGet<int64_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::UINT8:
                     return templatedGet<uint8_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::UINT16:
                     return templatedGet<uint16_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::UINT32:
                     return templatedGet<uint32_t, T>(getBuffer(), rp);
                     break;
                 case sd::DataType::UINT64:
                     return templatedGet<uint64_t, T>(getBuffer(), rp);
                     break;
                 
                 default:
#ifndef __CUDA_ARCH__
                     throw std::runtime_error("Can't set unknown data type");
#else
                     printf("Can't set unknown data type");
#endif
                 }
            return (T)0  ;

        }

        //////////////////////////////////////////////////////////////////////////
        std::string asIndexedString(Nd4jLong limit) {
            std::ostringstream os;
            os << "[";
            if (limit < 1 || limit > this->lengthOf())
                limit = this->lengthOf();
            for (Nd4jLong e = 0; e < limit; e++) {
                os << toStringValue(this->e<float>(e));
                if (e < limit - 1)
                    os << ", ";
            }
            os << "]";
            return os.str();
        }

        //////////////////////////////////////////////////////////////////////////
        std::string asString(Nd4jLong limit) {
            std::ostringstream os;
            os << "[";
            if (limit < 1 || limit > this->lengthOf())
                limit = this->lengthOf();
            for (Nd4jLong e = 0; e < limit; e++) {
                if (this->isR())
                    os << toStringValue(this->e<float>(e));
                else if (this->isZ())
                    os << toStringValue(this->e<Nd4jLong>(e));
                else if (this->isB())
                    os << toStringValue(this->e<bool>(e));
                else if (this->isS()) {
                    printf("minimal nd --missing---");
                }
                if (e < limit - 1)
                    os << ", ";
            }
            os << "]";
            return os.str();
        }

        static void printFormatted(NDArray const* arr, int depth, int limit=-1) {

            if (arr->rankOf() == 1) {
                printf("[ ");
                for (Nd4jLong i = 0; i < arr->lengthOf(); ++i) {
                    if (arr->isR())
                        printf("%f, ", arr->e<float>(i));
                    else if (arr->isZ())
                        printf("%lld, ", arr->e<Nd4jLong>(i));
                    else if (arr->isB())
                        printf("%s, ", arr->e<bool>(i) ? "true" : "false");
                    else if (arr->isS()) {
                        printf("\"minimal nd --missing--\", " );
                    }
                }
                printf("]\n");
            }
            else if (arr->rankOf() == 2) {
                Nd4jLong rows = arr->rows();
                Nd4jLong cols = arr->columns();
                char* padding = new char[depth + 1];
                memset(padding, ' ', depth);
                padding[depth] = 0;
                printf("[");
                for (Nd4jLong row = 0; row < rows; ++row) {
                    if (row && depth > 0)
                        printf("%s", padding);
                    printf("[");
                    Nd4jLong colLimit = cols > limit ? cols : limit;
                    for (Nd4jLong col = 0; col < colLimit; ++col) {
                        if (col)
                            printf(", ");
                        if (arr->isR())
                            printf("%f", arr->e<float>(row, col));
                        else if (arr->isZ())
                            printf("%lld", arr->e<Nd4jLong>(row, col));
                        else if (arr->isB())
                            printf("%s", arr->e<bool>(row, col) ? "true" : "false");
                        else if (arr->isS()) {
                            printf("\"%s\"",  "minimal nd --missing--");
                        }
                    }
                    if (row < rows - 1)
                        printf("]\n");
                    else
                        printf("]");
                }
                printf("]");
                if (padding)
                    delete[] padding;
            }
            else {
                //std::unique_ptr<ResultSet> arrs(arr->allTensorsAlongDimension({0}));
                size_t restCount = 2;
                printf("[");
                restCount = ShapeUtils::getNumOfSubArrs(arr->getShapeInfo(), { 0 });
                for (size_t arrIndex = 0; arrIndex < restCount; ++arrIndex) {
                    NDArray subArr = (*arr)(arrIndex, { 0 });
                    printFormatted(&subArr, depth + 1, limit);
                    if (arrIndex < restCount - 1) {
                        for (Nd4jLong i = 1; i < arr->rankOf(); ++i)
                            printf("\n");
                        for (Nd4jLong i = 0; i < depth - 2; ++i)
                            printf(" ");
                    }
                }
                printf("]");
            }
        }

        void  printBuffer(const char* msg=nullptr, Nd4jLong limit=-1, const bool sync=false) const {
            if (sync)
                syncToHost();

            if (limit == -1)
                limit = (int)this->lengthOf();

            if (msg != nullptr)
                printf("%s: [", msg);
            else
                printf("[");
            if (this->isR()) {
                for (Nd4jLong e = 0; e < limit; e++) {
                    if (e)
                        printf(", ");
                    printf("%f", this->e<float>(e));
                }
            }
            else if (this->isZ()) {
                for (Nd4jLong e = 0; e < limit; e++) {
                    if (this->dataType() != sd::DataType::INT64 && this->dataType() != sd::DataType::UINT64)
                        printf("%d", this->e<int>(e));
                    else
                        printf("%llu", this->e<Nd4jLong>(e));
                    if (e < limit - 1)
                        printf(", ");
                }
            }
            else if (this->isB()) {
                for (Nd4jLong e = 0; e < limit; e++) {
                    if (this->e<bool>(e))
                        printf("true");
                    else
                        printf("false");
                    if (e < limit - 1)
                        printf(", ");
                }
            }
            else if (this->isS()) {
                printf("minimal nd --missing---%d\n", 0);
            }
            printf("]\n");
            fflush(stdout);
        }

        void printIndexedBuffer(const char* msg=nullptr, Nd4jLong limit=-1) const {

            syncToHost();

            Nd4jLong rank = this->rankOf();

            bool rowFlag = (rank < 2) || (rank == 2 && this->sizeAt(0) == 1);

            if (msg)
                printf("%s: ", msg);

            if (this->isEmpty()) {
                printf("Empty\n");
            }
            else if (this->rankOf() == 0) {
                if (this->isZ())
                    printf("%lld\n", this->e<Nd4jLong>(0));
                else if (this->isR())
                    printf("%f\n", this->e<float>(0));
                else if (this->isB()) {
                    printf("%s\n", this->e<bool>(0) ? "true" : "false");
                }
                else if (this->isS()) {
                    // todo do we need this
                    printf("not supported in minimal nd %d", 0);
                }
            }
            else if (rowFlag && ews() == 1)
                printBuffer(nullptr, limit);
            else {
                if (msg)
                    printf("\n");
                printFormatted(this, 1, limit);
                printf("\n");
            }
            fflush(stdout);
        }


        void printShapeInfo(const char* msg=nullptr) const {

            int rank = shape::rank(_shapeInfo);
            int lim = shape::shapeInfoLength(rank);

            if (msg != nullptr)
                printf("shapeInfo %s: [", msg);
            else
                printf("shapeInfo: [");

            printf("%i,  ", rank);
            for (int i = 1; i < shape::shapeInfoLength(rank) - 3; i++) {
                if (i == rank + 1)
                    printf("  ");
                printf("%lld,", _shapeInfo[i]);
            }
            printf("  %lld,", shape::type(_shapeInfo));
            printf("%lld,", shape::elementWiseStride(_shapeInfo));
            printf("%lld]\n", (Nd4jLong)shape::order(_shapeInfo));

            fflush(stdout);
        }

        bool  reshapei(const std::vector<Nd4jLong>& cshape, const char order='c', const bool copyToNewBuff=false) {

            // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary
            if (order == ordering() && shape::shapeEquals(rankOf(), shapeOf(), cshape.size(), cshape.data()))
                return true;

            const bool isOutShapeEmpty = std::find(cshape.begin(), cshape.end(), 0) != cshape.end();

            if (isEmpty() && !isOutShapeEmpty)
                throw std::invalid_argument("NDArray::reshapei: can't reshape empty array to non-empty !");
            if (!isEmpty() && isOutShapeEmpty)
                throw std::invalid_argument("NDArray::reshapei: can't reshape non-empty array to empty !");
            if (isEmpty() && isOutShapeEmpty) {
                Nd4jLong* shapeInfoNew = ShapeBuilders::emptyShapeInfo(dataType(), order, cshape, nullptr);
                setShapeInfo(shapeInfoNew);
                RELEASE(shapeInfoNew, nullptr);
                return true;
            }

            std::vector<Nd4jLong> shape(cshape);
            int rank = shape.size();

            // looking for negative in shape

            int numberNegativesOnes = 0;

            Nd4jLong* shape_ = shape.data();
            for (int i = 0; i < (int)shape.size(); i++) {
                if (shape[i] < 0) {
                    if (numberNegativesOnes >= 1)
                        throw std::runtime_error("NDArray::reshapei: only one dimension can be negative at once");

                    numberNegativesOnes++;

                    int shapeLength = 1;
                    for (int j = 0; j < (int)shape.size(); j++)
                        if (i != j)
                            shapeLength *= shape_[j];

                    Nd4jLong realShape = std::abs(lengthOf() / shapeLength);
                    auto thisNewShape = new Nd4jLong[shape.size()];

                    for (int j = 0; j < (int)shape.size(); j++)
                        if (i != j)
                            thisNewShape[j] = shape_[j];
                        else
                            thisNewShape[j] = realShape;

                    shape_ = thisNewShape;
                }
            }

            for (int e = 0; e < (int)shape.size(); e++)
                shape[e] = shape_[e];

            if (numberNegativesOnes > 0)
                delete[] shape_;

            Nd4jLong arrLength = 1;
            for (const auto& item : shape)
                arrLength *= item;

            if (platformBuffer() == nullptr || arrLength != this->lengthOf()) {
                this->printShapeInfo("Mismatched shape");
                //printf("Shape requested: ", shape);
                nd4j_debug("Requested length in reshape: %lli; Existing length: %lli;\n", arrLength, this->lengthOf());
                throw std::runtime_error("NDArray::reshapei: bad input shape!");
            }

            Nd4jLong* shapeInfoNew;
            ALLOCATE(shapeInfoNew, nullptr, shape::shapeInfoLength(rank), Nd4jLong);

            bool canReshape = shape::reshapeC(shapeInfo(), order, shape.size(), shape.data(), shapeInfoNew);

            if (canReshape) {
                setShapeInfo(shapeInfoNew);
            }
            else {
                throw std::runtime_error("not implemented");
            }

            RELEASE(shapeInfoNew, nullptr);

            return canReshape;
        }

        template <typename T, typename R>
        FORCEINLINE R templatedGet(void* buffer, Nd4jLong index) const {
            auto b = reinterpret_cast<T*>(buffer);
            auto v = static_cast<R>(b[index]);
            return v;
        }

        //////////////////////////////////////////////////////////////////////////
        void setShapeInfo(Nd4jLong* shapeInfo) {
            auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(shapeInfo);
            _shapeInfo = buffer.primaryAsT<Nd4jLong>();
            _shapeInfoD = buffer.specialAsT<Nd4jLong>();

            if (shapeInfo != nullptr) {
                _dataType = ArrayOptions::dataType(_shapeInfo);
                if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
                    _length = 0;
                else
                    _length = shape::length(_shapeInfo);
            }
            else {
                _dataType = sd::DataType::INHERIT;
                _length = 0;
            }
        }

        //////////////////////////////////////////////////////////////////////////
        void setShapeInfo(Nd4jLong* shapeInfo, const sd::DataType dtype) {
            auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(shapeInfo);
            _shapeInfo = buffer.primaryAsT<Nd4jLong>();
            _shapeInfoD = buffer.specialAsT<Nd4jLong>();

            if (shapeInfo != nullptr) {
                _dataType = dtype;
                if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
                    _length = 0;
                else
                    _length = shape::length(_shapeInfo);
            }
            else {
                _dataType = sd::DataType::INHERIT;
                _length = 0;
            }
        }

        void  setShapeInfo(const ShapeDescriptor& descriptor) {

            auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(const_cast<ShapeDescriptor&>(descriptor));

            _shapeInfo = reinterpret_cast<Nd4jLong*>(shapeBuffer.primary());
#ifdef __CUDABLAS__
            _shapeInfoD = reinterpret_cast<Nd4jLong*>(shapeBuffer.special());
#endif

            if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
                _length = 0;
            else
                _length = shape::length(_shapeInfo);

            _dataType = ArrayOptions::dataType(_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        void  setShapeInfo(const ConstantDataBuffer& shapeBuffer) {

            _shapeInfo = reinterpret_cast<Nd4jLong*>(const_cast<ConstantDataBuffer&>(shapeBuffer).primary());
#ifdef __CUDABLAS__
            _shapeInfoD = reinterpret_cast<Nd4jLong*>(const_cast<ConstantDataBuffer&>(shapeBuffer).special());
#endif

            if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
                _length = 0;
            else
                _length = shape::length(_shapeInfo);

            _dataType = ArrayOptions::dataType(_shapeInfo);
        }


        //////////////////////////////////////////////////////////////////////////
        char ordering() const {
            return shape::order(_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isView() const {
            return _isView;
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong* shapeOf() const {
            return shape::shapeOf(_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong* stridesOf() const {
            return shape::stride(_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        int rankOf() const {
            return shape::rank(_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong lengthOf() const {
            return _length;
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong rows() const {
            if (this->rankOf() == 1)
                return 1;

            if (this->rankOf() > 2)
                throw std::runtime_error("Array with rank > 2 can't have rows");

            return shapeOf()[0];
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong columns() const {
            if (this->rankOf() == 1)
                return this->lengthOf();

            if (this->rankOf() > 2)
                throw std::runtime_error("Array with rank > 2 can't have columns");

            return shapeOf()[1];
        }

        //////////////////////////////////////////////////////////////////////////

        size_t sizeOfT() const {
            return DataTypeUtils::sizeOfElement(_dataType);
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong ews() const {
            if (this->isEmpty() || this->rankOf() == 0)
                return 1;

            return shape::elementWiseStride(_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool nonNull() const {
            if (isEmpty())
                return true;
             
                return getDataBuffer()->special() != nullptr && getSpecialShapeInfo() != nullptr;
                 
        }

        //////////////////////////////////////////////////////////////////////////
        bool isMatrix() const {
            if (isEmpty())
                return false;

            return 0 != shape::isMatrix(this->_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isVector() const {
            if (isEmpty())
                return false;
            if (rankOf() == 1)
                return true;
            return !isScalar() && shape::isVector(this->_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isColumnVector() const {
            if (isEmpty())
                return false;

            return !isScalar() && shape::isColumnVector(this->_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isRowVector() const {
            if (isEmpty())
                return false;

            // 1D edge case
            if (shape::rank(this->_shapeInfo) == 1)
                return true;

            return !isScalar() && shape::isRowVector(this->_shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isCommonVector(int& posOfNonUnityDim) const {

            return shape::isCommonVector(_shapeInfo, posOfNonUnityDim);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isScalar() const {
            return 0 != shape::isScalar(this->_shapeInfo);
        }

  

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong FORCEINLINE memoryFootprint() {
            Nd4jLong size = this->lengthOf() * this->sizeOfT();
            size += shape::shapeInfoByteLength(this->rankOf());
            return size;
        }

        //////////////////////////////////////////////////////////////////////////
        // still the definition of inline function must be in header file
        bool isSameShape(const std::vector<Nd4jLong>& shape) const {
            if (this->isScalar() && shape.size() == 1 && shape[0] == 0)
                return true;
            if (this->rankOf() != (int)shape.size())
                return false;
            for (int e = 0; e < this->rankOf(); e++) {
                if (this->shapeOf()[e] != shape[e] && shape[e] != -1)
                    return false;
            }
            return true;
        }

  

        //////////////////////////////////////////////////////////////////////////
        bool isSameShape(const NDArray* other) const {
            if (this->isEmpty() != other->isEmpty())
                return false;

            return isSameShape(std::vector<Nd4jLong>(other->_shapeInfo + 1, other->_shapeInfo + 1 + other->_shapeInfo[0]));
        }

        //////////////////////////////////////////////////////////////////////////
        bool isSameShape(const NDArray& other) const {
            return isSameShape(&other);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isSameShape(const std::initializer_list<Nd4jLong>& other) const {
            return isSameShape(std::vector<Nd4jLong>(other));
        }

        //////////////////////////////////////////////////////////////////////////
        bool areSameShapeAndType(const NDArray& other) const {

            if (rankOf() != other.rankOf() || _dataType != other._dataType)
                return false;

            for (int i = 0; i < rankOf(); ++i)
                if (sizeAt(i) != other.sizeAt(i))
                    return false;

            return true;
        }

        //////////////////////////////////////////////////////////////////////////
        // returns true if these two NDArrays have same _shapeInfo
        // still the definition of inline function must be in header file

        bool isSameShapeStrict(const NDArray& other) const {
            return shape::equalsStrict(_shapeInfo, other._shapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        bool isEmpty() const {
            if (this->_shapeInfo == nullptr)
                return false;

            return ArrayOptions::arrayType(this->getShapeInfo()) == ArrayType::EMPTY;
        }

        //////////////////////////////////////////////////////////////////////////
        bool operator==(const NDArray& other) const {
            // if (this->dataType() != other.dataType())    // this comparison is already present in equalsTo
            //         return false;

            if (!this->isSameShape(&other))
                return false;

            return this->equalsTo(&other);
        }

        //////////////////////////////////////////////////////////////////////////
        bool operator!=(const NDArray& other) const {
            if (this->dataType() != other.dataType())
                return true;

            if (!this->isSameShape(&other))
                return true;

            return !this->equalsTo(&other);
        }

        //////////////////////////////////////////////////////////////////////////
        DataType dataType() const {
            return _dataType;
            // return ArrayOptions::dataType(_shapeInfo);
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        T& t(const Nd4jLong i) {

            // if (i >= _length)
            //     throw std::invalid_argument("NDArray::t(i): input index is out of array length !");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            tickWriteHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(getOffset(i))));
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        T& t(const Nd4jLong i, const Nd4jLong j) {

            if (rankOf() != 2 || i >= sizeAt(0) || j >= sizeAt(1))
                throw std::invalid_argument("NDArray::t(i,j): one of input indexes is out of array length or rank!=2 !");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i,j): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            Nd4jLong coords[2] = { i, j };
            auto offset = shape::getOffset(getShapeInfo(), coords);
            tickWriteHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
        }

        template <typename T>
        T& t(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) {

            if (rankOf() != 3 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2))
                throw std::invalid_argument("NDArray::t(i,j,k): one of input indexes is out of array length or rank!=3!");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i,j,k): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            Nd4jLong coords[3] = { i, j, k };
            auto offset = shape::getOffset(getShapeInfo(), coords);
            tickWriteHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
        }

        template <typename T>
        T& t(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong w) {

            if (rankOf() != 4 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2) || w >= sizeAt(3))
                throw std::invalid_argument("NDArray::t(i,j,k,w): one of input indexes is out of array length or rank!=4 !");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i,j,k,w): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            Nd4jLong coords[4] = { i, j, k, w };
            auto offset = shape::getOffset(getShapeInfo(), coords);
            tickWriteHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        T t(const Nd4jLong i) const {

            // if (i >= _length)
            //     throw std::invalid_argument("NDArray::t(i): input index is out of array length !");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            tickReadHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(getOffset(i))));
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        T t(const Nd4jLong i, const Nd4jLong j) const {

            if (rankOf() != 2 || i >= sizeAt(0) || j >= sizeAt(1))
                throw std::invalid_argument("NDArray::t(i,j): one of input indexes is out of array length or rank!=2 !");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i,j): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            Nd4jLong coords[2] = { i, j };
            auto offset = shape::getOffset(getShapeInfo(), coords);
            tickReadHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
        }

        template <typename T>
        T t(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k) const {

            if (rankOf() != 3 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2))
                throw std::invalid_argument("NDArray::t(i,j,k): one of input indexes is out of array length or rank!=3!");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i,j,k): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            Nd4jLong coords[3] = { i, j, k };
            auto offset = shape::getOffset(getShapeInfo(), coords);
            tickReadHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
        }

        template <typename T>
        T t(const Nd4jLong i, const Nd4jLong j, const Nd4jLong k, const Nd4jLong w) const {

            if (rankOf() != 4 || i >= sizeAt(0) || j >= sizeAt(1) || k >= sizeAt(2) || w >= sizeAt(3))
                throw std::invalid_argument("NDArray::t(i,j,k,w): one of input indexes is out of array length or rank!=4!");
            if (DataTypeUtils::fromT<T>() != _dataType)
                throw std::invalid_argument("NDArray::t(i,j,k,w): type of array is not equal to template type T!");

            if (!isActualOnHostSide())
                syncToHost();

            Nd4jLong coords[4] = { i, j, k, w };
            auto offset = shape::getOffset(getShapeInfo(), coords);
            tickReadHost();
            return *(reinterpret_cast<T*>(bufferWithOffset(offset)));
        }

#ifndef __JAVACPP_HACK__
        ////////////////////////////////////////////////////////////////////////
        std::shared_ptr<DataBuffer> getDataBuffer() const {
            return _buffer;
        }

        ////////////////////////////////////////////////////////////////////////
        std::shared_ptr<DataBuffer> dataBuffer() {
            return _buffer;
        }
#endif

        ////////////////////////////////////////////////////////////////////////
        void* getBuffer() const {

            return _buffer->primary() != nullptr ? static_cast<int8_t*>(_buffer->primary()) + (_offset * sizeOfT()) : nullptr;
        }

        //////////////////////////////////////////////////////////////////////////
        void* buffer() {
            return _buffer->primary() != nullptr ? static_cast<int8_t*>(_buffer->primary()) + (_offset * sizeOfT()) : nullptr;
        }

        ////////////////////////////////////////////////////////////////////////
        Nd4jLong* getShapeInfo() const {
            return _shapeInfo;
        }

        //////////////////////////////////////////////////////////////////////////
        Nd4jLong* shapeInfo() {
            return _shapeInfo;
        }

        ////////////////////////////////////////////////////////////////////////
        Nd4jLong* specialShapeInfo() {
            if (_shapeInfoD == nullptr)
                return _shapeInfo;
            // FIXME: this should be fixed once CUDA backend added
            return _shapeInfoD;
        }

        ////////////////////////////////////////////////////////////////////////
        Nd4jLong getBufferOffset() const {
            return _offset;
        }

        ////////////////////////////////////////////////////////////////////////
        Nd4jLong bufferOffset() {
            return _offset;
        }

        ////////////////////////////////////////////////////////////////////////
        Nd4jLong* getSpecialShapeInfo() const {
            if (_shapeInfoD == nullptr)
                return _shapeInfo;
            // FIXME: this should be fixed once CUDA backend added
            return _shapeInfoD;
        }
        void*  bufferWithOffset(Nd4jLong offset) const {

            return getBuffer() != nullptr ? static_cast<int8_t*>(getBuffer()) + (offset * sizeOfT()) : nullptr;
        }

        Nd4jLong  getOffset(const Nd4jLong i) const {

            if (i >= lengthOf())
                throw std::invalid_argument("NDArray::getOffset: input index is out of array length !");

            return shape::getIndexOffset(i, _shapeInfo);
        }
        void  syncToDevice() const { }
        void  syncToHost() const { }
        void  tickWriteHost() const { }
        void  tickWriteDevice() const { }
        void  tickReadHost() const { }
        void  tickReadDevice() const { }
        void  tickBothActual() const { }
        bool  isActualOnHostSide() const { return true; }
        bool  isActualOnDeviceSide() const { return true; }
        void  makeBothBuffersActual() const { }

        bool  equalsTo(const NDArray& other, double eps=1e-5) const {
            return equalsTo(&other, eps);
        }
        bool  equalsTo(const NDArray* other, double eps = 1e-5) const {

            if (dataType() != other->dataType() || lengthOf() != other->lengthOf())
                return false;

            // we need to be able to compare [1, len] to [len]
            if ((rankOf() == 1 && other->rankOf() == 2) || (rankOf() == 2 && other->rankOf() == 1)) {
                // FIXME: do something here?
            }
            else if (!shape::equalsSoft(getShapeInfo(), other->getShapeInfo()))
                return false;

            //implement

             
            } 

    };

   
}



namespace sd {

    class ND4J_EXPORT NDArrayFactory {
    private:
        template <typename T>
        static void memcpyFromVector(void* ptr, const std::vector<T>& vector);
    public:
        template <typename T>
        static NDArray* empty_(sd::memory::Workspace* = nullptr);

        static NDArray* empty_(sd::DataType dataType, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray empty(sd::memory::Workspace* = nullptr);

        static NDArray empty(sd::DataType dataType, sd::memory::Workspace* = nullptr);
 
        template <typename T>
        static NDArray* create_(const T value, sd::memory::Workspace* = nullptr);
        static NDArray* create_(sd::DataType dtype, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray create(const T value, sd::memory::Workspace* = nullptr);
        static NDArray create(sd::DataType dtype, sd::memory::Workspace* = nullptr);
        template <typename T>
        static NDArray create(DataType type, const T scalar, sd::memory::Workspace* = nullptr);
         
        template <typename T>
        static NDArray* create_(char order, const std::vector<Nd4jLong>& shape, sd::memory::Workspace* = nullptr);

        static NDArray* create_(char order, const std::vector<Nd4jLong>& shape, sd::DataType dataType, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray* create_(char order, const std::vector<Nd4jLong>& shape, const std::vector<T>& data, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray create(char order, const std::vector<Nd4jLong>& shape, const std::vector<T>& data, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray create(char order, const std::vector<Nd4jLong>& shape, sd::memory::Workspace* = nullptr);
        static NDArray create(char order, const std::vector<Nd4jLong>& shape, sd::DataType dtype, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray create(const std::vector<T>& values, sd::memory::Workspace* = nullptr);

#ifndef __JAVACPP_HACK__
        // this method only available out of javacpp
        /**
         * This constructor creates vector of T
         *
         * @param values
         */

        template <typename T>
        static NDArray create(char order, const std::initializer_list<Nd4jLong>& shape, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray create(T* buffer, char order, const std::initializer_list<Nd4jLong>& shape, sd::memory::Workspace* = nullptr);

        template <typename T>
        static NDArray create(char order, const std::vector<Nd4jLong>& shape, const std::initializer_list<T>& data, sd::memory::Workspace* = nullptr);


#endif
    };



#include <type_traits>




    ////////////////////////////////////////////////////////////////////////
    template <>
    ND4J_EXPORT FORCEINLINE NDArray NDArrayFactory::create<bool>(const char order, const std::vector<Nd4jLong>& shape, const std::vector<bool>& data, sd::memory::Workspace* workspace) {

        if ((int)shape.size() > MAX_RANK)
            throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

        ShapeDescriptor descriptor(sd::DataType::BOOL, order, shape);

        if (descriptor.arrLength() != data.size()) {
            nd4j_printf("NDArrayFactory::create: data size [%li] doesn't match shape length [%lld]\n", data.size(), descriptor.arrLength());
            throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
        }

        bool* hostBuffer = nullptr;
        ALLOCATE(hostBuffer, workspace, data.size(), bool);
        std::copy(data.begin(), data.end(), hostBuffer);

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(hostBuffer, data.size() * sizeof(bool), sd::DataType::BOOL, true, workspace);

        NDArray result(buffer, descriptor, workspace);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong>& shape, const std::vector<T>& data, sd::memory::Workspace* workspace) {

        if ((int)shape.size() > MAX_RANK)
            throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

        ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shape);

        if (descriptor.arrLength() != data.size()) {
            nd4j_printf("NDArrayFactory::create: data size [%li] doesn't match shape length [%lld]\n", data.size(), descriptor.arrLength());
            throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
        }

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(data.data(), DataTypeUtils::fromT<T>(), descriptor.arrLength() * sizeof(T), workspace);

        NDArray result(buffer, descriptor, workspace);

        return result;
    }



    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    FORCEINLINE NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong>& shape, sd::memory::Workspace* workspace) {
        return create_(order, shape, DataTypeUtils::fromT<T>(), workspace);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE void NDArrayFactory::memcpyFromVector(void* ptr, const std::vector<T>& vector) {

        memcpy(ptr, vector.data(), vector.size() * sizeof(T));
    }

    template <>
    FORCEINLINE void ND4J_EXPORT NDArrayFactory::memcpyFromVector(void* ptr, const std::vector<bool>& vector) {
        auto p = reinterpret_cast<bool*>(ptr);
        for (Nd4jLong e = 0; e < vector.size(); e++)
            p[e] = vector[e];
    }


#ifndef __JAVACPP_HACK__
    
    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong>& shape, const std::initializer_list<T>& data, sd::memory::Workspace* workspace) {
        std::vector<T> vec(data);
        return create<T>(order, shape, vec, workspace);
    }

#endif

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE  NDArray* NDArrayFactory::create_(const T scalar, sd::memory::Workspace* workspace) {

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(1 * sizeof(T), DataTypeUtils::fromT<T>(), workspace, true);

        NDArray* res = new NDArray(buffer, ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>()), workspace);

        res->bufferAsT<T>()[0] = scalar;

        res->tickWriteHost();
        res->syncToDevice();

        return res;
    }

   
 


    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    FORCEINLINE NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong>& shape, const std::vector<T>& data, sd::memory::Workspace* workspace) {

        return new NDArray(NDArrayFactory::create<T>(order, shape, data, workspace));
    }

 
 
    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::create(const char order, const std::initializer_list<Nd4jLong>& shape, sd::memory::Workspace* workspace) {
        std::vector<Nd4jLong> vec(shape);
        return create<T>(order, vec, workspace);
    }


    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong>& shape, sd::memory::Workspace* workspace) {
        return create(order, shape, DataTypeUtils::fromT<T>(), workspace);
    }


    ////////////////////////////////////////////////////////////////////////
    FORCEINLINE NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong>& shape, sd::DataType dtype, sd::memory::Workspace* workspace) {

        if ((int)shape.size() > MAX_RANK)
            throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32");

        ShapeDescriptor descriptor(dtype, order, shape);

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(descriptor.arrLength() * DataTypeUtils::sizeOfElement(dtype), dtype, workspace);

        NDArray result(buffer, descriptor, workspace);

        buffer->setToZeroBuffers();

        return result;
    }


    ////////////////////////////////////////////////////////////////////////
    FORCEINLINE NDArray NDArrayFactory::create(sd::DataType dtype, sd::memory::Workspace* workspace) {

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(DataTypeUtils::sizeOfElement(dtype), dtype, workspace, true);

        NDArray res(buffer, ShapeDescriptor::scalarDescriptor(dtype), workspace);

        buffer->setToZeroBuffers();
        return res;
    }

    FORCEINLINE NDArray* NDArrayFactory::create_(sd::DataType dtype, sd::memory::Workspace* workspace) {
        auto result = new NDArray();
        *result = NDArrayFactory::create(dtype, workspace);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::create(const std::vector<T>& values, sd::memory::Workspace* workspace) {

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(values.size() * sizeof(T), DataTypeUtils::fromT<T>(), workspace, true);

        NDArray res(buffer, ShapeDescriptor::vectorDescriptor(values.size(), DataTypeUtils::fromT<T>()), workspace);

        memcpyFromVector<T>(res.getBuffer(), values);

        res.tickWriteHost();
        res.syncToDevice();

        return res;
    }


    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE  NDArray* NDArrayFactory::empty_(sd::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, workspace, false);

        RELEASE(shapeInfo, workspace);

        return result;
    }


    FORCEINLINE  NDArray* NDArrayFactory::empty_(sd::DataType dataType, sd::memory::Workspace* workspace) {
 

        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, workspace, false);

        RELEASE(shapeInfo, workspace);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::empty(sd::memory::Workspace* workspace) {
        return empty(DataTypeUtils::fromT<T>(), workspace);
    }


    ////////////////////////////////////////////////////////////////////////
    FORCEINLINE NDArray NDArrayFactory::empty(sd::DataType dataType, sd::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        NDArray result(nullptr, shapeInfo, workspace, false);

        RELEASE(shapeInfo, workspace);

        return result;
    }

  
   
    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    FORCEINLINE NDArray NDArrayFactory::create(T* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, sd::memory::Workspace* workspace) {

        if ((int)shape.size() > MAX_RANK)
            throw std::invalid_argument("NDArrayFactory::create: Rank of NDArray can't exceed 32");

        std::vector<Nd4jLong> shp(shape);
        ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shp);

        std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(buffer, descriptor.arrLength() * sizeof(T), descriptor.dataType(), false, workspace);

        NDArray result(pBuffer, descriptor, workspace);

        return result;
    }





 }