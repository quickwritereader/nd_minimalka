/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/


#ifndef DEV_TESTSARMCOMPUTEUTILS_H
#define DEV_TESTSARMCOMPUTEUTILS_H

#include <NdArrayMinimal.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Strides.h>
#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/Validate.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/TensorAllocator.h> 
#include <iostream>

 namespace sd{
            

            using Arm_DataType = arm_compute::DataType;
            using Arm_Tensor = arm_compute::Tensor;
            using Arm_ITensor = arm_compute::ITensor;
            using Arm_TensorInfo = arm_compute::TensorInfo;
            using Arm_TensorShape = arm_compute::TensorShape;
            using Arm_Strides = arm_compute::Strides;
 
            Arm_DataType getArmType(const sd::DataType& dType);

            Arm_TensorInfo getArmTensorInfo(int rank, Nd4jLong* bases, sd::DataType ndArrayType, arm_compute::DataLayout layout = arm_compute::DataLayout::UNKNOWN);

            Arm_TensorInfo getArmTensorInfo(const sd::NDArray& arr, arm_compute::DataLayout layout = arm_compute::DataLayout::UNKNOWN);

            Arm_Tensor getArmTensor(const sd::NDArray& arr, arm_compute::DataLayout layout = arm_compute::DataLayout::UNKNOWN);

            void copyFromTensor(const Arm_Tensor& inTensor, sd::NDArray& output);
            void copyToTensor(const sd::NDArray& input, Arm_Tensor& outTensor);
            void print_tensor(Arm_ITensor& tensor, const char* msg);
            bool isArmcomputeFriendly(const NDArray& arr);


            template<typename F>
            class ArmFunction {
            public:

               template<typename ...Args>
               void configure(NDArray *input , NDArray *output, arm_compute::DataLayout layout, Args&& ...args) {
                   
                   auto inInfo = getArmTensorInfo(*input, layout);
                   auto outInfo = getArmTensorInfo(*output, layout);  
                   in.allocator()->init(inInfo);
                   out.allocator()->init(outInfo);
                   armFunction.configure(&in,&out,std::forward<Args>(args) ...);
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
                   if (out.info()->has_padding()) {
                       //store pointer to our array to copy after run
                       out.allocator()->allocate();
                       outNd = output;
                   }
                   else {
                       //import
                       void* buff = output->getBuffer();
                       out.allocator()->import_memory(buff);
                   }

               }

               void run() {
                   armFunction.run();
                   if (outNd) {
                       copyFromTensor(out, *outNd);
                   }
               }

               private:
                   Arm_Tensor in;
                   Arm_Tensor out;
                   NDArray *outNd=nullptr;
                   F armFunction{};
            };
             
            }

#endif //DEV_TESTSARMCOMPUTEUTILS_H
