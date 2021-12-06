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

#pragma once
#include <common.h>
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
#include <tuple>
 namespace sd{

     using Arm_DataType = arm_compute::DataType;
     using Arm_Tensor = arm_compute::Tensor;
     using Arm_ITensor = arm_compute::ITensor;
     using Arm_TensorInfo = arm_compute::TensorInfo;
     using Arm_TensorShape = arm_compute::TensorShape;
     using Arm_Strides = arm_compute::Strides;
     using Arm_WeightsInfo = arm_compute::WeightsInfo;
     using Arm_PermutationVector = arm_compute::PermutationVector;
     using Arm_DataLayout = arm_compute::DataLayout;

     //utils
     Arm_DataType getArmType(const sd::DataType& dType);

     Arm_TensorInfo getArmTensorInfo(int rank, Nd4jLong* bases, sd::DataType ndArrayType, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

     Arm_TensorInfo getArmTensorInfo(const NDArray& arr, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

     Arm_Tensor getArmTensor(const NDArray& arr, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

     std::tuple<int, int, int, int> getAutoPadding(int rank);

     void copyFromTensor(const Arm_Tensor& inTensor, NDArray& output);
     void copyToTensor(const NDArray& input, Arm_Tensor& outTensor);
     void print_tensor(Arm_ITensor& tensor, const char* msg);
     bool isArmcomputeFriendly(const NDArray& arr);

     std::tuple<int, int, int, int> getAutoPadding(int rank);

     template<typename F>
     class ArmFunction {
     public:

         template<typename ...Args>
         void configure(NDArray* input, NDArray* output, Arm_DataLayout layout, Args&& ...args) {

             auto inInfo = getArmTensorInfo(*input, layout);
             auto outInfo = getArmTensorInfo(*output, layout);
             in.allocator()->init(inInfo);
             out.allocator()->init(outInfo);
             armFunction.configure(&in, &out, std::forward<Args>(args) ...);
             if (in.info()->has_padding() || input->ews()!=1) {
                 //allocate and copy
                 in.allocator()->allocate();
                 //copy 
                 copyToTensor(*input, in);
                 internal_printf("input copy %d\n", 0);
                 internal_print_arm_array(in, "in");
                 internal_print_nd_shape(*input,"shapeInfo");
                 internal_print_nd_array(*input,"INput");
             }
             else {

                 //import buffer
                 void* buff = input->buffer();
                 in.allocator()->import_memory(buff);
                 internal_printf("input import %d\n", 0);
             }
             if (out.info()->has_padding() || output->ews()!=1) {
                 //store pointer to our array to copy after run
                 out.allocator()->allocate();
                 outNd = output;
             }
             else {
                 //import
                 internal_printf("output import %d\n", 0);void* buff = output->buffer();
                 out.allocator()->import_memory(buff);
             }
         }

         void run() {
             armFunction.run();
             if (outNd) {
                 internal_printf("out copy %d\n", 0);
                 copyFromTensor(out, *outNd);
             }
         }

     private:
         Arm_Tensor in;
         Arm_Tensor out;
         NDArray* outNd = nullptr;
         F armFunction{};
     };

     template<typename F>
     class ArmFunctionHasPaddedBuffer {
     public:

         template<typename ...Args>
         void configure(NDArray* input, NDArray* output, Arm_DataLayout layout, Args&& ...args) {

             in = getArmTensor(*input, layout);
             out = getArmTensor(*output, layout);
             armFunction.configure(&in, &out, std::forward<Args>(args) ...);
         }

         void run() {
             armFunction.run();
         }

     private:
         Arm_Tensor in;
         Arm_Tensor out;
         F armFunction{};
     };

     #define    CREATE_IMPORTED
     template<typename F>
     class ArmFunctionWeighted {
     public:

         template<typename ...Args>
         void configure(const NDArray* input, const NDArray* weights, const NDArray* biases, NDArray* output, Arm_DataLayout layout, arm_compute::PermutationVector permuteVector, Args&& ...args) {
#if !defined(CREATE_IMPORTED)
             auto inInfo = getArmTensorInfo(*input, layout);
             in.allocator()->init(inInfo);
#else
             in = getArmTensor(*input, layout);
#endif
             auto wInfo = getArmTensorInfo(*weights, layout);
             Arm_Tensor* bias_ptr = nullptr;
             auto outInfo = getArmTensorInfo(*output, layout);

             w.allocator()->init(wInfo);

             if (biases) {
                 auto bInfo = getArmTensorInfo(*biases, layout);
                 b.allocator()->init(bInfo);
                 bias_ptr = &b;
             }
             out.allocator()->init(outInfo);

             if (permuteVector.num_dimensions() == 0) {
                 armFunction.configure(&in, &w, bias_ptr, &out, std::forward<Args>(args)...);
             }
             else {
                 //configure with permute kernel
                 Arm_TensorShape shape;
                 int rank = permuteVector.num_dimensions();
                 shape.set_num_dimensions(rank);
                 for (int i = 0; i < rank; i++) {
                     shape[i] = wInfo.dimension(permuteVector[i]);
                 }
                 for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
                     shape[i] = 1;
                 }

                 Arm_TensorInfo wPermInfo(shape, 1, wInfo.data_type(), layout);
                 wPerm.allocator()->init(wPermInfo);
                 permuter.configure(&w, &wPerm, permuteVector);
                 armFunction.configure(&in, &wPerm, bias_ptr, &out, std::forward<Args>(args)...);
                 wPerm.allocator()->allocate();
                 runPerm = true;
             }

#if !defined(CREATE_IMPORTED)
             if (in.info()->has_padding() || in->ews!=1) {
                 //allocate and copy
                 in.allocator()->allocate();
                 //copy 
                 copyToTensor(*input, in);
                 internal_printf("input copy %d\n", 0);
             }
             else {
                 //import buffer
                 auto buff = input->buffer();
                 in.allocator()->import_memory((void*)buff);
                 internal_printf("input import %d\n", 0);
             }
#endif

             if (w.info()->has_padding() || weights->ews()!=1) {
                 //allocate and copy
                 w.allocator()->allocate();
                 //copy 
                 copyToTensor(*weights, w);
                 internal_printf("weight copy %d\n", 0);
             }
             else {
                 //import buffer
                 auto buff = weights->buffer();
                 w.allocator()->import_memory((void*)buff);
                 internal_printf("weight import %d\n", 0);
             }

             if (bias_ptr) {
                 //check bias also
                 if (b.info()->has_padding() || biases->ews()!=1) {
                     //allocate and copy
                     b.allocator()->allocate();
                     //copy 
                     copyToTensor(*biases, b);
                     internal_printf("b copy %d\n", 0);
                 }
                 else {
                     //import buffer
                     auto buff = biases->buffer();
                     b.allocator()->import_memory((void*)buff);
                     internal_printf("b import %d\n", 0);
                 }
             }
             if (out.info()->has_padding() || output->ews()!=1) {
                 //store pointer to our array to copy after run
                 out.allocator()->allocate();
                 outNd = output;
                 internal_printf("out copy %d\n", 0);
             }
             else {
                 //import
                 void* buff = output->buffer();
                 out.allocator()->import_memory(buff);
                 internal_printf("output import %d\n", 0);
             }

             internal_print_arm_array(in, "in tensor:");
         }

         void run() {
             if (runPerm) {
                 permuter.run();
             }
             armFunction.run();
             if (outNd) {
                 internal_printf("out copy %d\n", 0);
                 copyFromTensor(out, *outNd);
             }
         }

     private:
         bool runPerm = false;
         Arm_Tensor in;
         Arm_Tensor b;
         Arm_Tensor w;
         Arm_Tensor wPerm;
         Arm_Tensor out;
         NDArray* outNd = nullptr;
         arm_compute::NEPermute permuter;
         F armFunction{};
     };

     template<typename F>
     class ArmFunctionUniversal {
     public:

         template<typename ...Args>
         void configure(NDArray* input, NDArray* output, Arm_DataLayout layout, Args&& ...args) {
             bool inputHasPaddedBuffer = input->hasPaddedBuffer();
             bool outputHasPaddedBuffer = output->hasPaddedBuffer();
             if (inputHasPaddedBuffer) {
                 in = getArmTensor(*input, layout);
                 internal_printf("input is a padded buffer %d\n", 0);
             }
             else {
                 auto inInfo = getArmTensorInfo(*input, layout);
                 in.allocator()->init(inInfo);
             }
             if (outputHasPaddedBuffer) {
                 out = getArmTensor(*output, layout);
                 internal_printf("output is a padded buffer %d\n", 0);
             }
             else {
                 auto outInfo = getArmTensorInfo(*output, layout);
                 out.allocator()->init(outInfo);
             }

             armFunction.configure(&in, &out, std::forward<Args>(args) ...);

             if (!inputHasPaddedBuffer) {
                 if (in.info()->has_padding() || input->ews() != 1) {
                     //allocate and copy
                     in.allocator()->allocate();
                     inputNd = input;
                 }
                 else {
                     //import only for ews()==1
                     in.allocator()->import_memory(input->buffer());
                     internal_printf("input import %d\n", 0);
                 }
             }
             if (!outputHasPaddedBuffer) {
                 if (out.info()->has_padding() || output->ews()!=1) {
                     //store pointer to our array to copy after run
                     out.allocator()->allocate();
                     outNd = output;
                 }
                 else {
                     //import only for ews()==1
                     out.allocator()->import_memory(output->buffer());
                     internal_printf("output import %d\n", 0);
                 }
             }
         }

         void run() {
             if (inputNd) {
                 //copy 
                 copyToTensor(*inputNd, in);
                 internal_printf("input copy %d\n", 0);
                 internal_print_nd_array(*inputNd,"input");
                 internal_print_arm_array(in, "in");
             }
             armFunction.run();
             
             if (outNd) {
                 copyFromTensor(out, *outNd);
                 internal_printf("output copy %d\n", 0);
                 internal_print_arm_array(out, "out");
             }
         }

     private:
         Arm_Tensor in;
         Arm_Tensor out;
         NDArray* inputNd = nullptr;
         NDArray* outNd = nullptr;
         F armFunction{};
     };

     template<typename F>
     class ArmFunctionUniversalWeighted {
     public:

         template<typename ...Args>
         void configure(const NDArray* input, const NDArray* weights, const NDArray* biases, NDArray* output, Arm_DataLayout layout, arm_compute::PermutationVector permuteVector, Args&& ...args) {

             bool inputHasPaddedBuffer = input->hasPaddedBuffer();
             bool weightsHasPaddedBuffer = weights->hasPaddedBuffer();
             bool outputHasPaddedBuffer = output->hasPaddedBuffer();
             bool biasesHasPaddedBuffer = false;
             if (inputHasPaddedBuffer) {
                 in = getArmTensor(*input, layout);
                 internal_printf("input is a padded buffer %d\n", 1);
             }
             else {
                 in.allocator()->init(getArmTensorInfo(*input, layout));
             }
             if (weightsHasPaddedBuffer) {
                 w = getArmTensor(*weights, layout);
                 internal_printf("weights is a padded buffer %d\n", 1);
             }
             else {
                 w.allocator()->init(getArmTensorInfo(*weights, layout));
             }
             if (outputHasPaddedBuffer) {
                 out = getArmTensor(*output, layout);
                 internal_printf("output is a padded buffer %d\n", 1);
             }
             else {
                 out.allocator()->init(getArmTensorInfo(*output, layout));
             }

             Arm_Tensor* bias_ptr = nullptr;

             if (biases) {
                 biasesHasPaddedBuffer = biases->hasPaddedBuffer();
                 if (biasesHasPaddedBuffer) {
                     b = getArmTensor(*biases, layout);
                     internal_printf("biases is a padded buffer %d\n", 1);
                 }
                 else {
                     b.allocator()->init(getArmTensorInfo(*biases, layout));
                 }
                 bias_ptr = &b;
             }

             if (permuteVector.num_dimensions() == 0) {
                 armFunction.configure(&in, &w, bias_ptr, &out, std::forward<Args>(args)...);
             }
             else {
                 //configure with permute kernel
                 Arm_TensorShape shape;
                 int rank = permuteVector.num_dimensions();
                 shape.set_num_dimensions(rank);
                 auto wInfoPtr = w.info();
                 for (int i = 0; i < rank; i++) {
                     shape[i] = wInfoPtr->dimension(permuteVector[i]);
                 }
                 for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
                     shape[i] = 1;
                 }

                 Arm_TensorInfo wPermInfo(shape, 1, wInfoPtr->data_type(), layout);
                 wPerm.allocator()->init(wPermInfo);
                 permuter.configure(&w, &wPerm, permuteVector);
                 armFunction.configure(&in, &wPerm, bias_ptr, &out, std::forward<Args>(args)...);
                 wPerm.allocator()->allocate();
                 runPerm = true;
             }

             //import buffer
             if (!inputHasPaddedBuffer) {
                 if (in.info()->has_padding() || input->ews()!=1) {
                     //allocate and copy
                     in.allocator()->allocate();
                     inputNd = input;
                 }
                 else {
                     //import buffer
                     in.allocator()->import_memory(input->buffer());
                     internal_printf("input import %d\n", 1);
                 }
             }
             if (!weightsHasPaddedBuffer) {
                 if (w.info()->has_padding() || weights->ews()!=1) {
                     //store pointer to our array to copy after run
                     w.allocator()->allocate();
                     wNd = weights;
                 }
                 else {
                     //import
                     w.allocator()->import_memory(weights->buffer());
                     internal_printf("weights import %d\n", 1);
                 }
             }
             if (biases && !biasesHasPaddedBuffer) {
                 if (b.info()->has_padding() || biases->ews()!=1) {
                     //store pointer to our array to copy after run
                     b.allocator()->allocate();
                     bNd = biases;
                 }
                 else {
                     //import
                     b.allocator()->import_memory(biases->buffer());
                     internal_printf("biases import %d\n", 1);
                 }
             }
             if (!outputHasPaddedBuffer) {
                 if (out.info()->has_padding() || output->ews()!=1) {
                     //store pointer to our array to copy after run
                     out.allocator()->allocate();
                     outNd = output;
                 }
                 else {
                     //import
                     out.allocator()->import_memory(output->buffer());
                     internal_printf("output import %d\n", 1);
                 }
             }
         }

         void run() {
             if (inputNd) {
                 //copy 
                 copyToTensor(*inputNd, in);
                 internal_printf("input copy %d\n", 1);
             }
             if (bNd) {
                 //copy 
                 copyToTensor(*bNd, b);
                 internal_printf("biases copy %d\n", 1);
             }
             if (wNd) {
                 //copy 
                 copyToTensor(*wNd, w);
                 internal_printf("weights copy %d\n", 1);
             }
             if (runPerm) {
                 permuter.run();
             }
             armFunction.run();
             if (outNd) {
                 copyFromTensor(out, *outNd);
                 internal_printf("output copy %d\n", 1);
             }
         }

     private:
         bool runPerm = false;
         Arm_Tensor in;
         Arm_Tensor b;
         Arm_Tensor w;
         Arm_Tensor wPerm;
         Arm_Tensor out;
         NDArray* inputNd = nullptr;
         NDArray* wNd = nullptr;
         NDArray* bNd = nullptr;
         NDArray* outNd = nullptr;
         arm_compute::NEPermute permuter;
         F armFunction{};
     };
  }

