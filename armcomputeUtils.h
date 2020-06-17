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
            Arm_Tensor getArmTensor(const sd::NDArray& arr, arm_compute::DataLayout layout = arm_compute::DataLayout::UNKNOWN);


            void print_tensor(Arm_ITensor& tensor, const char* msg);
            bool isArmcomputeFriendly(const NDArray& arr);

            }

#endif //DEV_TESTSARMCOMPUTEUTILS_H
