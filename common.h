#pragma once
#define ARM_COMPUTE_ASSERTS_ENABLED 1
#define internal_print_arm_array(a,b) print_tensor(a,b)
#define internal_print_nd_array(a,b) ((a).printIndexedBuffer(b))
#define internal_print_nd_shape(a,b) ((a).printShapeInfo(b))
#define internal_printf  nd4j_printf
//#define internal_print_arm_array(a,b)  
//#define internal_print_nd_array(a,b)  
//#define internal_print_nd_shape(a,b)  