#include "kernel_opencv_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvFixedPoseResJacKernel(float* sensor_from_rig,
                                unsigned int sensor_from_rig_num_alloc,
                                float* calib,
                                unsigned int calib_num_alloc,
                                SharedIndex* calib_indices,
                                float* point,
                                unsigned int point_num_alloc,
                                SharedIndex* point_indices,
                                float* pixel,
                                unsigned int pixel_num_alloc,
                                float* pose,
                                unsigned int pose_num_alloc,
                                float* out_res,
                                unsigned int out_res_num_alloc,
                                float* out_calib_jac,
                                unsigned int out_calib_jac_num_alloc,
                                float* const out_calib_njtr,
                                unsigned int out_calib_njtr_num_alloc,
                                float* const out_calib_precond_diag,
                                unsigned int out_calib_precond_diag_num_alloc,
                                float* const out_calib_precond_tril,
                                unsigned int out_calib_precond_tril_num_alloc,
                                float* out_point_jac,
                                unsigned int out_point_jac_num_alloc,
                                float* const out_point_njtr,
                                unsigned int out_point_njtr_num_alloc,
                                float* const out_point_precond_diag,
                                unsigned int out_point_precond_diag_num_alloc,
                                float* const out_point_precond_tril,
                                unsigned int out_point_precond_tril_num_alloc,
                                size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68;
  LoadShared<4, float, float>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r8,
                                         r9);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = 1.00000000000000000e+00;
    r14 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r20, r18 * r21);
    r24 = r16 * r19;
    r23 = fmaf(r6, r24, r23);
    r23 = fmaf(r17, r22, r23);
    r24 = r23 * r23;
    r24 = r14 * r24;
    r25 = r13 + r24;
    r26 = r15 * r21;
    r26 = fmaf(r6, r26, r18 * r20);
    r26 = fmaf(r16, r22, r26);
    r26 = fmaf(r17, r19, r26);
    r27 = r14 * r26;
    r27 = r27 * r26;
    r25 = r25 + r27;
    r7 = fmaf(r10, r25, r7);
    r28 = fmaf(r15, r22, r18 * r19);
    r29 = r17 * r20;
    r28 = fmaf(r6, r29, r28);
    r28 = fmaf(r16, r21, r28);
    r29 = 2.00000000000000000e+00;
    r30 = r29 * r26;
    r31 = r28 * r30;
    r32 = fmaf(r16, r20, r15 * r19);
    r32 = fmaf(r17, r21, r32);
    r32 = fmaf(r6, r32, r18 * r22);
    r22 = r14 * r32;
    r33 = fmaf(r23, r22, r31);
    r34 = r23 * r29;
    r34 = r34 * r28;
    r35 = fmaf(r32, r30, r34);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r36, r37, r38);
    r39 = r15 * r17;
    r39 = r39 * r29;
    r40 = r16 * r18;
    r41 = fmaf(r29, r40, r39);
    r42 = r17 * r18;
    r43 = r15 * r16;
    r43 = r43 * r29;
    r42 = fmaf(r14, r42, r43);
    r44 = r17 * r17;
    r44 = r14 * r44;
    r45 = r16 * r16;
    r45 = fmaf(r14, r45, r13);
    r46 = r44 + r45;
    r7 = fmaf(r11, r33, r7);
    r7 = fmaf(r12, r35, r7);
    r7 = fmaf(r38, r41, r7);
    r7 = fmaf(r37, r42, r7);
    r7 = fmaf(r36, r46, r7);
    r46 = r23 * r29;
    r46 = fmaf(r32, r46, r31);
    r8 = fmaf(r10, r46, r8);
    r31 = r17 * r18;
    r31 = fmaf(r29, r31, r43);
    r44 = r13 + r44;
    r43 = r15 * r15;
    r43 = r14 * r43;
    r44 = r44 + r43;
    r42 = r16 * r17;
    r42 = r42 * r29;
    r41 = r15 * r18;
    r41 = fmaf(r14, r41, r42);
    r30 = r23 * r30;
    r47 = fmaf(r28, r22, r30);
    r24 = r13 + r24;
    r48 = r28 * r28;
    r48 = r14 * r48;
    r24 = r24 + r48;
    r8 = fmaf(r36, r31, r8);
    r8 = fmaf(r37, r44, r8);
    r8 = fmaf(r38, r41, r8);
    r8 = fmaf(r12, r47, r8);
    r8 = fmaf(r11, r24, r8);
    r41 = fmaf(r8, r8, r7 * r7);
    r44 = sqrtf(r41);
    r44 = r2 + r44;
    r2 = 1.0 / r44;
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r31,
                       r49,
                       r50,
                       r51);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r52 = r31 * r7;
    r53 = r2 * r52;
    r54 = -9.99999999999999955e-07;
    r22 = fmaf(r26, r22, r34);
    r10 = fmaf(r10, r22, r9);
    r40 = fmaf(r14, r40, r39);
    r45 = r43 + r45;
    r43 = r15 * r18;
    r43 = fmaf(r29, r43, r42);
    r42 = r29 * r28;
    r42 = fmaf(r32, r42, r30);
    r27 = r13 + r27;
    r27 = r27 + r48;
    r10 = fmaf(r36, r40, r10);
    r10 = fmaf(r38, r45, r10);
    r10 = fmaf(r37, r43, r10);
    r10 = fmaf(r11, r42, r10);
    r10 = fmaf(r12, r27, r10);
    r12 = fmaf(r10, r10, r41);
    r11 = rsqrtf(r12);
    r43 = r10 * r11;
    r37 = copysign(1.0, r43);
    r37 = fmaf(r54, r37, r43);
    r54 = acosf(r37);
    r43 = r54 * r54;
    r45 = r43 * r43;
    r38 = r54 * r45;
    r40 = fmaf(r51, r38, r54);
    r36 = r54 * r43;
    r48 = r36 * r36;
    r30 = r54 * r48;
    r32 = r45 * r45;
    r54 = r54 * r32;
    r40 = fmaf(r50, r36, r40);
    r40 = fmaf(r0, r30, r40);
    r40 = fmaf(r1, r54, r40);
    r4 = fmaf(r40, r53, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r40 * r2;
    r14 = r49 * r8;
    r5 = fmaf(r14, r3, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r3 = r7 * r40;
    r3 = r3 * r2;
    r39 = r8 * r40;
    r39 = r39 * r2;
    r9 = r36 * r53;
    r26 = r2 * r36;
    r26 = r26 * r14;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r3,
                                          r39,
                                          r9,
                                          r26);
    r34 = r38 * r53;
    r55 = r2 * r38;
    r55 = r55 * r14;
    r56 = r30 * r53;
    r57 = r2 * r30;
    r57 = r57 * r14;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r55,
                                          r56,
                                          r57);
    r58 = r54 * r53;
    r59 = r2 * r14;
    r59 = r59 * r54;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r58,
                                          r59);
    r60 = r7 * r40;
    r4 = r6 * r4;
    r60 = r60 * r2;
    r60 = r60 * r4;
    r61 = r6 * r8;
    r61 = r61 * r40;
    r61 = r61 * r5;
    r61 = r61 * r2;
    r62 = r6 * r5;
    r62 = r62 * r2;
    r62 = r62 * r36;
    r63 = r53 * r4;
    r62 = fmaf(r36, r63, r14 * r62);
    r64 = r6 * r5;
    r64 = r64 * r2;
    r64 = r64 * r38;
    r64 = fmaf(r38, r63, r14 * r64);
    WriteSum4<float, float>((float*)inout_shared, r60, r61, r62, r64);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r6 * r5;
    r62 = r6 * r5;
    r62 = r62 * r2;
    r62 = r62 * r30;
    r62 = fmaf(r30, r63, r14 * r62);
    r61 = r6 * r5;
    r61 = r61 * r2;
    r61 = r61 * r14;
    r63 = fmaf(r54, r63, r54 * r61);
    WriteSum4<float, float>((float*)inout_shared, r62, r63, r4, r64);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r7 * r7;
    r44 = r44 * r44;
    r44 = 1.0 / r44;
    r63 = r40 * r44;
    r64 = r64 * r40;
    r64 = r64 * r63;
    r62 = r8 * r8;
    r62 = r62 * r40;
    r62 = r62 * r63;
    r61 = r44 * r52;
    r61 = r61 * r52;
    r44 = r49 * r44;
    r60 = r8 * r14;
    r44 = r44 * r60;
    r65 = fmaf(r48, r44, r48 * r61);
    r66 = r38 * r38;
    r66 = fmaf(r66, r44, r66 * r61);
    WriteSum4<float, float>((float*)inout_shared, r64, r62, r65, r66);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r30 * r30;
    r65 = fmaf(r65, r44, r65 * r61);
    r62 = r54 * r54;
    r62 = fmaf(r44, r62, r61 * r62);
    WriteSum4<float, float>((float*)inout_shared, r65, r62, r13, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = 0.00000000000000000e+00;
    r64 = r7 * r36;
    r64 = r64 * r52;
    r64 = r64 * r63;
    r67 = r7 * r38;
    r67 = r67 * r52;
    r67 = r67 * r63;
    r68 = r7 * r30;
    r68 = r68 * r52;
    r68 = r68 * r63;
    WriteSum4<float, float>((float*)inout_shared, r62, r64, r67, r68);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r7 * r52;
    r68 = r68 * r54;
    r68 = r68 * r63;
    r36 = r36 * r63;
    r36 = r36 * r60;
    WriteSum4<float, float>((float*)inout_shared, r68, r3, r62, r36);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r38 * r63;
    r38 = r38 * r60;
    r30 = r30 * r63;
    r30 = r30 * r60;
    r54 = r54 * r63;
    r54 = r54 * r60;
    WriteSum4<float, float>((float*)inout_shared, r38, r30, r54, r62);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fmaf(r32, r44, r32 * r61);
    r30 = r48 * r48;
    r30 = fmaf(r30, r44, r30 * r61);
    WriteSum4<float, float>((float*)inout_shared, r39, r54, r66, r30);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r9, r26, r30, r65);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r32 * r32;
    r65 = fmaf(r44, r65, r61 * r65);
    WriteSum4<float, float>((float*)inout_shared, r34, r55, r65, r56);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r57, r58, r59, r62);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = -5.00000000000000000e-01;
    r59 = r10 * r62;
    r58 = r12 * r12;
    r58 = r12 * r58;
    r58 = rsqrtf(r58);
    r59 = r59 * r58;
    r58 = r29 * r22;
    r12 = r29 * r25;
    r57 = r29 * r46;
    r57 = fmaf(r8, r57, r7 * r12);
    r58 = fmaf(r10, r58, r57);
    r58 = fmaf(r58, r59, r22 * r11);
    r37 = r37 * r37;
    r37 = fmaf(r6, r37, r13);
    r37 = rsqrtf(r37);
    r58 = r58 * r37;
    r13 = -9.00000000000000000e+00;
    r13 = r1 * r13;
    r13 = r13 * r32;
    r32 = fmaf(r58, r13, r6 * r58);
    r1 = -3.00000000000000000e+00;
    r1 = r50 * r1;
    r1 = r1 * r43;
    r43 = -5.00000000000000000e+00;
    r43 = r51 * r43;
    r43 = r43 * r45;
    r45 = -7.00000000000000000e+00;
    r45 = r0 * r45;
    r45 = r45 * r48;
    r32 = fmaf(r58, r1, r32);
    r32 = fmaf(r58, r43, r32);
    r32 = fmaf(r58, r45, r32);
    r58 = r31 * r25;
    r58 = r58 * r40;
    r58 = fmaf(r2, r58, r32 * r53);
    r41 = rsqrtf(r41);
    r41 = r62 * r41;
    r41 = r41 * r63;
    r52 = r52 * r41;
    r58 = fmaf(r57, r52, r58);
    r63 = r49 * r46;
    r63 = r63 * r40;
    r62 = r32 * r2;
    r62 = fmaf(r14, r62, r2 * r63);
    r63 = r57 * r14;
    r62 = fmaf(r41, r63, r62);
    r63 = r31 * r33;
    r63 = r63 * r40;
    r48 = r29 * r24;
    r0 = r29 * r33;
    r0 = fmaf(r7, r0, r8 * r48);
    r63 = fmaf(r0, r52, r2 * r63);
    r48 = r29 * r42;
    r48 = fmaf(r10, r48, r0);
    r48 = fmaf(r42, r11, r48 * r59);
    r51 = r6 * r48;
    r50 = r48 * r37;
    r51 = fmaf(r1, r50, r37 * r51);
    r51 = fmaf(r43, r50, r51);
    r51 = fmaf(r45, r50, r51);
    r51 = fmaf(r13, r50, r51);
    r63 = fmaf(r51, r53, r63);
    r50 = r0 * r14;
    r12 = r51 * r2;
    r12 = fmaf(r14, r12, r41 * r50);
    r50 = r49 * r24;
    r50 = r50 * r40;
    r12 = fmaf(r2, r50, r12);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r58,
                                          r62,
                                          r63,
                                          r12);
    r50 = r29 * r35;
    r56 = r29 * r47;
    r56 = fmaf(r8, r56, r7 * r50);
    r50 = r31 * r35;
    r50 = r50 * r40;
    r50 = fmaf(r2, r50, r56 * r52);
    r52 = r29 * r27;
    r52 = fmaf(r10, r52, r56);
    r59 = fmaf(r52, r59, r27 * r11);
    r52 = r6 * r59;
    r11 = r59 * r37;
    r11 = fmaf(r43, r11, r37 * r52);
    r52 = r59 * r37;
    r11 = fmaf(r1, r52, r11);
    r1 = r59 * r37;
    r11 = fmaf(r45, r1, r11);
    r45 = r59 * r37;
    r11 = fmaf(r13, r45, r11);
    r50 = fmaf(r11, r53, r50);
    r53 = r49 * r47;
    r53 = r53 * r40;
    r45 = r11 * r2;
    r45 = fmaf(r14, r45, r2 * r53);
    r53 = r56 * r14;
    r45 = fmaf(r41, r53, r45);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r50,
                                          r45);
    r53 = r6 * r5;
    r53 = fmaf(r58, r4, r62 * r53);
    r41 = r6 * r5;
    r41 = fmaf(r63, r4, r12 * r41);
    r1 = r6 * r5;
    r4 = fmaf(r50, r4, r45 * r1);
    WriteSum3<float, float>((float*)inout_shared, r53, r41, r4);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r58, r58, r62 * r62);
    r41 = fmaf(r63, r63, r12 * r12);
    r53 = fmaf(r45, r45, r50 * r50);
    WriteSum3<float, float>((float*)inout_shared, r4, r41, r53);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fmaf(r58, r63, r62 * r12);
    r62 = fmaf(r62, r45, r58 * r50);
    r50 = fmaf(r63, r50, r12 * r45);
    WriteSum3<float, float>((float*)inout_shared, r53, r62, r50);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void OpencvFixedPoseResJac(float* sensor_from_rig,
                           unsigned int sensor_from_rig_num_alloc,
                           float* calib,
                           unsigned int calib_num_alloc,
                           SharedIndex* calib_indices,
                           float* point,
                           unsigned int point_num_alloc,
                           SharedIndex* point_indices,
                           float* pixel,
                           unsigned int pixel_num_alloc,
                           float* pose,
                           unsigned int pose_num_alloc,
                           float* out_res,
                           unsigned int out_res_num_alloc,
                           float* out_calib_jac,
                           unsigned int out_calib_jac_num_alloc,
                           float* const out_calib_njtr,
                           unsigned int out_calib_njtr_num_alloc,
                           float* const out_calib_precond_diag,
                           unsigned int out_calib_precond_diag_num_alloc,
                           float* const out_calib_precond_tril,
                           unsigned int out_calib_precond_tril_num_alloc,
                           float* out_point_jac,
                           unsigned int out_point_jac_num_alloc,
                           float* const out_point_njtr,
                           unsigned int out_point_njtr_num_alloc,
                           float* const out_point_precond_diag,
                           unsigned int out_point_precond_diag_num_alloc,
                           float* const out_point_precond_tril,
                           unsigned int out_point_precond_tril_num_alloc,
                           size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPoseResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar