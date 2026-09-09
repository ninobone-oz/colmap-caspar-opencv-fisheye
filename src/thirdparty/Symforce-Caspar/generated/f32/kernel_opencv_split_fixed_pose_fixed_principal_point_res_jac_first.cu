#include "kernel_opencv_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedPrincipalPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = 1.00000000000000000e+00;
    r12 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r17, r18, r19, r20);
    r21 = fmaf(r13, r18, r16 * r19);
    r22 = r14 * r17;
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r15, r20, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = r11 + r22;
    r24 = r13 * r19;
    r24 = fmaf(r4, r24, r16 * r18);
    r24 = fmaf(r14, r20, r24);
    r24 = fmaf(r15, r17, r24);
    r25 = r12 * r24;
    r25 = r25 * r24;
    r23 = r23 + r25;
    r5 = fmaf(r8, r23, r5);
    r26 = fmaf(r13, r20, r16 * r17);
    r27 = r15 * r18;
    r26 = fmaf(r4, r27, r26);
    r26 = fmaf(r14, r19, r26);
    r27 = 2.00000000000000000e+00;
    r28 = r27 * r24;
    r29 = r26 * r28;
    r30 = fmaf(r14, r18, r13 * r17);
    r30 = fmaf(r15, r19, r30);
    r30 = fmaf(r4, r30, r16 * r20);
    r20 = r12 * r30;
    r31 = fmaf(r21, r20, r29);
    r32 = r21 * r27;
    r32 = r32 * r26;
    r33 = fmaf(r30, r28, r32);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r34, r35, r36);
    r37 = r13 * r15;
    r37 = r37 * r27;
    r38 = r14 * r16;
    r39 = fmaf(r27, r38, r37);
    r40 = r15 * r16;
    r41 = r13 * r14;
    r41 = r41 * r27;
    r40 = fmaf(r12, r40, r41);
    r42 = r15 * r15;
    r42 = r12 * r42;
    r43 = r14 * r14;
    r43 = fmaf(r12, r43, r11);
    r44 = r42 + r43;
    r5 = fmaf(r9, r31, r5);
    r5 = fmaf(r10, r33, r5);
    r5 = fmaf(r36, r39, r5);
    r5 = fmaf(r35, r40, r5);
    r5 = fmaf(r34, r44, r5);
    r44 = r21 * r27;
    r44 = fmaf(r30, r44, r29);
    r6 = fmaf(r8, r44, r6);
    r29 = r15 * r16;
    r29 = fmaf(r27, r29, r41);
    r42 = r11 + r42;
    r41 = r13 * r13;
    r41 = r12 * r41;
    r42 = r42 + r41;
    r40 = r14 * r15;
    r40 = r40 * r27;
    r39 = r13 * r16;
    r39 = fmaf(r12, r39, r40);
    r28 = r21 * r28;
    r45 = fmaf(r26, r20, r28);
    r22 = r11 + r22;
    r46 = r26 * r26;
    r46 = r12 * r46;
    r22 = r22 + r46;
    r6 = fmaf(r34, r29, r6);
    r6 = fmaf(r35, r42, r6);
    r6 = fmaf(r36, r39, r6);
    r6 = fmaf(r10, r45, r6);
    r6 = fmaf(r9, r22, r6);
    r39 = fmaf(r6, r6, r5 * r5);
    r42 = sqrtf(r39);
    r42 = r0 + r42;
    r0 = 1.0 / r42;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r29,
                       r47,
                       r48,
                       r49);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r50 = r29 * r5;
    r51 = r0 * r50;
    r52 = -9.99999999999999955e-07;
    r20 = fmaf(r24, r20, r32);
    r8 = fmaf(r8, r20, r7);
    r38 = fmaf(r12, r38, r37);
    r43 = r41 + r43;
    r41 = r13 * r16;
    r41 = fmaf(r27, r41, r40);
    r40 = r27 * r26;
    r40 = fmaf(r30, r40, r28);
    r25 = r11 + r25;
    r25 = r25 + r46;
    r8 = fmaf(r34, r38, r8);
    r8 = fmaf(r36, r43, r8);
    r8 = fmaf(r35, r41, r8);
    r8 = fmaf(r9, r40, r8);
    r8 = fmaf(r10, r25, r8);
    r10 = fmaf(r8, r8, r39);
    r9 = rsqrtf(r10);
    r41 = r8 * r9;
    r35 = copysign(1.0, r41);
    r35 = fmaf(r52, r35, r41);
    r52 = acosf(r35);
  };
  LoadShared<2, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r41,
                       r43);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r52 * r52;
    r38 = r36 * r36;
    r34 = r38 * r38;
    r46 = r52 * r34;
    r28 = fmaf(r43, r46, r52);
    r30 = r52 * r36;
    r12 = r30 * r30;
    r37 = r52 * r12;
    r52 = r52 * r38;
    r28 = fmaf(r48, r30, r28);
    r28 = fmaf(r41, r37, r28);
    r28 = fmaf(r49, r52, r28);
    r2 = fmaf(r28, r51, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r28 * r0;
    r7 = r47 * r6;
    r3 = fmaf(r7, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fmaf(r2, r2, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    r1 = r5 * r28;
    r1 = r1 * r0;
    r24 = r6 * r28;
    r24 = r24 * r0;
    r32 = r30 * r51;
    r53 = r0 * r30;
    r53 = r53 * r7;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r24,
                                          r32,
                                          r53);
    r53 = r52 * r51;
    r32 = r0 * r52;
    r32 = r32 * r7;
    r24 = r37 * r51;
    r1 = r0 * r37;
    r1 = r1 * r7;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r53,
                                          r32,
                                          r24,
                                          r1);
    r1 = r46 * r51;
    r24 = r0 * r7;
    r24 = r24 * r46;
    WriteIdx2<1024, float, float, float2>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r24);
    r24 = r5 * r28;
    r2 = r4 * r2;
    r24 = r24 * r0;
    r24 = r24 * r2;
    r1 = r4 * r6;
    r1 = r1 * r28;
    r1 = r1 * r3;
    r1 = r1 * r0;
    r32 = r4 * r3;
    r32 = r32 * r0;
    r32 = r32 * r30;
    r53 = r51 * r2;
    r32 = fmaf(r30, r53, r7 * r32);
    r54 = r4 * r3;
    r54 = r54 * r0;
    r54 = r54 * r52;
    r54 = fmaf(r52, r53, r7 * r54);
    WriteSum4<float, float>((float*)inout_shared, r24, r1, r32, r54);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r4 * r3;
    r54 = r54 * r0;
    r54 = r54 * r37;
    r54 = fmaf(r37, r53, r7 * r54);
    r32 = r4 * r3;
    r32 = r32 * r0;
    r32 = r32 * r7;
    r53 = fmaf(r46, r53, r46 * r32);
    WriteSum2<float, float>((float*)inout_shared, r54, r53);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r5 * r5;
    r42 = r42 * r42;
    r42 = 1.0 / r42;
    r54 = r28 * r42;
    r53 = r53 * r28;
    r53 = r53 * r54;
    r32 = r6 * r6;
    r32 = r32 * r28;
    r32 = r32 * r54;
    r1 = r42 * r50;
    r1 = r1 * r50;
    r42 = r47 * r42;
    r24 = r6 * r7;
    r42 = r42 * r24;
    r55 = fmaf(r12, r42, r12 * r1);
    r56 = r52 * r52;
    r56 = fmaf(r56, r42, r56 * r1);
    WriteSum4<float, float>((float*)inout_shared, r53, r32, r55, r56);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r37 * r37;
    r55 = fmaf(r55, r42, r55 * r1);
    r32 = r46 * r46;
    r32 = fmaf(r42, r32, r1 * r32);
    WriteSum2<float, float>((float*)inout_shared, r55, r32);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = 0.00000000000000000e+00;
    r53 = r5 * r30;
    r53 = r53 * r50;
    r53 = r53 * r54;
    r57 = r5 * r52;
    r57 = r57 * r50;
    r57 = r57 * r54;
    r58 = r5 * r37;
    r58 = r58 * r50;
    r58 = r58 * r54;
    WriteSum4<float, float>((float*)inout_shared, r32, r53, r57, r58);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r5 * r50;
    r58 = r58 * r46;
    r58 = r58 * r54;
    r30 = r30 * r54;
    r30 = r30 * r24;
    r52 = r52 * r54;
    r52 = r52 * r24;
    r37 = r37 * r54;
    r37 = r37 * r24;
    WriteSum4<float, float>((float*)inout_shared, r58, r30, r52, r37);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r46 * r54;
    r46 = r46 * r24;
    r24 = fmaf(r34, r42, r34 * r1);
    r37 = r12 * r12;
    r37 = fmaf(r37, r42, r37 * r1);
    WriteSum4<float, float>((float*)inout_shared, r46, r24, r56, r37);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r34 * r34;
    r56 = fmaf(r42, r56, r1 * r56);
    WriteSum3<float, float>((float*)inout_shared, r37, r55, r56);
  };
  FlushSumShared<3, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = -5.00000000000000000e-01;
    r55 = r8 * r56;
    r37 = r10 * r10;
    r37 = r10 * r37;
    r37 = rsqrtf(r37);
    r55 = r55 * r37;
    r37 = r27 * r20;
    r10 = r27 * r23;
    r42 = r27 * r44;
    r42 = fmaf(r6, r42, r5 * r10);
    r37 = fmaf(r8, r37, r42);
    r37 = fmaf(r37, r55, r20 * r9);
    r35 = r35 * r35;
    r35 = fmaf(r4, r35, r11);
    r35 = rsqrtf(r35);
    r37 = r37 * r35;
    r11 = -3.00000000000000000e+00;
    r11 = r48 * r11;
    r11 = r11 * r36;
    r36 = fmaf(r37, r11, r4 * r37);
    r48 = -7.00000000000000000e+00;
    r48 = r41 * r48;
    r48 = r48 * r12;
    r12 = -9.00000000000000000e+00;
    r12 = r43 * r12;
    r12 = r12 * r34;
    r34 = -5.00000000000000000e+00;
    r34 = r49 * r34;
    r34 = r34 * r38;
    r36 = fmaf(r37, r48, r36);
    r36 = fmaf(r37, r12, r36);
    r36 = fmaf(r37, r34, r36);
    r39 = rsqrtf(r39);
    r39 = r56 * r39;
    r39 = r39 * r54;
    r50 = r50 * r39;
    r54 = fmaf(r42, r50, r36 * r51);
    r56 = r29 * r23;
    r56 = r56 * r28;
    r54 = fmaf(r0, r56, r54);
    r56 = r42 * r7;
    r37 = r36 * r0;
    r37 = fmaf(r7, r37, r39 * r56);
    r56 = r47 * r44;
    r56 = r56 * r28;
    r37 = fmaf(r0, r56, r37);
    r56 = r27 * r22;
    r38 = r27 * r31;
    r38 = fmaf(r5, r38, r6 * r56);
    r56 = r29 * r31;
    r56 = r56 * r28;
    r56 = fmaf(r0, r56, r38 * r50);
    r49 = r27 * r40;
    r49 = fmaf(r8, r49, r38);
    r49 = fmaf(r40, r9, r49 * r55);
    r43 = r4 * r49;
    r41 = r49 * r35;
    r43 = fmaf(r12, r41, r35 * r43);
    r43 = fmaf(r48, r41, r43);
    r43 = fmaf(r11, r41, r43);
    r43 = fmaf(r34, r41, r43);
    r56 = fmaf(r43, r51, r56);
    r41 = r38 * r7;
    r10 = r47 * r22;
    r10 = r10 * r28;
    r10 = fmaf(r0, r10, r39 * r41);
    r41 = r43 * r0;
    r10 = fmaf(r7, r41, r10);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r54,
                                          r37,
                                          r56,
                                          r10);
    r41 = r27 * r25;
    r1 = r27 * r33;
    r24 = r27 * r45;
    r24 = fmaf(r6, r24, r5 * r1);
    r41 = fmaf(r8, r41, r24);
    r55 = fmaf(r41, r55, r25 * r9);
    r41 = r4 * r55;
    r9 = r55 * r35;
    r9 = fmaf(r48, r9, r35 * r41);
    r41 = r55 * r35;
    r9 = fmaf(r34, r41, r9);
    r34 = r55 * r35;
    r9 = fmaf(r12, r34, r9);
    r12 = r55 * r35;
    r9 = fmaf(r11, r12, r9);
    r12 = r29 * r33;
    r12 = r12 * r28;
    r12 = fmaf(r0, r12, r9 * r51);
    r12 = fmaf(r24, r50, r12);
    r50 = r24 * r7;
    r51 = r9 * r0;
    r51 = fmaf(r7, r51, r39 * r50);
    r50 = r47 * r45;
    r50 = r50 * r28;
    r51 = fmaf(r0, r50, r51);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r12,
                                          r51);
    r50 = r4 * r3;
    r50 = fmaf(r54, r2, r37 * r50);
    r39 = r4 * r3;
    r39 = fmaf(r56, r2, r10 * r39);
    r34 = r4 * r3;
    r2 = fmaf(r12, r2, r51 * r34);
    WriteSum3<float, float>((float*)inout_shared, r50, r39, r2);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r54, r54, r37 * r37);
    r39 = fmaf(r56, r56, r10 * r10);
    r50 = fmaf(r51, r51, r12 * r12);
    WriteSum3<float, float>((float*)inout_shared, r2, r39, r50);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fmaf(r37, r10, r54 * r56);
    r54 = fmaf(r54, r12, r37 * r51);
    r51 = fmaf(r10, r51, r56 * r12);
    WriteSum3<float, float>((float*)inout_shared, r50, r54, r51);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedPrincipalPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  OpencvSplitFixedPoseFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
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