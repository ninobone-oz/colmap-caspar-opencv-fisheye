#include "kernel_opencv_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPrincipalPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
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
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99;

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
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    r21 = r15 * r20;
    r22 = r14 * r17;
    r23 = r21 + r22;
    r24 = r13 * r18;
    r23 = fmaf(r16, r19, r23);
    r23 = fmaf(r4, r24, r23);
    r25 = r12 * r23;
    r25 = r25 * r23;
    r26 = r11 + r25;
    r27 = r15 * r17;
    r27 = fmaf(r4, r27, r14 * r20);
    r27 = fmaf(r16, r18, r27);
    r27 = fmaf(r13, r19, r27);
    r28 = r12 * r27;
    r28 = r28 * r27;
    r26 = r26 + r28;
    r5 = fmaf(r8, r26, r5);
    r29 = fmaf(r16, r17, r13 * r20);
    r30 = r14 * r19;
    r29 = fmaf(r4, r30, r29);
    r29 = fmaf(r15, r18, r29);
    r30 = 2.00000000000000000e+00;
    r31 = r30 * r27;
    r32 = r29 * r31;
    r33 = fmaf(r14, r18, r13 * r17);
    r33 = fmaf(r15, r19, r33);
    r33 = fmaf(r4, r33, r16 * r20);
    r34 = r12 * r33;
    r35 = fmaf(r23, r34, r32);
    r36 = r23 * r30;
    r36 = r36 * r29;
    r37 = fmaf(r33, r31, r36);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r38,
                       r39,
                       r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r17 * r19;
    r41 = r41 * r30;
    r42 = r18 * r20;
    r42 = fmaf(r30, r42, r41);
    r43 = r19 * r20;
    r44 = r17 * r18;
    r44 = r44 * r30;
    r43 = fmaf(r12, r43, r44);
    r45 = r19 * r19;
    r45 = r12 * r45;
    r46 = r18 * r18;
    r46 = fmaf(r12, r46, r11);
    r47 = r45 + r46;
    r5 = fmaf(r9, r35, r5);
    r5 = fmaf(r10, r37, r5);
    r5 = fmaf(r40, r42, r5);
    r5 = fmaf(r39, r43, r5);
    r5 = fmaf(r38, r47, r5);
    r48 = r23 * r30;
    r48 = fmaf(r33, r48, r32);
    r6 = fmaf(r8, r48, r6);
    r32 = r19 * r20;
    r32 = fmaf(r30, r32, r44);
    r45 = r11 + r45;
    r44 = r17 * r17;
    r44 = r44 * r12;
    r45 = r45 + r44;
    r49 = r18 * r19;
    r49 = r49 * r30;
    r50 = r17 * r20;
    r50 = fmaf(r12, r50, r49);
    r51 = r23 * r31;
    r52 = fmaf(r29, r34, r51);
    r25 = r11 + r25;
    r53 = r12 * r29;
    r53 = r53 * r29;
    r25 = r25 + r53;
    r6 = fmaf(r38, r32, r6);
    r6 = fmaf(r39, r45, r6);
    r6 = fmaf(r40, r50, r6);
    r6 = fmaf(r10, r52, r6);
    r6 = fmaf(r9, r25, r6);
    r54 = fmaf(r6, r6, r5 * r5);
    r55 = sqrtf(r54);
    r55 = r0 + r55;
    r0 = 1.0 / r55;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r56,
                       r57,
                       r58,
                       r59);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r60 = r56 * r5;
    r61 = r0 * r60;
    r62 = -9.99999999999999955e-07;
    r36 = fmaf(r27, r34, r36);
    r7 = fmaf(r8, r36, r7);
    r63 = r18 * r20;
    r63 = fmaf(r12, r63, r41);
    r46 = r44 + r46;
    r44 = r17 * r20;
    r44 = fmaf(r30, r44, r49);
    r49 = r30 * r29;
    r49 = fmaf(r33, r49, r51);
    r28 = r11 + r28;
    r28 = r28 + r53;
    r7 = fmaf(r38, r63, r7);
    r7 = fmaf(r40, r46, r7);
    r7 = fmaf(r39, r44, r7);
    r7 = fmaf(r9, r49, r7);
    r7 = fmaf(r10, r28, r7);
    r39 = fmaf(r7, r7, r54);
    r40 = rsqrtf(r39);
    r38 = r7 * r40;
    r53 = copysign(1.0, r38);
    r53 = fmaf(r62, r53, r38);
    r62 = acosf(r53);
  };
  LoadShared<2, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r38,
                       r51);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r62 * r62;
    r64 = r41 * r41;
    r65 = r64 * r64;
    r66 = r62 * r65;
    r67 = fmaf(r51, r66, r62);
    r68 = r62 * r41;
    r69 = r68 * r68;
    r70 = r62 * r69;
    r62 = r62 * r64;
    r67 = fmaf(r58, r68, r67);
    r67 = fmaf(r38, r70, r67);
    r67 = fmaf(r59, r62, r67);
    r2 = fmaf(r67, r61, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r67 * r0;
    r71 = r57 * r6;
    r3 = fmaf(r71, r1, r3);
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
    r1 = r56 * r67;
    r72 = r30 * r33;
    r73 = 5.00000000000000000e-01;
    r74 = fmaf(r73, r22, r73 * r21);
    r75 = -5.00000000000000000e-01;
    r76 = r16 * r73;
    r74 = fmaf(r75, r24, r74);
    r74 = fmaf(r19, r76, r74);
    r77 = r13 * r20;
    r78 = r16 * r17;
    r78 = fmaf(r75, r78, r75 * r77);
    r77 = r15 * r18;
    r78 = fmaf(r75, r77, r78);
    r79 = r14 * r19;
    r78 = fmaf(r73, r79, r78);
    r72 = fmaf(r78, r31, r74 * r72);
    r79 = r30 * r29;
    r77 = r14 * r20;
    r80 = r15 * r17;
    r80 = fmaf(r73, r80, r75 * r77);
    r77 = r16 * r18;
    r80 = fmaf(r75, r77, r80);
    r81 = r13 * r19;
    r80 = fmaf(r75, r81, r80);
    r81 = r23 * r30;
    r77 = r13 * r17;
    r82 = r14 * r18;
    r82 = fmaf(r75, r82, r75 * r77);
    r77 = r15 * r19;
    r82 = fmaf(r75, r77, r82);
    r82 = fmaf(r20, r76, r82);
    r81 = r81 * r82;
    r79 = fmaf(r80, r79, r81);
    r72 = r72 + r79;
    r77 = r30 * r29;
    r77 = r77 * r74;
    r83 = r12 * r23;
    r83 = fmaf(r78, r83, r77);
    r84 = r82 * r31;
    r83 = r83 + r84;
    r83 = fmaf(r80, r34, r83);
    r83 = fmaf(r9, r83, r10 * r72);
    r72 = r27 * r74;
    r85 = -4.00000000000000000e+00;
    r72 = r72 * r85;
    r86 = r23 * r80;
    r87 = r85 * r86;
    r88 = r72 + r87;
    r83 = fmaf(r8, r88, r83);
    r1 = r1 * r83;
    r88 = r30 * r5;
    r89 = r30 * r6;
    r90 = r12 * r29;
    r91 = r82 * r34;
    r90 = fmaf(r78, r90, r91);
    r92 = r23 * r30;
    r93 = r80 * r31;
    r92 = fmaf(r74, r92, r93);
    r90 = r90 + r92;
    r94 = r29 * r85;
    r95 = r82 * r94;
    r87 = r87 + r95;
    r87 = fmaf(r9, r87, r10 * r90);
    r90 = r30 * r33;
    r90 = fmaf(r80, r90, r77);
    r77 = r23 * r30;
    r77 = fmaf(r78, r77, r84);
    r90 = r90 + r77;
    r87 = fmaf(r8, r90, r87);
    r89 = fmaf(r87, r89, r83 * r88);
    r88 = r89 * r60;
    r54 = rsqrtf(r54);
    r54 = r75 * r54;
    r55 = r55 * r55;
    r55 = 1.0 / r55;
    r83 = r67 * r55;
    r54 = r54 * r83;
    r88 = fmaf(r54, r88, r0 * r1);
    r1 = r7 * r75;
    r90 = r39 * r39;
    r90 = r39 * r90;
    r90 = rsqrtf(r90);
    r1 = r1 * r90;
    r90 = r30 * r7;
    r39 = r30 * r29;
    r39 = r39 * r78;
    r84 = r30 * r33;
    r84 = r84 * r82;
    r96 = r39 + r84;
    r92 = r92 + r96;
    r97 = r12 * r27;
    r74 = fmaf(r74, r34, r78 * r97);
    r74 = r74 + r79;
    r74 = fmaf(r8, r74, r9 * r92);
    r95 = r72 + r95;
    r74 = fmaf(r10, r95, r74);
    r90 = fmaf(r74, r90, r89);
    r74 = fmaf(r74, r40, r90 * r1);
    r90 = r4 * r74;
    r53 = r53 * r53;
    r53 = fmaf(r4, r53, r11);
    r53 = rsqrtf(r53);
    r11 = -7.00000000000000000e+00;
    r11 = r38 * r11;
    r11 = r11 * r53;
    r11 = r11 * r69;
    r90 = fmaf(r74, r11, r53 * r90);
    r38 = -3.00000000000000000e+00;
    r38 = r58 * r38;
    r38 = r38 * r53;
    r38 = r38 * r41;
    r41 = -9.00000000000000000e+00;
    r41 = r51 * r41;
    r51 = r64 * r64;
    r41 = r41 * r53;
    r41 = r41 * r51;
    r58 = -5.00000000000000000e+00;
    r58 = r59 * r58;
    r58 = r58 * r53;
    r58 = r58 * r64;
    r90 = fmaf(r74, r38, r90);
    r90 = fmaf(r74, r41, r90);
    r90 = fmaf(r74, r58, r90);
    r88 = fmaf(r90, r61, r88);
    r64 = r57 * r67;
    r64 = r64 * r87;
    r87 = r90 * r0;
    r87 = fmaf(r71, r87, r0 * r64);
    r64 = r89 * r71;
    r87 = fmaf(r54, r64, r87);
    r64 = r30 * r6;
    r59 = r12 * r29;
    r59 = fmaf(r80, r59, r81);
    r81 = r13 * r20;
    r95 = r15 * r18;
    r95 = fmaf(r73, r95, r73 * r81);
    r81 = r14 * r19;
    r95 = fmaf(r75, r81, r95);
    r95 = fmaf(r17, r76, r95);
    r81 = r95 * r31;
    r72 = r16 * r19;
    r21 = fmaf(r75, r21, r75 * r72);
    r21 = fmaf(r75, r22, r21);
    r21 = fmaf(r73, r24, r21);
    r59 = r59 + r81;
    r59 = fmaf(r21, r34, r59);
    r24 = r30 * r33;
    r24 = fmaf(r30, r86, r95 * r24);
    r22 = r30 * r29;
    r22 = r22 * r82;
    r72 = fmaf(r21, r31, r22);
    r24 = r24 + r72;
    r24 = fmaf(r8, r24, r10 * r59);
    r59 = r23 * r85;
    r59 = r59 * r95;
    r92 = r21 * r94;
    r97 = r59 + r92;
    r24 = fmaf(r9, r97, r24);
    r97 = r30 * r5;
    r93 = r84 + r93;
    r84 = r23 * r30;
    r84 = r84 * r21;
    r98 = r30 * r29;
    r98 = fmaf(r95, r98, r84);
    r93 = r93 + r98;
    r99 = r27 * r82;
    r99 = r99 * r85;
    r59 = r59 + r99;
    r59 = fmaf(r8, r59, r10 * r93);
    r95 = fmaf(r95, r34, r12 * r86);
    r95 = r95 + r72;
    r59 = fmaf(r9, r95, r59);
    r97 = fmaf(r59, r97, r24 * r64);
    r64 = r97 * r60;
    r95 = r12 * r27;
    r95 = fmaf(r80, r95, r91);
    r95 = r95 + r98;
    r98 = r30 * r33;
    r98 = fmaf(r21, r98, r81);
    r98 = r98 + r79;
    r98 = fmaf(r9, r98, r8 * r95);
    r92 = r99 + r92;
    r98 = fmaf(r10, r92, r98);
    r92 = r30 * r7;
    r92 = fmaf(r98, r92, r97);
    r92 = fmaf(r92, r1, r98 * r40);
    r98 = fmaf(r92, r11, r92 * r41);
    r99 = r4 * r92;
    r98 = fmaf(r53, r99, r98);
    r98 = fmaf(r92, r58, r98);
    r98 = fmaf(r92, r38, r98);
    r64 = fmaf(r98, r61, r54 * r64);
    r99 = r56 * r67;
    r99 = r99 * r59;
    r64 = fmaf(r0, r99, r64);
    r99 = r97 * r71;
    r59 = r98 * r0;
    r59 = fmaf(r71, r59, r54 * r99);
    r99 = r57 * r67;
    r99 = r99 * r24;
    r59 = fmaf(r0, r99, r59);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r88,
                                          r87,
                                          r64,
                                          r59);
    r99 = r27 * r78;
    r99 = r99 * r85;
    r24 = r14 * r20;
    r95 = r15 * r17;
    r95 = fmaf(r75, r95, r73 * r24);
    r24 = r13 * r19;
    r95 = fmaf(r73, r24, r95);
    r95 = fmaf(r18, r76, r95);
    r94 = r95 * r94;
    r76 = r99 + r94;
    r24 = r23 * r30;
    r24 = r24 * r95;
    r22 = r22 + r24;
    r73 = r12 * r27;
    r22 = fmaf(r21, r73, r22);
    r22 = fmaf(r78, r34, r22);
    r22 = fmaf(r8, r22, r10 * r76);
    r76 = r30 * r29;
    r73 = r30 * r33;
    r73 = fmaf(r95, r73, r21 * r76);
    r73 = r73 + r77;
    r22 = fmaf(r9, r73, r22);
    r73 = r30 * r7;
    r76 = r30 * r6;
    r31 = r95 * r31;
    r84 = r84 + r31;
    r84 = r84 + r96;
    r96 = r12 * r29;
    r34 = fmaf(r95, r34, r21 * r96);
    r34 = r34 + r77;
    r34 = fmaf(r10, r34, r8 * r84);
    r82 = r23 * r82;
    r82 = r82 * r85;
    r94 = r82 + r94;
    r34 = fmaf(r9, r94, r34);
    r94 = r30 * r5;
    r85 = r12 * r23;
    r85 = fmaf(r21, r85, r39);
    r85 = r85 + r91;
    r85 = r85 + r31;
    r82 = r99 + r82;
    r82 = fmaf(r8, r82, r9 * r85);
    r8 = r30 * r33;
    r8 = fmaf(r78, r8, r24);
    r8 = r8 + r72;
    r82 = fmaf(r10, r8, r82);
    r94 = fmaf(r82, r94, r34 * r76);
    r73 = fmaf(r22, r73, r94);
    r73 = fmaf(r73, r1, r22 * r40);
    r22 = r4 * r73;
    r22 = fmaf(r73, r38, r53 * r22);
    r22 = fmaf(r73, r11, r22);
    r22 = fmaf(r73, r58, r22);
    r22 = fmaf(r73, r41, r22);
    r76 = r94 * r60;
    r76 = fmaf(r54, r76, r22 * r61);
    r8 = r56 * r67;
    r8 = r8 * r82;
    r76 = fmaf(r0, r8, r76);
    r8 = r94 * r71;
    r82 = r22 * r0;
    r82 = fmaf(r71, r82, r54 * r8);
    r8 = r57 * r67;
    r8 = r8 * r34;
    r82 = fmaf(r0, r8, r82);
    r8 = r30 * r47;
    r34 = r30 * r32;
    r34 = fmaf(r6, r34, r5 * r8);
    r8 = r34 * r60;
    r10 = r56 * r47;
    r10 = r10 * r67;
    r10 = fmaf(r0, r10, r54 * r8);
    r8 = r30 * r63;
    r8 = fmaf(r7, r8, r34);
    r8 = fmaf(r8, r1, r63 * r40);
    r72 = r4 * r8;
    r72 = fmaf(r8, r11, r53 * r72);
    r72 = fmaf(r8, r41, r72);
    r72 = fmaf(r8, r58, r72);
    r72 = fmaf(r8, r38, r72);
    r10 = fmaf(r72, r61, r10);
    r24 = r34 * r71;
    r78 = r57 * r32;
    r78 = r78 * r67;
    r78 = fmaf(r0, r78, r54 * r24);
    r24 = r72 * r0;
    r78 = fmaf(r71, r24, r78);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r76,
                                          r82,
                                          r10,
                                          r78);
    r24 = r30 * r43;
    r85 = r30 * r45;
    r85 = fmaf(r6, r85, r5 * r24);
    r24 = r85 * r60;
    r9 = r30 * r44;
    r9 = fmaf(r7, r9, r85);
    r9 = fmaf(r44, r40, r9 * r1);
    r99 = r4 * r9;
    r99 = fmaf(r9, r41, r53 * r99);
    r99 = fmaf(r9, r11, r99);
    r99 = fmaf(r9, r38, r99);
    r99 = fmaf(r9, r58, r99);
    r24 = fmaf(r99, r61, r54 * r24);
    r31 = r56 * r43;
    r31 = r31 * r67;
    r24 = fmaf(r0, r31, r24);
    r31 = r99 * r0;
    r91 = r85 * r71;
    r91 = fmaf(r54, r91, r71 * r31);
    r31 = r57 * r45;
    r31 = r31 * r67;
    r91 = fmaf(r0, r31, r91);
    r31 = r56 * r42;
    r31 = r31 * r67;
    r39 = r30 * r42;
    r21 = r30 * r50;
    r21 = fmaf(r6, r21, r5 * r39);
    r39 = r21 * r60;
    r39 = fmaf(r54, r39, r0 * r31);
    r31 = r30 * r46;
    r31 = fmaf(r7, r31, r21);
    r31 = fmaf(r46, r40, r31 * r1);
    r84 = fmaf(r31, r11, r31 * r58);
    r77 = r4 * r31;
    r84 = fmaf(r53, r77, r84);
    r84 = fmaf(r31, r38, r84);
    r84 = fmaf(r31, r41, r84);
    r39 = fmaf(r84, r61, r39);
    r77 = r84 * r0;
    r95 = r21 * r71;
    r95 = fmaf(r54, r95, r71 * r77);
    r77 = r57 * r50;
    r77 = r77 * r67;
    r95 = fmaf(r0, r77, r95);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r24,
                                          r91,
                                          r39,
                                          r95);
    r77 = r4 * r2;
    r3 = r4 * r3;
    r77 = fmaf(r87, r3, r88 * r77);
    r96 = r4 * r2;
    r96 = fmaf(r59, r3, r64 * r96);
    r75 = r4 * r2;
    r75 = fmaf(r82, r3, r76 * r75);
    r79 = r4 * r2;
    r79 = fmaf(r78, r3, r10 * r79);
    WriteSum4<float, float>((float*)inout_shared, r77, r96, r75, r79);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r4 * r2;
    r79 = fmaf(r91, r3, r24 * r79);
    r75 = r4 * r2;
    r75 = fmaf(r95, r3, r39 * r75);
    WriteSum2<float, float>((float*)inout_shared, r79, r75);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r87, r87, r88 * r88);
    r79 = fmaf(r64, r64, r59 * r59);
    r96 = fmaf(r82, r82, r76 * r76);
    r77 = fmaf(r78, r78, r10 * r10);
    WriteSum4<float, float>((float*)inout_shared, r75, r79, r96, r77);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r91, r91, r24 * r24);
    r96 = fmaf(r95, r95, r39 * r39);
    WriteSum2<float, float>((float*)inout_shared, r77, r96);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r96 = fmaf(r88, r64, r87 * r59);
    r77 = fmaf(r87, r82, r88 * r76);
    r79 = fmaf(r87, r78, r88 * r10);
    r75 = fmaf(r87, r91, r88 * r24);
    WriteSum4<float, float>((float*)inout_shared, r96, r77, r79, r75);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = fmaf(r87, r95, r88 * r39);
    r88 = fmaf(r59, r82, r64 * r76);
    r75 = fmaf(r64, r10, r59 * r78);
    r79 = fmaf(r59, r91, r64 * r24);
    WriteSum4<float, float>((float*)inout_shared, r87, r88, r75, r79);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r64, r39, r59 * r95);
    r59 = fmaf(r82, r78, r76 * r10);
    r79 = fmaf(r76, r24, r82 * r91);
    r82 = fmaf(r82, r95, r76 * r39);
    WriteSum4<float, float>((float*)inout_shared, r64, r59, r79, r82);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fmaf(r10, r24, r78 * r91);
    r10 = fmaf(r10, r39, r78 * r95);
    r39 = fmaf(r24, r39, r91 * r95);
    WriteSum3<float, float>((float*)inout_shared, r82, r10, r39);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r5 * r67;
    r39 = r39 * r0;
    r10 = r6 * r67;
    r10 = r10 * r0;
    r82 = r68 * r61;
    r24 = r0 * r68;
    r24 = r24 * r71;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r39,
                                          r10,
                                          r82,
                                          r24);
    r24 = r62 * r61;
    r82 = r0 * r62;
    r82 = r82 * r71;
    r10 = r70 * r61;
    r39 = r0 * r70;
    r39 = r39 * r71;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r24,
                                          r82,
                                          r10,
                                          r39);
    r39 = r66 * r61;
    r10 = r0 * r71;
    r10 = r10 * r66;
    WriteIdx2<1024, float, float, float2>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r39,
                                          r10);
    r10 = r4 * r5;
    r10 = r10 * r67;
    r10 = r10 * r2;
    r10 = r10 * r0;
    r39 = r6 * r67;
    r82 = r0 * r3;
    r39 = r39 * r82;
    r82 = r71 * r82;
    r24 = r4 * r2;
    r24 = r24 * r61;
    r95 = fmaf(r68, r24, r68 * r82);
    r91 = fmaf(r62, r24, r62 * r82);
    WriteSum4<float, float>((float*)inout_shared, r10, r39, r95, r91);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = fmaf(r70, r24, r70 * r82);
    r24 = fmaf(r66, r24, r66 * r82);
    WriteSum2<float, float>((float*)inout_shared, r91, r24);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r5 * r5;
    r24 = r24 * r67;
    r24 = r24 * r83;
    r91 = r6 * r6;
    r91 = r91 * r67;
    r91 = r91 * r83;
    r82 = r55 * r60;
    r82 = r82 * r60;
    r55 = r57 * r55;
    r95 = r6 * r71;
    r55 = r55 * r95;
    r39 = fmaf(r69, r55, r69 * r82);
    r10 = r62 * r62;
    r10 = fmaf(r10, r55, r10 * r82);
    WriteSum4<float, float>((float*)inout_shared, r24, r91, r39, r10);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r70 * r70;
    r39 = fmaf(r39, r55, r39 * r82);
    r91 = r66 * r66;
    r91 = fmaf(r55, r91, r82 * r91);
    WriteSum2<float, float>((float*)inout_shared, r39, r91);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = 0.00000000000000000e+00;
    r24 = r5 * r68;
    r24 = r24 * r60;
    r24 = r24 * r83;
    r78 = r5 * r62;
    r78 = r78 * r60;
    r78 = r78 * r83;
    r79 = r5 * r70;
    r79 = r79 * r60;
    r79 = r79 * r83;
    WriteSum4<float, float>((float*)inout_shared, r91, r24, r78, r79);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r5 * r60;
    r79 = r79 * r83;
    r79 = r79 * r66;
    r68 = r68 * r83;
    r68 = r68 * r95;
    r62 = r62 * r83;
    r62 = r62 * r95;
    r70 = r70 * r83;
    r70 = r70 * r95;
    WriteSum4<float, float>((float*)inout_shared, r79, r68, r62, r70);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r83 * r66;
    r66 = r66 * r95;
    r51 = fmaf(r51, r55, r51 * r82);
    r69 = r69 * r69;
    r69 = fmaf(r69, r55, r69 * r82);
    WriteSum4<float, float>((float*)inout_shared, r66, r51, r10, r69);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r65 * r65;
    r65 = fmaf(r55, r65, r82 * r65);
    WriteSum3<float, float>((float*)inout_shared, r69, r39, r65);
  };
  FlushSumShared<3, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r30 * r36;
    r39 = r30 * r26;
    r69 = r30 * r48;
    r69 = fmaf(r6, r69, r5 * r39);
    r65 = fmaf(r7, r65, r69);
    r65 = fmaf(r65, r1, r36 * r40);
    r39 = r4 * r65;
    r39 = fmaf(r65, r38, r53 * r39);
    r39 = fmaf(r65, r11, r39);
    r39 = fmaf(r65, r41, r39);
    r39 = fmaf(r65, r58, r39);
    r69 = r69 * r54;
    r55 = fmaf(r60, r69, r39 * r61);
    r82 = r56 * r26;
    r82 = r82 * r67;
    r55 = fmaf(r0, r82, r55);
    r82 = r39 * r0;
    r82 = fmaf(r71, r82, r71 * r69);
    r69 = r57 * r48;
    r69 = r69 * r67;
    r82 = fmaf(r0, r69, r82);
    r69 = r30 * r25;
    r10 = r30 * r35;
    r10 = fmaf(r5, r10, r6 * r69);
    r69 = r10 * r60;
    r51 = r56 * r35;
    r51 = r51 * r67;
    r51 = fmaf(r0, r51, r54 * r69);
    r69 = r30 * r49;
    r69 = fmaf(r7, r69, r10);
    r69 = fmaf(r49, r40, r69 * r1);
    r66 = r4 * r69;
    r66 = fmaf(r69, r41, r53 * r66);
    r66 = fmaf(r69, r11, r66);
    r66 = fmaf(r69, r38, r66);
    r66 = fmaf(r69, r58, r66);
    r51 = fmaf(r66, r61, r51);
    r95 = r10 * r71;
    r83 = r57 * r25;
    r83 = r83 * r67;
    r83 = fmaf(r0, r83, r54 * r95);
    r95 = r66 * r0;
    r83 = fmaf(r71, r95, r83);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r55,
                                          r82,
                                          r51,
                                          r83);
    r95 = r30 * r28;
    r70 = r30 * r37;
    r62 = r30 * r52;
    r62 = fmaf(r6, r62, r5 * r70);
    r95 = fmaf(r7, r95, r62);
    r1 = fmaf(r95, r1, r28 * r40);
    r95 = r4 * r1;
    r11 = fmaf(r1, r11, r53 * r95);
    r11 = fmaf(r1, r58, r11);
    r11 = fmaf(r1, r41, r11);
    r11 = fmaf(r1, r38, r11);
    r38 = r56 * r37;
    r38 = r38 * r67;
    r38 = fmaf(r0, r38, r11 * r61);
    r61 = r62 * r60;
    r38 = fmaf(r54, r61, r38);
    r61 = r62 * r71;
    r41 = r11 * r0;
    r41 = fmaf(r71, r41, r54 * r61);
    r61 = r57 * r52;
    r61 = r61 * r67;
    r41 = fmaf(r0, r61, r41);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r38,
                                          r41);
    r61 = r4 * r2;
    r61 = fmaf(r82, r3, r55 * r61);
    r54 = r4 * r2;
    r54 = fmaf(r83, r3, r51 * r54);
    r58 = r4 * r2;
    r3 = fmaf(r41, r3, r38 * r58);
    WriteSum3<float, float>((float*)inout_shared, r61, r54, r3);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r55, r55, r82 * r82);
    r54 = fmaf(r51, r51, r83 * r83);
    r61 = fmaf(r41, r41, r38 * r38);
    WriteSum3<float, float>((float*)inout_shared, r3, r54, r61);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r82, r83, r55 * r51);
    r55 = fmaf(r55, r38, r82 * r41);
    r41 = fmaf(r83, r41, r51 * r38);
    WriteSum3<float, float>((float*)inout_shared, r61, r55, r41);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPrincipalPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
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
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
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
  OpencvSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
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