#include "kernel_opencv_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvFixedPointResJacKernel(float* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 float* sensor_from_rig,
                                 unsigned int sensor_from_rig_num_alloc,
                                 float* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 float* pixel,
                                 unsigned int pixel_num_alloc,
                                 float* point,
                                 unsigned int point_num_alloc,
                                 float* out_res,
                                 unsigned int out_res_num_alloc,
                                 float* out_pose_jac,
                                 unsigned int out_pose_jac_num_alloc,
                                 float* const out_pose_njtr,
                                 unsigned int out_pose_njtr_num_alloc,
                                 float* const out_pose_precond_diag,
                                 unsigned int out_pose_precond_diag_num_alloc,
                                 float* const out_pose_precond_tril,
                                 unsigned int out_pose_precond_tril_num_alloc,
                                 float* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 float* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 float* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 float* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
                                 size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88;
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
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11, r12);
    r13 = 1.00000000000000000e+00;
    r14 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r15,
                       r16,
                       r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r19,
                                         r20,
                                         r21,
                                         r22);
    r23 = r17 * r22;
    r24 = r16 * r19;
    r25 = r23 + r24;
    r26 = r15 * r20;
    r25 = fmaf(r18, r21, r25);
    r25 = fmaf(r6, r26, r25);
    r27 = r14 * r25;
    r27 = r27 * r25;
    r28 = r13 + r27;
    r29 = r17 * r19;
    r29 = fmaf(r6, r29, r16 * r22);
    r29 = fmaf(r18, r20, r29);
    r29 = fmaf(r15, r21, r29);
    r30 = r14 * r29;
    r30 = r30 * r29;
    r28 = r28 + r30;
    r28 = fmaf(r10, r28, r7);
    r7 = fmaf(r18, r19, r15 * r22);
    r31 = r16 * r21;
    r7 = fmaf(r6, r31, r7);
    r7 = fmaf(r17, r20, r7);
    r31 = 2.00000000000000000e+00;
    r32 = r31 * r29;
    r33 = r7 * r32;
    r34 = fmaf(r16, r20, r15 * r19);
    r34 = fmaf(r17, r21, r34);
    r34 = fmaf(r6, r34, r18 * r22);
    r35 = r14 * r34;
    r36 = fmaf(r25, r35, r33);
    r37 = r25 * r31;
    r37 = r37 * r7;
    r38 = fmaf(r34, r32, r37);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r39,
                       r40,
                       r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = r19 * r21;
    r42 = r42 * r31;
    r43 = r20 * r22;
    r43 = fmaf(r31, r43, r42);
    r44 = r21 * r22;
    r45 = r19 * r20;
    r45 = r45 * r31;
    r44 = fmaf(r14, r44, r45);
    r46 = r21 * r21;
    r46 = r14 * r46;
    r47 = r20 * r20;
    r47 = fmaf(r14, r47, r13);
    r48 = r46 + r47;
    r28 = fmaf(r11, r36, r28);
    r28 = fmaf(r12, r38, r28);
    r28 = fmaf(r41, r43, r28);
    r28 = fmaf(r40, r44, r28);
    r28 = fmaf(r39, r48, r28);
    r38 = r25 * r31;
    r38 = fmaf(r34, r38, r33);
    r38 = fmaf(r10, r38, r8);
    r8 = r21 * r22;
    r8 = fmaf(r31, r8, r45);
    r46 = r13 + r46;
    r45 = r19 * r19;
    r45 = r45 * r14;
    r46 = r46 + r45;
    r33 = r20 * r21;
    r33 = r33 * r31;
    r36 = r19 * r22;
    r36 = fmaf(r14, r36, r33);
    r49 = r25 * r32;
    r50 = fmaf(r7, r35, r49);
    r27 = r13 + r27;
    r51 = r14 * r7;
    r51 = r51 * r7;
    r27 = r27 + r51;
    r38 = fmaf(r39, r8, r38);
    r38 = fmaf(r40, r46, r38);
    r38 = fmaf(r41, r36, r38);
    r38 = fmaf(r12, r50, r38);
    r38 = fmaf(r11, r27, r38);
    r27 = fmaf(r38, r38, r28 * r28);
    r50 = sqrtf(r27);
    r50 = r2 + r50;
    r2 = 1.0 / r50;
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r52,
                       r53,
                       r54,
                       r55);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r56 = r52 * r28;
    r57 = r2 * r56;
    r58 = -9.99999999999999955e-07;
    r37 = fmaf(r29, r35, r37);
    r37 = fmaf(r10, r37, r9);
    r9 = r20 * r22;
    r9 = fmaf(r14, r9, r42);
    r47 = r45 + r47;
    r45 = r19 * r22;
    r45 = fmaf(r31, r45, r33);
    r33 = r31 * r7;
    r33 = fmaf(r34, r33, r49);
    r30 = r13 + r30;
    r30 = r30 + r51;
    r37 = fmaf(r39, r9, r37);
    r37 = fmaf(r41, r47, r37);
    r37 = fmaf(r40, r45, r37);
    r37 = fmaf(r11, r33, r37);
    r37 = fmaf(r12, r30, r37);
    r30 = fmaf(r37, r37, r27);
    r33 = rsqrtf(r30);
    r40 = r37 * r33;
    r41 = copysign(1.0, r40);
    r41 = fmaf(r58, r41, r40);
    r58 = acosf(r41);
    r40 = r58 * r58;
    r39 = r40 * r40;
    r51 = r58 * r39;
    r49 = fmaf(r55, r51, r58);
    r42 = r58 * r40;
    r59 = r42 * r42;
    r60 = r58 * r59;
    r61 = r39 * r39;
    r58 = r58 * r61;
    r49 = fmaf(r54, r42, r49);
    r49 = fmaf(r0, r60, r49);
    r49 = fmaf(r1, r58, r49);
    r4 = fmaf(r49, r57, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r49 * r2;
    r62 = r53 * r38;
    r5 = fmaf(r62, r3, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r3 = r52 * r49;
    r63 = r31 * r34;
    r64 = 5.00000000000000000e-01;
    r65 = fmaf(r64, r24, r64 * r23);
    r66 = -5.00000000000000000e-01;
    r67 = r18 * r64;
    r65 = fmaf(r66, r26, r65);
    r65 = fmaf(r21, r67, r65);
    r68 = r15 * r22;
    r69 = r18 * r19;
    r69 = fmaf(r66, r69, r66 * r68);
    r68 = r17 * r20;
    r69 = fmaf(r66, r68, r69);
    r70 = r16 * r21;
    r69 = fmaf(r64, r70, r69);
    r63 = fmaf(r69, r32, r65 * r63);
    r70 = r31 * r7;
    r68 = r16 * r22;
    r71 = r17 * r19;
    r71 = fmaf(r64, r71, r66 * r68);
    r68 = r18 * r20;
    r71 = fmaf(r66, r68, r71);
    r72 = r15 * r21;
    r71 = fmaf(r66, r72, r71);
    r72 = r25 * r31;
    r68 = r15 * r19;
    r73 = r16 * r20;
    r73 = fmaf(r66, r73, r66 * r68);
    r68 = r17 * r21;
    r73 = fmaf(r66, r68, r73);
    r73 = fmaf(r22, r67, r73);
    r72 = r72 * r73;
    r70 = fmaf(r71, r70, r72);
    r63 = r63 + r70;
    r68 = r31 * r7;
    r68 = r68 * r65;
    r74 = r14 * r25;
    r74 = fmaf(r69, r74, r68);
    r75 = r73 * r32;
    r74 = r74 + r75;
    r74 = fmaf(r71, r35, r74);
    r74 = fmaf(r11, r74, r12 * r63);
    r63 = r29 * r65;
    r76 = -4.00000000000000000e+00;
    r63 = r63 * r76;
    r77 = r25 * r71;
    r78 = r76 * r77;
    r79 = r63 + r78;
    r74 = fmaf(r10, r79, r74);
    r3 = r3 * r74;
    r79 = r31 * r28;
    r80 = r31 * r38;
    r81 = r14 * r7;
    r82 = r73 * r35;
    r81 = fmaf(r69, r81, r82);
    r83 = r25 * r31;
    r84 = r71 * r32;
    r83 = fmaf(r65, r83, r84);
    r81 = r81 + r83;
    r85 = r7 * r76;
    r86 = r73 * r85;
    r78 = r78 + r86;
    r78 = fmaf(r11, r78, r12 * r81);
    r81 = r31 * r34;
    r81 = fmaf(r71, r81, r68);
    r68 = r25 * r31;
    r68 = fmaf(r69, r68, r75);
    r81 = r81 + r68;
    r78 = fmaf(r10, r81, r78);
    r80 = fmaf(r78, r80, r74 * r79);
    r27 = rsqrtf(r27);
    r27 = r66 * r27;
    r50 = r50 * r50;
    r50 = 1.0 / r50;
    r79 = r49 * r50;
    r27 = r27 * r79;
    r74 = r80 * r27;
    r3 = fmaf(r56, r74, r2 * r3);
    r81 = -5.00000000000000000e+00;
    r81 = r55 * r81;
    r41 = r41 * r41;
    r41 = fmaf(r6, r41, r13);
    r41 = rsqrtf(r41);
    r81 = r81 * r41;
    r81 = r81 * r39;
    r39 = r37 * r66;
    r55 = r30 * r30;
    r55 = r30 * r55;
    r55 = rsqrtf(r55);
    r39 = r39 * r55;
    r55 = r31 * r37;
    r30 = r31 * r7;
    r30 = r30 * r69;
    r75 = r31 * r34;
    r75 = r75 * r73;
    r87 = r30 + r75;
    r83 = r83 + r87;
    r88 = r14 * r29;
    r65 = fmaf(r65, r35, r69 * r88);
    r65 = r65 + r70;
    r65 = fmaf(r10, r65, r11 * r83);
    r86 = r63 + r86;
    r65 = fmaf(r12, r86, r65);
    r55 = fmaf(r65, r55, r80);
    r65 = fmaf(r65, r33, r55 * r39);
    r55 = -9.00000000000000000e+00;
    r55 = r1 * r55;
    r55 = r55 * r41;
    r55 = r55 * r61;
    r1 = fmaf(r65, r55, r65 * r81);
    r80 = -3.00000000000000000e+00;
    r80 = r54 * r80;
    r80 = r80 * r41;
    r80 = r80 * r40;
    r40 = -7.00000000000000000e+00;
    r40 = r0 * r40;
    r40 = r40 * r41;
    r40 = r40 * r59;
    r0 = r6 * r65;
    r1 = fmaf(r41, r0, r1);
    r1 = fmaf(r65, r80, r1);
    r1 = fmaf(r65, r40, r1);
    r3 = fmaf(r1, r57, r3);
    r0 = r53 * r49;
    r0 = r0 * r78;
    r74 = fmaf(r62, r74, r2 * r0);
    r0 = r1 * r2;
    r74 = fmaf(r62, r0, r74);
    r0 = r14 * r29;
    r0 = fmaf(r71, r0, r82);
    r78 = r25 * r31;
    r54 = r18 * r21;
    r23 = fmaf(r66, r23, r66 * r54);
    r23 = fmaf(r66, r24, r23);
    r23 = fmaf(r64, r26, r23);
    r78 = r78 * r23;
    r26 = r31 * r7;
    r24 = r15 * r22;
    r54 = r17 * r20;
    r54 = fmaf(r64, r54, r64 * r24);
    r24 = r16 * r21;
    r54 = fmaf(r66, r24, r54);
    r54 = fmaf(r19, r67, r54);
    r26 = fmaf(r54, r26, r78);
    r0 = r0 + r26;
    r24 = r31 * r34;
    r86 = r54 * r32;
    r24 = fmaf(r23, r24, r86);
    r24 = r24 + r70;
    r24 = fmaf(r11, r24, r10 * r0);
    r0 = r29 * r73;
    r0 = r0 * r76;
    r70 = r23 * r85;
    r63 = r0 + r70;
    r24 = fmaf(r12, r63, r24);
    r63 = r31 * r37;
    r83 = r31 * r38;
    r88 = r14 * r7;
    r88 = fmaf(r71, r88, r72);
    r88 = r88 + r86;
    r88 = fmaf(r23, r35, r88);
    r86 = r31 * r34;
    r86 = fmaf(r31, r77, r54 * r86);
    r72 = r31 * r7;
    r72 = r72 * r73;
    r71 = fmaf(r23, r32, r72);
    r86 = r86 + r71;
    r86 = fmaf(r10, r86, r12 * r88);
    r88 = r25 * r76;
    r88 = r88 * r54;
    r70 = r88 + r70;
    r86 = fmaf(r11, r70, r86);
    r70 = r31 * r28;
    r84 = r75 + r84;
    r84 = r84 + r26;
    r88 = r0 + r88;
    r88 = fmaf(r10, r88, r12 * r84);
    r54 = fmaf(r54, r35, r14 * r77);
    r54 = r54 + r71;
    r88 = fmaf(r11, r54, r88);
    r70 = fmaf(r88, r70, r86 * r83);
    r63 = fmaf(r24, r63, r70);
    r63 = fmaf(r63, r39, r24 * r33);
    r24 = r6 * r63;
    r24 = fmaf(r63, r40, r41 * r24);
    r24 = fmaf(r63, r81, r24);
    r24 = fmaf(r63, r80, r24);
    r24 = fmaf(r63, r55, r24);
    r83 = r52 * r49;
    r83 = r83 * r88;
    r83 = fmaf(r2, r83, r24 * r57);
    r88 = r70 * r56;
    r83 = fmaf(r27, r88, r83);
    r88 = r53 * r49;
    r88 = r88 * r86;
    r86 = r24 * r2;
    r86 = fmaf(r62, r86, r2 * r88);
    r88 = r70 * r62;
    r86 = fmaf(r27, r88, r86);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r3,
                                          r74,
                                          r83,
                                          r86);
    r88 = r52 * r49;
    r54 = r14 * r25;
    r54 = fmaf(r23, r54, r30);
    r30 = r16 * r22;
    r77 = r17 * r19;
    r77 = fmaf(r66, r77, r64 * r30);
    r30 = r15 * r21;
    r77 = fmaf(r64, r30, r77);
    r77 = fmaf(r20, r67, r77);
    r32 = r77 * r32;
    r54 = r54 + r82;
    r54 = r54 + r32;
    r73 = r25 * r73;
    r73 = r73 * r76;
    r82 = r29 * r69;
    r82 = r82 * r76;
    r76 = r73 + r82;
    r76 = fmaf(r10, r76, r11 * r54);
    r54 = r25 * r31;
    r54 = r54 * r77;
    r67 = r31 * r34;
    r67 = fmaf(r69, r67, r54);
    r67 = r67 + r71;
    r76 = fmaf(r12, r67, r76);
    r88 = r88 * r76;
    r85 = r77 * r85;
    r82 = r82 + r85;
    r54 = r72 + r54;
    r72 = r14 * r29;
    r54 = fmaf(r23, r72, r54);
    r54 = fmaf(r69, r35, r54);
    r54 = fmaf(r10, r54, r12 * r82);
    r82 = r31 * r7;
    r69 = r31 * r34;
    r69 = fmaf(r77, r69, r23 * r82);
    r69 = r69 + r68;
    r54 = fmaf(r11, r69, r54);
    r69 = r31 * r37;
    r82 = r31 * r38;
    r32 = r78 + r32;
    r32 = r32 + r87;
    r87 = r14 * r7;
    r35 = fmaf(r77, r35, r23 * r87);
    r35 = r35 + r68;
    r35 = fmaf(r12, r35, r10 * r32);
    r85 = r73 + r85;
    r35 = fmaf(r11, r85, r35);
    r85 = r31 * r28;
    r85 = fmaf(r76, r85, r35 * r82);
    r69 = fmaf(r54, r69, r85);
    r69 = fmaf(r69, r39, r54 * r33);
    r54 = r6 * r69;
    r54 = fmaf(r69, r40, r41 * r54);
    r54 = fmaf(r69, r80, r54);
    r54 = fmaf(r69, r81, r54);
    r54 = fmaf(r69, r55, r54);
    r88 = fmaf(r54, r57, r2 * r88);
    r82 = r85 * r56;
    r88 = fmaf(r27, r82, r88);
    r82 = r85 * r62;
    r76 = r54 * r2;
    r76 = fmaf(r62, r76, r27 * r82);
    r82 = r53 * r49;
    r82 = r82 * r35;
    r76 = fmaf(r2, r82, r76);
    r82 = r31 * r48;
    r35 = r31 * r8;
    r35 = fmaf(r38, r35, r28 * r82);
    r82 = r35 * r56;
    r11 = r52 * r48;
    r11 = r11 * r49;
    r11 = fmaf(r2, r11, r27 * r82);
    r82 = r31 * r9;
    r82 = fmaf(r37, r82, r35);
    r82 = fmaf(r82, r39, r9 * r33);
    r73 = fmaf(r82, r81, r82 * r40);
    r12 = r6 * r82;
    r73 = fmaf(r41, r12, r73);
    r73 = fmaf(r82, r80, r73);
    r73 = fmaf(r82, r55, r73);
    r11 = fmaf(r73, r57, r11);
    r12 = r73 * r2;
    r32 = r53 * r8;
    r32 = r32 * r49;
    r32 = fmaf(r2, r32, r62 * r12);
    r12 = r35 * r62;
    r32 = fmaf(r27, r12, r32);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r88,
                                          r76,
                                          r11,
                                          r32);
    r12 = r52 * r44;
    r12 = r12 * r49;
    r10 = r31 * r44;
    r68 = r31 * r46;
    r68 = fmaf(r38, r68, r28 * r10);
    r10 = r68 * r56;
    r10 = fmaf(r27, r10, r2 * r12);
    r12 = r31 * r45;
    r12 = fmaf(r37, r12, r68);
    r12 = fmaf(r45, r33, r12 * r39);
    r77 = r6 * r12;
    r77 = fmaf(r12, r40, r41 * r77);
    r77 = fmaf(r12, r81, r77);
    r77 = fmaf(r12, r80, r77);
    r77 = fmaf(r12, r55, r77);
    r10 = fmaf(r77, r57, r10);
    r87 = r53 * r46;
    r87 = r87 * r49;
    r23 = r68 * r62;
    r23 = fmaf(r27, r23, r2 * r87);
    r87 = r77 * r2;
    r23 = fmaf(r62, r87, r23);
    r87 = r31 * r43;
    r78 = r31 * r36;
    r78 = fmaf(r38, r78, r28 * r87);
    r87 = r78 * r56;
    r72 = r52 * r43;
    r72 = r72 * r49;
    r72 = fmaf(r2, r72, r27 * r87);
    r87 = r31 * r47;
    r87 = fmaf(r37, r87, r78);
    r33 = fmaf(r47, r33, r87 * r39);
    r80 = fmaf(r33, r80, r33 * r40);
    r40 = r6 * r33;
    r80 = fmaf(r41, r40, r80);
    r80 = fmaf(r33, r81, r80);
    r80 = fmaf(r33, r55, r80);
    r72 = fmaf(r80, r57, r72);
    r40 = r80 * r2;
    r55 = r53 * r36;
    r55 = r55 * r49;
    r55 = fmaf(r2, r55, r62 * r40);
    r40 = r78 * r62;
    r55 = fmaf(r27, r40, r55);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r10,
                                          r23,
                                          r72,
                                          r55);
    r40 = r6 * r5;
    r4 = r6 * r4;
    r40 = fmaf(r3, r4, r74 * r40);
    r27 = r6 * r5;
    r27 = fmaf(r83, r4, r86 * r27);
    r81 = r6 * r5;
    r81 = fmaf(r88, r4, r76 * r81);
    r41 = r6 * r5;
    r41 = fmaf(r11, r4, r32 * r41);
    WriteSum4<float, float>((float*)inout_shared, r40, r27, r81, r41);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r6 * r5;
    r41 = fmaf(r10, r4, r23 * r41);
    r81 = r6 * r5;
    r81 = fmaf(r72, r4, r55 * r81);
    WriteSum2<float, float>((float*)inout_shared, r41, r81);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fmaf(r3, r3, r74 * r74);
    r41 = fmaf(r83, r83, r86 * r86);
    r27 = fmaf(r88, r88, r76 * r76);
    r40 = fmaf(r32, r32, r11 * r11);
    WriteSum4<float, float>((float*)inout_shared, r81, r41, r27, r40);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r10, r10, r23 * r23);
    r27 = fmaf(r72, r72, r55 * r55);
    WriteSum2<float, float>((float*)inout_shared, r40, r27);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r74, r86, r3 * r83);
    r40 = fmaf(r74, r76, r3 * r88);
    r41 = fmaf(r3, r11, r74 * r32);
    r81 = fmaf(r74, r23, r3 * r10);
    WriteSum4<float, float>((float*)inout_shared, r27, r40, r41, r81);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r74, r55, r3 * r72);
    r3 = fmaf(r86, r76, r83 * r88);
    r81 = fmaf(r86, r32, r83 * r11);
    r41 = fmaf(r86, r23, r83 * r10);
    WriteSum4<float, float>((float*)inout_shared, r74, r3, r81, r41);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fmaf(r86, r55, r83 * r72);
    r83 = fmaf(r88, r11, r76 * r32);
    r41 = fmaf(r76, r23, r88 * r10);
    r88 = fmaf(r88, r72, r76 * r55);
    WriteSum4<float, float>((float*)inout_shared, r86, r83, r41, r88);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = fmaf(r11, r10, r32 * r23);
    r32 = fmaf(r32, r55, r11 * r72);
    r72 = fmaf(r10, r72, r23 * r55);
    WriteSum3<float, float>((float*)inout_shared, r88, r32, r72);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r28 * r49;
    r72 = r72 * r2;
    r32 = r38 * r49;
    r32 = r32 * r2;
    r88 = r42 * r57;
    r10 = r2 * r42;
    r10 = r10 * r62;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r72,
                                          r32,
                                          r88,
                                          r10);
    r55 = r51 * r57;
    r23 = r2 * r51;
    r23 = r23 * r62;
    r11 = r60 * r57;
    r41 = r2 * r60;
    r41 = r41 * r62;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r55,
                                          r23,
                                          r11,
                                          r41);
    r83 = r58 * r57;
    r86 = r2 * r62;
    r86 = r86 * r58;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r83,
                                          r86);
    r76 = r28 * r49;
    r76 = r76 * r2;
    r76 = r76 * r4;
    r81 = r6 * r38;
    r81 = r81 * r49;
    r81 = r81 * r5;
    r81 = r81 * r2;
    r3 = r6 * r5;
    r3 = r3 * r2;
    r3 = r3 * r42;
    r57 = r57 * r4;
    r3 = fmaf(r42, r57, r62 * r3);
    r74 = r6 * r5;
    r74 = r74 * r2;
    r74 = r74 * r51;
    r74 = fmaf(r51, r57, r62 * r74);
    WriteSum4<float, float>((float*)inout_shared, r76, r81, r3, r74);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r6 * r5;
    r3 = r6 * r5;
    r3 = r3 * r2;
    r3 = r3 * r60;
    r3 = fmaf(r60, r57, r62 * r3);
    r81 = r6 * r5;
    r81 = r81 * r2;
    r81 = r81 * r62;
    r57 = fmaf(r58, r57, r58 * r81);
    WriteSum4<float, float>((float*)inout_shared, r3, r57, r4, r74);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r28 * r28;
    r74 = r74 * r49;
    r74 = r74 * r79;
    r4 = r38 * r38;
    r4 = r4 * r49;
    r4 = r4 * r79;
    r57 = r50 * r56;
    r57 = r57 * r56;
    r50 = r53 * r50;
    r3 = r38 * r62;
    r50 = r50 * r3;
    r81 = fmaf(r59, r50, r59 * r57);
    r76 = r51 * r51;
    r76 = fmaf(r76, r50, r76 * r57);
    WriteSum4<float, float>((float*)inout_shared, r74, r4, r81, r76);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r60 * r60;
    r81 = fmaf(r81, r50, r81 * r57);
    r4 = r58 * r58;
    r4 = fmaf(r50, r4, r57 * r4);
    WriteSum4<float, float>((float*)inout_shared, r81, r4, r13, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = 0.00000000000000000e+00;
    r4 = r28 * r42;
    r4 = r4 * r56;
    r4 = r4 * r79;
    r74 = r28 * r51;
    r74 = r74 * r56;
    r74 = r74 * r79;
    r40 = r28 * r60;
    r40 = r40 * r56;
    r40 = r40 * r79;
    WriteSum4<float, float>((float*)inout_shared, r13, r4, r74, r40);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r28 * r56;
    r40 = r40 * r58;
    r40 = r40 * r79;
    r42 = r42 * r79;
    r42 = r42 * r3;
    WriteSum4<float, float>((float*)inout_shared, r40, r72, r13, r42);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r51 * r79;
    r51 = r51 * r3;
    r60 = r60 * r79;
    r60 = r60 * r3;
    r79 = r58 * r79;
    r79 = r79 * r3;
    WriteSum4<float, float>((float*)inout_shared, r51, r60, r79, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fmaf(r61, r50, r61 * r57);
    r59 = r59 * r59;
    r59 = fmaf(r59, r50, r59 * r57);
    WriteSum4<float, float>((float*)inout_shared, r32, r79, r76, r59);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r88, r10, r59, r81);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r61 * r61;
    r61 = fmaf(r50, r61, r57 * r61);
    WriteSum4<float, float>((float*)inout_shared, r55, r23, r61, r11);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r41, r83, r86, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
}

void OpencvFixedPointResJac(float* pose,
                            unsigned int pose_num_alloc,
                            SharedIndex* pose_indices,
                            float* sensor_from_rig,
                            unsigned int sensor_from_rig_num_alloc,
                            float* calib,
                            unsigned int calib_num_alloc,
                            SharedIndex* calib_indices,
                            float* pixel,
                            unsigned int pixel_num_alloc,
                            float* point,
                            unsigned int point_num_alloc,
                            float* out_res,
                            unsigned int out_res_num_alloc,
                            float* out_pose_jac,
                            unsigned int out_pose_jac_num_alloc,
                            float* const out_pose_njtr,
                            unsigned int out_pose_njtr_num_alloc,
                            float* const out_pose_precond_diag,
                            unsigned int out_pose_precond_diag_num_alloc,
                            float* const out_pose_precond_tril,
                            unsigned int out_pose_precond_tril_num_alloc,
                            float* out_calib_jac,
                            unsigned int out_calib_jac_num_alloc,
                            float* const out_calib_njtr,
                            unsigned int out_calib_njtr_num_alloc,
                            float* const out_calib_precond_diag,
                            unsigned int out_calib_precond_diag_num_alloc,
                            float* const out_calib_precond_tril,
                            unsigned int out_calib_precond_tril_num_alloc,
                            size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar