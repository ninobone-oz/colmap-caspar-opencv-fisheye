#include "kernel_opencv_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89;

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
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
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
    r26 = fmaf(r8, r26, r5);
    r5 = fmaf(r16, r17, r13 * r20);
    r29 = r14 * r19;
    r5 = fmaf(r4, r29, r5);
    r5 = fmaf(r15, r18, r5);
    r29 = 2.00000000000000000e+00;
    r30 = r29 * r27;
    r31 = r5 * r30;
    r32 = fmaf(r14, r18, r13 * r17);
    r32 = fmaf(r15, r19, r32);
    r32 = fmaf(r4, r32, r16 * r20);
    r33 = r12 * r32;
    r34 = fmaf(r23, r33, r31);
    r35 = r23 * r29;
    r35 = r35 * r5;
    r36 = fmaf(r32, r30, r35);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r37,
                       r38,
                       r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r17 * r19;
    r40 = r40 * r29;
    r41 = r18 * r20;
    r41 = fmaf(r29, r41, r40);
    r42 = r19 * r20;
    r43 = r17 * r18;
    r43 = r43 * r29;
    r42 = fmaf(r12, r42, r43);
    r44 = r19 * r19;
    r44 = r12 * r44;
    r45 = r18 * r18;
    r45 = fmaf(r12, r45, r11);
    r46 = r44 + r45;
    r26 = fmaf(r9, r34, r26);
    r26 = fmaf(r10, r36, r26);
    r26 = fmaf(r39, r41, r26);
    r26 = fmaf(r38, r42, r26);
    r26 = fmaf(r37, r46, r26);
    r36 = r23 * r29;
    r36 = fmaf(r32, r36, r31);
    r36 = fmaf(r8, r36, r6);
    r6 = r19 * r20;
    r6 = fmaf(r29, r6, r43);
    r44 = r11 + r44;
    r43 = r17 * r17;
    r43 = r43 * r12;
    r44 = r44 + r43;
    r31 = r18 * r19;
    r31 = r31 * r29;
    r34 = r17 * r20;
    r34 = fmaf(r12, r34, r31);
    r47 = r23 * r30;
    r48 = fmaf(r5, r33, r47);
    r25 = r11 + r25;
    r49 = r12 * r5;
    r49 = r49 * r5;
    r25 = r25 + r49;
    r36 = fmaf(r37, r6, r36);
    r36 = fmaf(r38, r44, r36);
    r36 = fmaf(r39, r34, r36);
    r36 = fmaf(r10, r48, r36);
    r36 = fmaf(r9, r25, r36);
    r25 = fmaf(r36, r36, r26 * r26);
    r48 = sqrtf(r25);
    r48 = r0 + r48;
    r0 = 1.0 / r48;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r50,
                       r51,
                       r52,
                       r53);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r54 = r50 * r26;
    r55 = r0 * r54;
    r56 = -9.99999999999999955e-07;
    r35 = fmaf(r27, r33, r35);
    r35 = fmaf(r8, r35, r7);
    r7 = r18 * r20;
    r7 = fmaf(r12, r7, r40);
    r45 = r43 + r45;
    r43 = r17 * r20;
    r43 = fmaf(r29, r43, r31);
    r31 = r29 * r5;
    r31 = fmaf(r32, r31, r47);
    r28 = r11 + r28;
    r28 = r28 + r49;
    r35 = fmaf(r37, r7, r35);
    r35 = fmaf(r39, r45, r35);
    r35 = fmaf(r38, r43, r35);
    r35 = fmaf(r9, r31, r35);
    r35 = fmaf(r10, r28, r35);
    r28 = fmaf(r35, r35, r25);
    r31 = rsqrtf(r28);
    r38 = r35 * r31;
    r39 = copysign(1.0, r38);
    r39 = fmaf(r56, r39, r38);
    r56 = acosf(r39);
  };
  LoadShared<2, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r38,
                       r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r49 = r56 * r56;
    r47 = r49 * r49;
    r40 = r47 * r47;
    r57 = r56 * r40;
    r58 = fmaf(r37, r57, r56);
    r59 = r56 * r49;
    r60 = r59 * r59;
    r61 = r56 * r60;
    r56 = r56 * r47;
    r58 = fmaf(r52, r59, r58);
    r58 = fmaf(r38, r61, r58);
    r58 = fmaf(r53, r56, r58);
    r2 = fmaf(r58, r55, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r58 * r0;
    r62 = r51 * r36;
    r3 = fmaf(r62, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r50 * r58;
    r63 = r29 * r32;
    r64 = 5.00000000000000000e-01;
    r65 = fmaf(r64, r22, r64 * r21);
    r66 = -5.00000000000000000e-01;
    r67 = r16 * r64;
    r65 = fmaf(r66, r24, r65);
    r65 = fmaf(r19, r67, r65);
    r68 = r13 * r20;
    r69 = r16 * r17;
    r69 = fmaf(r66, r69, r66 * r68);
    r68 = r15 * r18;
    r69 = fmaf(r66, r68, r69);
    r70 = r14 * r19;
    r69 = fmaf(r64, r70, r69);
    r63 = fmaf(r69, r30, r65 * r63);
    r70 = r29 * r5;
    r68 = r14 * r20;
    r71 = r15 * r17;
    r71 = fmaf(r64, r71, r66 * r68);
    r68 = r16 * r18;
    r71 = fmaf(r66, r68, r71);
    r72 = r13 * r19;
    r71 = fmaf(r66, r72, r71);
    r72 = r23 * r29;
    r68 = r13 * r17;
    r73 = r14 * r18;
    r73 = fmaf(r66, r73, r66 * r68);
    r68 = r15 * r19;
    r73 = fmaf(r66, r68, r73);
    r73 = fmaf(r20, r67, r73);
    r72 = r72 * r73;
    r70 = fmaf(r71, r70, r72);
    r63 = r63 + r70;
    r68 = r29 * r5;
    r68 = r68 * r65;
    r74 = r12 * r23;
    r74 = fmaf(r69, r74, r68);
    r75 = r73 * r30;
    r74 = r74 + r75;
    r74 = fmaf(r71, r33, r74);
    r74 = fmaf(r9, r74, r10 * r63);
    r63 = r27 * r65;
    r76 = -4.00000000000000000e+00;
    r63 = r63 * r76;
    r77 = r23 * r71;
    r78 = r76 * r77;
    r79 = r63 + r78;
    r74 = fmaf(r8, r79, r74);
    r1 = r1 * r74;
    r79 = r29 * r26;
    r80 = r29 * r36;
    r81 = r12 * r5;
    r82 = r73 * r33;
    r81 = fmaf(r69, r81, r82);
    r83 = r23 * r29;
    r84 = r71 * r30;
    r83 = fmaf(r65, r83, r84);
    r81 = r81 + r83;
    r85 = r5 * r76;
    r86 = r73 * r85;
    r78 = r78 + r86;
    r78 = fmaf(r9, r78, r10 * r81);
    r81 = r29 * r32;
    r81 = fmaf(r71, r81, r68);
    r68 = r23 * r29;
    r68 = fmaf(r69, r68, r75);
    r81 = r81 + r68;
    r78 = fmaf(r8, r81, r78);
    r80 = fmaf(r78, r80, r74 * r79);
    r25 = rsqrtf(r25);
    r25 = r66 * r25;
    r48 = r48 * r48;
    r48 = 1.0 / r48;
    r79 = r58 * r48;
    r25 = r25 * r79;
    r74 = r80 * r25;
    r1 = fmaf(r54, r74, r0 * r1);
    r81 = r35 * r66;
    r75 = r28 * r28;
    r75 = r28 * r75;
    r75 = rsqrtf(r75);
    r81 = r81 * r75;
    r75 = r29 * r35;
    r28 = r29 * r5;
    r28 = r28 * r69;
    r87 = r29 * r32;
    r87 = r87 * r73;
    r88 = r28 + r87;
    r83 = r83 + r88;
    r89 = r12 * r27;
    r65 = fmaf(r65, r33, r69 * r89);
    r65 = r65 + r70;
    r65 = fmaf(r8, r65, r9 * r83);
    r86 = r63 + r86;
    r65 = fmaf(r10, r86, r65);
    r75 = fmaf(r65, r75, r80);
    r65 = fmaf(r65, r31, r75 * r81);
    r75 = r4 * r65;
    r39 = r39 * r39;
    r39 = fmaf(r4, r39, r11);
    r39 = rsqrtf(r39);
    r11 = -7.00000000000000000e+00;
    r11 = r38 * r11;
    r11 = r11 * r39;
    r11 = r11 * r60;
    r75 = fmaf(r65, r11, r39 * r75);
    r38 = -3.00000000000000000e+00;
    r38 = r52 * r38;
    r38 = r38 * r39;
    r38 = r38 * r49;
    r49 = -9.00000000000000000e+00;
    r49 = r37 * r49;
    r37 = r47 * r47;
    r49 = r49 * r39;
    r49 = r49 * r37;
    r52 = -5.00000000000000000e+00;
    r52 = r53 * r52;
    r52 = r52 * r39;
    r52 = r52 * r47;
    r75 = fmaf(r65, r38, r75);
    r75 = fmaf(r65, r49, r75);
    r75 = fmaf(r65, r52, r75);
    r1 = fmaf(r75, r55, r1);
    r47 = r51 * r58;
    r47 = r47 * r78;
    r78 = r75 * r0;
    r78 = fmaf(r62, r78, r0 * r47);
    r78 = fmaf(r62, r74, r78);
    r74 = r29 * r36;
    r47 = r12 * r5;
    r47 = fmaf(r71, r47, r72);
    r72 = r13 * r20;
    r53 = r15 * r18;
    r53 = fmaf(r64, r53, r64 * r72);
    r72 = r14 * r19;
    r53 = fmaf(r66, r72, r53);
    r53 = fmaf(r17, r67, r53);
    r72 = r53 * r30;
    r80 = r16 * r19;
    r21 = fmaf(r66, r21, r66 * r80);
    r21 = fmaf(r66, r22, r21);
    r21 = fmaf(r64, r24, r21);
    r47 = r47 + r72;
    r47 = fmaf(r21, r33, r47);
    r24 = r29 * r32;
    r24 = fmaf(r29, r77, r53 * r24);
    r22 = r29 * r5;
    r22 = r22 * r73;
    r80 = fmaf(r21, r30, r22);
    r24 = r24 + r80;
    r24 = fmaf(r8, r24, r10 * r47);
    r47 = r23 * r76;
    r47 = r47 * r53;
    r86 = r21 * r85;
    r63 = r47 + r86;
    r24 = fmaf(r9, r63, r24);
    r63 = r29 * r26;
    r84 = r87 + r84;
    r87 = r23 * r29;
    r87 = r87 * r21;
    r83 = r29 * r5;
    r83 = fmaf(r53, r83, r87);
    r84 = r84 + r83;
    r89 = r27 * r73;
    r89 = r89 * r76;
    r47 = r47 + r89;
    r47 = fmaf(r8, r47, r10 * r84);
    r53 = fmaf(r53, r33, r12 * r77);
    r53 = r53 + r80;
    r47 = fmaf(r9, r53, r47);
    r63 = fmaf(r47, r63, r24 * r74);
    r74 = r63 * r54;
    r53 = r12 * r27;
    r53 = fmaf(r71, r53, r82);
    r53 = r53 + r83;
    r83 = r29 * r32;
    r83 = fmaf(r21, r83, r72);
    r83 = r83 + r70;
    r83 = fmaf(r9, r83, r8 * r53);
    r86 = r89 + r86;
    r83 = fmaf(r10, r86, r83);
    r86 = r29 * r35;
    r86 = fmaf(r83, r86, r63);
    r86 = fmaf(r86, r81, r83 * r31);
    r83 = fmaf(r86, r11, r86 * r49);
    r89 = r4 * r86;
    r83 = fmaf(r39, r89, r83);
    r83 = fmaf(r86, r52, r83);
    r83 = fmaf(r86, r38, r83);
    r74 = fmaf(r83, r55, r25 * r74);
    r89 = r50 * r58;
    r89 = r89 * r47;
    r74 = fmaf(r0, r89, r74);
    r89 = r63 * r62;
    r47 = r83 * r0;
    r47 = fmaf(r62, r47, r25 * r89);
    r89 = r51 * r58;
    r89 = r89 * r24;
    r47 = fmaf(r0, r89, r47);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r78,
                                          r74,
                                          r47);
    r89 = r27 * r69;
    r89 = r89 * r76;
    r24 = r14 * r20;
    r53 = r15 * r17;
    r53 = fmaf(r66, r53, r64 * r24);
    r24 = r13 * r19;
    r53 = fmaf(r64, r24, r53);
    r53 = fmaf(r18, r67, r53);
    r85 = r53 * r85;
    r67 = r89 + r85;
    r24 = r23 * r29;
    r24 = r24 * r53;
    r22 = r22 + r24;
    r64 = r12 * r27;
    r22 = fmaf(r21, r64, r22);
    r22 = fmaf(r69, r33, r22);
    r22 = fmaf(r8, r22, r10 * r67);
    r67 = r29 * r5;
    r64 = r29 * r32;
    r64 = fmaf(r53, r64, r21 * r67);
    r64 = r64 + r68;
    r22 = fmaf(r9, r64, r22);
    r64 = r29 * r35;
    r67 = r29 * r36;
    r30 = r53 * r30;
    r87 = r87 + r30;
    r87 = r87 + r88;
    r88 = r12 * r5;
    r33 = fmaf(r53, r33, r21 * r88);
    r33 = r33 + r68;
    r33 = fmaf(r10, r33, r8 * r87);
    r73 = r23 * r73;
    r73 = r73 * r76;
    r85 = r73 + r85;
    r33 = fmaf(r9, r85, r33);
    r85 = r29 * r26;
    r76 = r12 * r23;
    r76 = fmaf(r21, r76, r28);
    r76 = r76 + r82;
    r76 = r76 + r30;
    r73 = r89 + r73;
    r73 = fmaf(r8, r73, r9 * r76);
    r8 = r29 * r32;
    r8 = fmaf(r69, r8, r24);
    r8 = r8 + r80;
    r73 = fmaf(r10, r8, r73);
    r85 = fmaf(r73, r85, r33 * r67);
    r64 = fmaf(r22, r64, r85);
    r64 = fmaf(r64, r81, r22 * r31);
    r22 = r4 * r64;
    r22 = fmaf(r64, r38, r39 * r22);
    r22 = fmaf(r64, r11, r22);
    r22 = fmaf(r64, r52, r22);
    r22 = fmaf(r64, r49, r22);
    r67 = r85 * r54;
    r67 = fmaf(r25, r67, r22 * r55);
    r8 = r50 * r58;
    r8 = r8 * r73;
    r67 = fmaf(r0, r8, r67);
    r8 = r85 * r62;
    r73 = r22 * r0;
    r73 = fmaf(r62, r73, r25 * r8);
    r8 = r51 * r58;
    r8 = r8 * r33;
    r73 = fmaf(r0, r8, r73);
    r8 = r29 * r46;
    r33 = r29 * r6;
    r33 = fmaf(r36, r33, r26 * r8);
    r8 = r33 * r54;
    r10 = r50 * r46;
    r10 = r10 * r58;
    r10 = fmaf(r0, r10, r25 * r8);
    r8 = r29 * r7;
    r8 = fmaf(r35, r8, r33);
    r8 = fmaf(r8, r81, r7 * r31);
    r80 = r4 * r8;
    r80 = fmaf(r8, r11, r39 * r80);
    r80 = fmaf(r8, r49, r80);
    r80 = fmaf(r8, r52, r80);
    r80 = fmaf(r8, r38, r80);
    r10 = fmaf(r80, r55, r10);
    r24 = r33 * r62;
    r69 = r51 * r6;
    r69 = r69 * r58;
    r69 = fmaf(r0, r69, r25 * r24);
    r24 = r80 * r0;
    r69 = fmaf(r62, r24, r69);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r73,
                                          r10,
                                          r69);
    r24 = r29 * r42;
    r76 = r29 * r44;
    r76 = fmaf(r36, r76, r26 * r24);
    r24 = r76 * r54;
    r9 = r29 * r43;
    r9 = fmaf(r35, r9, r76);
    r9 = fmaf(r43, r31, r9 * r81);
    r89 = r4 * r9;
    r89 = fmaf(r9, r49, r39 * r89);
    r89 = fmaf(r9, r11, r89);
    r89 = fmaf(r9, r38, r89);
    r89 = fmaf(r9, r52, r89);
    r24 = fmaf(r89, r55, r25 * r24);
    r30 = r50 * r42;
    r30 = r30 * r58;
    r24 = fmaf(r0, r30, r24);
    r30 = r89 * r0;
    r82 = r76 * r62;
    r82 = fmaf(r25, r82, r62 * r30);
    r30 = r51 * r44;
    r30 = r30 * r58;
    r82 = fmaf(r0, r30, r82);
    r30 = r50 * r41;
    r30 = r30 * r58;
    r28 = r29 * r41;
    r21 = r29 * r34;
    r21 = fmaf(r36, r21, r26 * r28);
    r28 = r21 * r54;
    r28 = fmaf(r25, r28, r0 * r30);
    r30 = r29 * r45;
    r30 = fmaf(r35, r30, r21);
    r31 = fmaf(r45, r31, r30 * r81);
    r11 = fmaf(r31, r11, r31 * r52);
    r52 = r4 * r31;
    r11 = fmaf(r39, r52, r11);
    r11 = fmaf(r31, r38, r11);
    r11 = fmaf(r31, r49, r11);
    r28 = fmaf(r11, r55, r28);
    r52 = r11 * r0;
    r49 = r21 * r62;
    r49 = fmaf(r25, r49, r62 * r52);
    r52 = r51 * r34;
    r52 = r52 * r58;
    r49 = fmaf(r0, r52, r49);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r24,
                                          r82,
                                          r28,
                                          r49);
    r52 = r4 * r2;
    r3 = r4 * r3;
    r52 = fmaf(r78, r3, r1 * r52);
    r25 = r4 * r2;
    r25 = fmaf(r47, r3, r74 * r25);
    r38 = r4 * r2;
    r38 = fmaf(r73, r3, r67 * r38);
    r39 = r4 * r2;
    r39 = fmaf(r69, r3, r10 * r39);
    WriteSum4<float, float>((float*)inout_shared, r52, r25, r38, r39);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r4 * r2;
    r39 = fmaf(r82, r3, r24 * r39);
    r38 = r4 * r2;
    r38 = fmaf(r49, r3, r28 * r38);
    WriteSum2<float, float>((float*)inout_shared, r39, r38);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r78, r78, r1 * r1);
    r39 = fmaf(r74, r74, r47 * r47);
    r25 = fmaf(r73, r73, r67 * r67);
    r52 = fmaf(r69, r69, r10 * r10);
    WriteSum4<float, float>((float*)inout_shared, r38, r39, r25, r52);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r82, r82, r24 * r24);
    r25 = fmaf(r49, r49, r28 * r28);
    WriteSum2<float, float>((float*)inout_shared, r52, r25);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fmaf(r1, r74, r78 * r47);
    r52 = fmaf(r78, r73, r1 * r67);
    r39 = fmaf(r78, r69, r1 * r10);
    r38 = fmaf(r78, r82, r1 * r24);
    WriteSum4<float, float>((float*)inout_shared, r25, r52, r39, r38);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fmaf(r78, r49, r1 * r28);
    r1 = fmaf(r47, r73, r74 * r67);
    r38 = fmaf(r74, r10, r47 * r69);
    r39 = fmaf(r47, r82, r74 * r24);
    WriteSum4<float, float>((float*)inout_shared, r78, r1, r38, r39);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r74, r28, r47 * r49);
    r47 = fmaf(r73, r69, r67 * r10);
    r39 = fmaf(r67, r24, r73 * r82);
    r73 = fmaf(r73, r49, r67 * r28);
    WriteSum4<float, float>((float*)inout_shared, r74, r47, r39, r73);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r10, r24, r69 * r82);
    r10 = fmaf(r10, r28, r69 * r49);
    r28 = fmaf(r24, r28, r82 * r49);
    WriteSum3<float, float>((float*)inout_shared, r73, r10, r28);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r26 * r58;
    r28 = r28 * r0;
    r10 = r36 * r58;
    r10 = r10 * r0;
    r73 = r59 * r55;
    r24 = r0 * r59;
    r24 = r24 * r62;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r10,
                                          r73,
                                          r24);
    r24 = r56 * r55;
    r73 = r0 * r56;
    r73 = r73 * r62;
    r10 = r61 * r55;
    r28 = r0 * r61;
    r28 = r28 * r62;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r24,
                                          r73,
                                          r10,
                                          r28);
    r28 = r57 * r55;
    r10 = r0 * r62;
    r10 = r10 * r57;
    WriteIdx2<1024, float, float, float2>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r10);
    r10 = r4 * r26;
    r10 = r10 * r58;
    r10 = r10 * r2;
    r10 = r10 * r0;
    r28 = r36 * r58;
    r3 = r0 * r3;
    r28 = r28 * r3;
    r3 = r62 * r3;
    r73 = r4 * r2;
    r73 = r73 * r55;
    r55 = fmaf(r59, r73, r59 * r3);
    r24 = fmaf(r56, r73, r56 * r3);
    WriteSum4<float, float>((float*)inout_shared, r10, r28, r55, r24);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r61, r73, r61 * r3);
    r73 = fmaf(r57, r73, r57 * r3);
    WriteSum2<float, float>((float*)inout_shared, r24, r73);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r26 * r26;
    r73 = r73 * r58;
    r73 = r73 * r79;
    r24 = r36 * r36;
    r24 = r24 * r58;
    r24 = r24 * r79;
    r3 = r48 * r54;
    r3 = r3 * r54;
    r48 = r51 * r48;
    r55 = r36 * r62;
    r48 = r48 * r55;
    r28 = fmaf(r60, r48, r60 * r3);
    r10 = r56 * r56;
    r10 = fmaf(r10, r48, r10 * r3);
    WriteSum4<float, float>((float*)inout_shared, r73, r24, r28, r10);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r61 * r61;
    r28 = fmaf(r28, r48, r28 * r3);
    r24 = r57 * r57;
    r24 = fmaf(r48, r24, r3 * r24);
    WriteSum2<float, float>((float*)inout_shared, r28, r24);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = 0.00000000000000000e+00;
    r73 = r26 * r59;
    r73 = r73 * r54;
    r73 = r73 * r79;
    r49 = r26 * r56;
    r49 = r49 * r54;
    r49 = r49 * r79;
    r82 = r26 * r61;
    r82 = r82 * r54;
    r82 = r82 * r79;
    WriteSum4<float, float>((float*)inout_shared, r24, r73, r49, r82);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = r26 * r54;
    r82 = r82 * r57;
    r82 = r82 * r79;
    r59 = r59 * r79;
    r59 = r59 * r55;
    r56 = r56 * r79;
    r56 = r56 * r55;
    r61 = r61 * r79;
    r61 = r61 * r55;
    WriteSum4<float, float>((float*)inout_shared, r82, r59, r56, r61);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r57 * r79;
    r79 = r79 * r55;
    r37 = fmaf(r37, r48, r37 * r3);
    r60 = r60 * r60;
    r60 = fmaf(r60, r48, r60 * r3);
    WriteSum4<float, float>((float*)inout_shared, r79, r37, r10, r60);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r40 * r40;
    r40 = fmaf(r48, r40, r3 * r40);
    WriteSum3<float, float>((float*)inout_shared, r60, r28, r40);
  };
  FlushSumShared<3, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
}

void OpencvSplitFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar