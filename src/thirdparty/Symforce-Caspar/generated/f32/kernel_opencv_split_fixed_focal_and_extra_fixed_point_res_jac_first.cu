#include "kernel_opencv_split_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* point,
        unsigned int point_num_alloc,
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
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89;
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = -9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = r13 * r18;
    r20 = r12 * r15;
    r21 = r19 + r20;
    r22 = r11 * r16;
    r21 = fmaf(r14, r17, r21);
    r21 = fmaf(r4, r22, r21);
    r23 = 2.00000000000000000e+00;
    r24 = r21 * r23;
    r25 = fmaf(r14, r15, r11 * r18);
    r26 = r12 * r17;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r13, r16, r25);
    r24 = r24 * r25;
    r26 = r13 * r15;
    r26 = fmaf(r4, r26, r12 * r18);
    r26 = fmaf(r14, r16, r26);
    r26 = fmaf(r11, r17, r26);
    r27 = -2.00000000000000000e+00;
    r28 = fmaf(r12, r16, r11 * r15);
    r28 = fmaf(r13, r17, r28);
    r28 = fmaf(r4, r28, r14 * r18);
    r29 = r27 * r28;
    r30 = fmaf(r26, r29, r24);
    r30 = fmaf(r8, r30, r7);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r7,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r15 * r17;
    r33 = r33 * r23;
    r34 = r16 * r18;
    r34 = fmaf(r27, r34, r33);
    r35 = r15 * r15;
    r35 = r35 * r27;
    r36 = 1.00000000000000000e+00;
    r37 = r16 * r16;
    r37 = fmaf(r27, r37, r36);
    r38 = r35 + r37;
    r39 = r16 * r17;
    r39 = r39 * r23;
    r40 = r15 * r18;
    r40 = fmaf(r23, r40, r39);
    r41 = r23 * r25;
    r42 = r23 * r26;
    r43 = r21 * r42;
    r41 = fmaf(r28, r41, r43);
    r44 = r27 * r26;
    r44 = r44 * r26;
    r45 = r36 + r44;
    r46 = r27 * r25;
    r46 = r46 * r25;
    r45 = r45 + r46;
    r30 = fmaf(r7, r34, r30);
    r30 = fmaf(r32, r38, r30);
    r30 = fmaf(r31, r40, r30);
    r30 = fmaf(r9, r41, r30);
    r30 = fmaf(r10, r45, r30);
    r45 = r27 * r21;
    r45 = r45 * r21;
    r41 = r36 + r45;
    r41 = r41 + r44;
    r41 = fmaf(r8, r41, r5);
    r5 = r25 * r42;
    r44 = fmaf(r21, r29, r5);
    r24 = fmaf(r28, r42, r24);
    r47 = r16 * r18;
    r47 = fmaf(r23, r47, r33);
    r33 = r17 * r18;
    r48 = r15 * r16;
    r48 = r48 * r23;
    r33 = fmaf(r27, r33, r48);
    r49 = r17 * r17;
    r49 = r27 * r49;
    r37 = r49 + r37;
    r41 = fmaf(r9, r44, r41);
    r41 = fmaf(r10, r24, r41);
    r41 = fmaf(r32, r47, r41);
    r41 = fmaf(r31, r33, r41);
    r41 = fmaf(r7, r37, r41);
    r24 = r21 * r23;
    r24 = fmaf(r28, r24, r5);
    r24 = fmaf(r8, r24, r6);
    r6 = r17 * r18;
    r6 = fmaf(r23, r6, r48);
    r49 = r36 + r49;
    r49 = r49 + r35;
    r35 = r15 * r18;
    r35 = fmaf(r27, r35, r39);
    r43 = fmaf(r25, r29, r43);
    r45 = r36 + r45;
    r45 = r45 + r46;
    r24 = fmaf(r7, r6, r24);
    r24 = fmaf(r31, r49, r24);
    r24 = fmaf(r32, r35, r24);
    r24 = fmaf(r10, r43, r24);
    r24 = fmaf(r9, r45, r24);
    r45 = fmaf(r24, r24, r41 * r41);
    r43 = fmaf(r30, r30, r45);
    r32 = rsqrtf(r43);
    r31 = r30 * r32;
    r7 = copysign(1.0, r31);
    r7 = fmaf(r0, r7, r31);
    r0 = acosf(r7);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r31,
                                         r46);
    r39 = r46 * r0;
    r48 = r0 * r0;
    r5 = r48 * r48;
    r44 = r5 * r5;
    r39 = fmaf(r44, r39, r0);
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r44,
                                         r50,
                                         r51,
                                         r52);
    r53 = r0 * r48;
    r54 = r31 * r0;
    r55 = r48 * r48;
    r54 = r54 * r55;
    r39 = fmaf(r48, r54, r39);
    r56 = r52 * r0;
    r39 = fmaf(r5, r56, r39);
    r39 = fmaf(r51, r53, r39);
    r56 = 9.99999999999999955e-07;
    r54 = sqrtf(r45);
    r54 = r56 + r54;
    r56 = 1.0 / r54;
    r53 = r39 * r56;
    r5 = r44 * r41;
    r2 = fmaf(r53, r5, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r50 * r24;
    r3 = fmaf(r53, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r57 = fmaf(r2, r2, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r57);
  if (global_thread_idx < problem_size) {
    r57 = r23 * r28;
    r58 = 5.00000000000000000e-01;
    r59 = fmaf(r58, r20, r58 * r19);
    r60 = -5.00000000000000000e-01;
    r61 = r14 * r58;
    r59 = fmaf(r60, r22, r59);
    r59 = fmaf(r17, r61, r59);
    r62 = r11 * r18;
    r63 = r14 * r15;
    r63 = fmaf(r60, r63, r60 * r62);
    r62 = r13 * r16;
    r63 = fmaf(r60, r62, r63);
    r64 = r12 * r17;
    r63 = fmaf(r58, r64, r63);
    r57 = fmaf(r63, r42, r59 * r57);
    r64 = r23 * r25;
    r62 = r12 * r18;
    r65 = r13 * r15;
    r65 = fmaf(r58, r65, r60 * r62);
    r62 = r14 * r16;
    r65 = fmaf(r60, r62, r65);
    r66 = r11 * r17;
    r65 = fmaf(r60, r66, r65);
    r66 = r21 * r23;
    r62 = r11 * r15;
    r67 = r12 * r16;
    r67 = fmaf(r60, r67, r60 * r62);
    r62 = r13 * r17;
    r67 = fmaf(r60, r62, r67);
    r67 = fmaf(r18, r61, r67);
    r66 = r66 * r67;
    r64 = fmaf(r65, r64, r66);
    r57 = r57 + r64;
    r62 = r23 * r25;
    r62 = r62 * r59;
    r68 = r27 * r21;
    r68 = fmaf(r63, r68, r62);
    r69 = r67 * r42;
    r68 = r68 + r69;
    r68 = fmaf(r65, r29, r68);
    r68 = fmaf(r9, r68, r10 * r57);
    r57 = r26 * r59;
    r70 = -4.00000000000000000e+00;
    r57 = r57 * r70;
    r71 = r21 * r65;
    r72 = r70 * r71;
    r73 = r57 + r72;
    r68 = fmaf(r8, r73, r68);
    r73 = r44 * r68;
    r74 = r23 * r41;
    r75 = r23 * r24;
    r76 = r27 * r25;
    r77 = r67 * r29;
    r76 = fmaf(r63, r76, r77);
    r78 = r21 * r23;
    r79 = r65 * r42;
    r78 = fmaf(r59, r78, r79);
    r76 = r76 + r78;
    r80 = r25 * r70;
    r81 = r67 * r80;
    r72 = r72 + r81;
    r72 = fmaf(r9, r72, r10 * r76);
    r76 = r23 * r28;
    r76 = fmaf(r65, r76, r62);
    r62 = r21 * r23;
    r62 = fmaf(r63, r62, r69);
    r76 = r76 + r62;
    r72 = fmaf(r8, r76, r72);
    r75 = fmaf(r72, r75, r68 * r74);
    r74 = r75 * r5;
    r39 = r60 * r39;
    r45 = rsqrtf(r45);
    r54 = r54 * r54;
    r54 = 1.0 / r54;
    r39 = r39 * r45;
    r39 = r39 * r54;
    r74 = fmaf(r39, r74, r53 * r73);
    r73 = r56 * r5;
    r54 = r30 * r60;
    r45 = r43 * r43;
    r45 = r43 * r45;
    r45 = rsqrtf(r45);
    r54 = r54 * r45;
    r45 = r23 * r30;
    r43 = r23 * r25;
    r43 = r43 * r63;
    r76 = r23 * r28;
    r76 = r76 * r67;
    r69 = r43 + r76;
    r78 = r78 + r69;
    r82 = r27 * r26;
    r59 = fmaf(r59, r29, r63 * r82);
    r59 = r59 + r64;
    r59 = fmaf(r8, r59, r9 * r78);
    r81 = r57 + r81;
    r59 = fmaf(r10, r81, r59);
    r45 = fmaf(r59, r45, r75);
    r59 = fmaf(r59, r32, r45 * r54);
    r45 = r4 * r59;
    r7 = r7 * r7;
    r7 = fmaf(r4, r7, r36);
    r7 = rsqrtf(r7);
    r81 = -7.00000000000000000e+00;
    r81 = r31 * r81;
    r55 = r7 * r55;
    r57 = r55 * r48;
    r81 = r81 * r57;
    r45 = fmaf(r59, r81, r7 * r45);
    r78 = -3.00000000000000000e+00;
    r78 = r51 * r78;
    r78 = r78 * r7;
    r78 = r78 * r48;
    r51 = -9.00000000000000000e+00;
    r51 = r46 * r51;
    r51 = r51 * r57;
    r51 = r51 * r48;
    r48 = -5.00000000000000000e+00;
    r48 = r52 * r48;
    r48 = r48 * r55;
    r45 = fmaf(r59, r78, r45);
    r45 = fmaf(r59, r51, r45);
    r45 = fmaf(r59, r48, r45);
    r74 = fmaf(r45, r73, r74);
    r55 = r50 * r72;
    r57 = r45 * r56;
    r57 = fmaf(r1, r57, r53 * r55);
    r55 = r1 * r39;
    r57 = fmaf(r75, r55, r57);
    r82 = r23 * r24;
    r83 = r27 * r25;
    r83 = fmaf(r65, r83, r66);
    r66 = r11 * r18;
    r84 = r13 * r16;
    r84 = fmaf(r58, r84, r58 * r66);
    r66 = r12 * r17;
    r84 = fmaf(r60, r66, r84);
    r84 = fmaf(r15, r61, r84);
    r66 = r84 * r42;
    r85 = r14 * r17;
    r19 = fmaf(r60, r19, r60 * r85);
    r19 = fmaf(r60, r20, r19);
    r19 = fmaf(r58, r22, r19);
    r83 = r83 + r66;
    r83 = fmaf(r19, r29, r83);
    r22 = r23 * r28;
    r22 = fmaf(r23, r71, r84 * r22);
    r20 = r23 * r25;
    r20 = r20 * r67;
    r85 = fmaf(r19, r42, r20);
    r22 = r22 + r85;
    r22 = fmaf(r8, r22, r10 * r83);
    r83 = r21 * r70;
    r83 = r83 * r84;
    r86 = r19 * r80;
    r87 = r83 + r86;
    r22 = fmaf(r9, r87, r22);
    r87 = r23 * r41;
    r79 = r76 + r79;
    r76 = r21 * r23;
    r76 = r76 * r19;
    r88 = r23 * r25;
    r88 = fmaf(r84, r88, r76);
    r79 = r79 + r88;
    r89 = r26 * r67;
    r89 = r89 * r70;
    r83 = r83 + r89;
    r83 = fmaf(r8, r83, r10 * r79);
    r84 = fmaf(r84, r29, r27 * r71);
    r84 = r84 + r85;
    r83 = fmaf(r9, r84, r83);
    r87 = fmaf(r83, r87, r22 * r82);
    r82 = r87 * r5;
    r84 = r27 * r26;
    r84 = fmaf(r65, r84, r77);
    r84 = r84 + r88;
    r88 = r23 * r28;
    r88 = fmaf(r19, r88, r66);
    r88 = r88 + r64;
    r88 = fmaf(r9, r88, r8 * r84);
    r86 = r89 + r86;
    r88 = fmaf(r10, r86, r88);
    r86 = r23 * r30;
    r86 = fmaf(r88, r86, r87);
    r86 = fmaf(r86, r54, r88 * r32);
    r88 = fmaf(r86, r81, r86 * r51);
    r89 = r4 * r86;
    r88 = fmaf(r7, r89, r88);
    r88 = fmaf(r86, r48, r88);
    r88 = fmaf(r86, r78, r88);
    r82 = fmaf(r88, r73, r39 * r82);
    r89 = r44 * r83;
    r82 = fmaf(r53, r89, r82);
    r89 = r88 * r56;
    r89 = fmaf(r1, r89, r87 * r55);
    r84 = r50 * r22;
    r89 = fmaf(r53, r84, r89);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r74,
                                          r57,
                                          r82,
                                          r89);
    r84 = r26 * r63;
    r84 = r84 * r70;
    r64 = r12 * r18;
    r66 = r13 * r15;
    r66 = fmaf(r60, r66, r58 * r64);
    r64 = r11 * r17;
    r66 = fmaf(r58, r64, r66);
    r66 = fmaf(r16, r61, r66);
    r80 = r66 * r80;
    r61 = r84 + r80;
    r64 = r21 * r23;
    r64 = r64 * r66;
    r20 = r20 + r64;
    r58 = r27 * r26;
    r20 = fmaf(r19, r58, r20);
    r20 = fmaf(r63, r29, r20);
    r20 = fmaf(r8, r20, r10 * r61);
    r61 = r23 * r25;
    r58 = r23 * r28;
    r58 = fmaf(r66, r58, r19 * r61);
    r58 = r58 + r62;
    r20 = fmaf(r9, r58, r20);
    r58 = r23 * r30;
    r61 = r23 * r24;
    r42 = r66 * r42;
    r76 = r76 + r42;
    r76 = r76 + r69;
    r69 = r27 * r25;
    r29 = fmaf(r66, r29, r19 * r69);
    r29 = r29 + r62;
    r29 = fmaf(r10, r29, r8 * r76);
    r67 = r21 * r67;
    r67 = r67 * r70;
    r80 = r67 + r80;
    r29 = fmaf(r9, r80, r29);
    r80 = r23 * r41;
    r70 = r27 * r21;
    r70 = fmaf(r19, r70, r43);
    r70 = r70 + r77;
    r70 = r70 + r42;
    r67 = r84 + r67;
    r67 = fmaf(r8, r67, r9 * r70);
    r8 = r23 * r28;
    r8 = fmaf(r63, r8, r64);
    r8 = r8 + r85;
    r67 = fmaf(r10, r8, r67);
    r80 = fmaf(r67, r80, r29 * r61);
    r58 = fmaf(r20, r58, r80);
    r58 = fmaf(r58, r54, r20 * r32);
    r20 = r4 * r58;
    r20 = fmaf(r58, r78, r7 * r20);
    r20 = fmaf(r58, r81, r20);
    r20 = fmaf(r58, r48, r20);
    r20 = fmaf(r58, r51, r20);
    r61 = r80 * r5;
    r61 = fmaf(r39, r61, r20 * r73);
    r8 = r44 * r67;
    r61 = fmaf(r53, r8, r61);
    r8 = r20 * r56;
    r8 = fmaf(r1, r8, r80 * r55);
    r10 = r50 * r29;
    r8 = fmaf(r53, r10, r8);
    r10 = r23 * r37;
    r85 = r23 * r6;
    r85 = fmaf(r24, r85, r41 * r10);
    r10 = r85 * r5;
    r64 = r44 * r37;
    r64 = fmaf(r53, r64, r39 * r10);
    r10 = r23 * r34;
    r10 = fmaf(r30, r10, r85);
    r10 = fmaf(r10, r54, r34 * r32);
    r63 = r4 * r10;
    r63 = fmaf(r10, r81, r7 * r63);
    r63 = fmaf(r10, r51, r63);
    r63 = fmaf(r10, r48, r63);
    r63 = fmaf(r10, r78, r63);
    r64 = fmaf(r63, r73, r64);
    r70 = r50 * r6;
    r70 = fmaf(r53, r70, r85 * r55);
    r9 = r63 * r56;
    r70 = fmaf(r1, r9, r70);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r61,
                                          r8,
                                          r64,
                                          r70);
    r9 = r23 * r33;
    r84 = r23 * r49;
    r84 = fmaf(r24, r84, r41 * r9);
    r9 = r84 * r5;
    r42 = r23 * r40;
    r42 = fmaf(r30, r42, r84);
    r42 = fmaf(r40, r32, r42 * r54);
    r77 = r4 * r42;
    r77 = fmaf(r42, r51, r7 * r77);
    r77 = fmaf(r42, r81, r77);
    r77 = fmaf(r42, r78, r77);
    r77 = fmaf(r42, r48, r77);
    r9 = fmaf(r77, r73, r39 * r9);
    r43 = r44 * r33;
    r9 = fmaf(r53, r43, r9);
    r43 = r77 * r56;
    r43 = fmaf(r84, r55, r1 * r43);
    r19 = r50 * r49;
    r43 = fmaf(r53, r19, r43);
    r19 = r44 * r47;
    r76 = r23 * r47;
    r62 = r23 * r35;
    r62 = fmaf(r24, r62, r41 * r76);
    r76 = r62 * r5;
    r76 = fmaf(r39, r76, r53 * r19);
    r19 = r23 * r38;
    r19 = fmaf(r30, r19, r62);
    r32 = fmaf(r38, r32, r19 * r54);
    r81 = fmaf(r32, r81, r32 * r48);
    r48 = r4 * r32;
    r81 = fmaf(r7, r48, r81);
    r81 = fmaf(r32, r78, r81);
    r81 = fmaf(r32, r51, r81);
    r76 = fmaf(r81, r73, r76);
    r73 = r81 * r56;
    r55 = fmaf(r62, r55, r1 * r73);
    r73 = r50 * r35;
    r55 = fmaf(r53, r73, r55);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r43,
                                          r76,
                                          r55);
    r73 = r4 * r2;
    r3 = r4 * r3;
    r73 = fmaf(r57, r3, r74 * r73);
    r53 = r4 * r2;
    r53 = fmaf(r89, r3, r82 * r53);
    r1 = r4 * r2;
    r1 = fmaf(r8, r3, r61 * r1);
    r48 = r4 * r2;
    r48 = fmaf(r70, r3, r64 * r48);
    WriteSum4<float, float>((float*)inout_shared, r73, r53, r1, r48);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r4 * r2;
    r48 = fmaf(r43, r3, r9 * r48);
    r1 = r4 * r2;
    r1 = fmaf(r55, r3, r76 * r1);
    WriteSum2<float, float>((float*)inout_shared, r48, r1);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r57, r57, r74 * r74);
    r48 = fmaf(r82, r82, r89 * r89);
    r53 = fmaf(r8, r8, r61 * r61);
    r73 = fmaf(r70, r70, r64 * r64);
    WriteSum4<float, float>((float*)inout_shared, r1, r48, r53, r73);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r43, r43, r9 * r9);
    r53 = fmaf(r55, r55, r76 * r76);
    WriteSum2<float, float>((float*)inout_shared, r73, r53);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fmaf(r74, r82, r57 * r89);
    r73 = fmaf(r57, r8, r74 * r61);
    r48 = fmaf(r57, r70, r74 * r64);
    r1 = fmaf(r57, r43, r74 * r9);
    WriteSum4<float, float>((float*)inout_shared, r53, r73, r48, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fmaf(r57, r55, r74 * r76);
    r74 = fmaf(r89, r8, r82 * r61);
    r1 = fmaf(r82, r64, r89 * r70);
    r48 = fmaf(r89, r43, r82 * r9);
    WriteSum4<float, float>((float*)inout_shared, r57, r74, r1, r48);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fmaf(r82, r76, r89 * r55);
    r89 = fmaf(r8, r70, r61 * r64);
    r48 = fmaf(r61, r9, r8 * r43);
    r8 = fmaf(r8, r55, r61 * r76);
    WriteSum4<float, float>((float*)inout_shared, r82, r89, r48, r8);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r64, r9, r70 * r43);
    r64 = fmaf(r64, r76, r70 * r55);
    r76 = fmaf(r9, r76, r43 * r55);
    WriteSum3<float, float>((float*)inout_shared, r8, r64, r76);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r76, r3);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r36, r36);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedFocalAndExtraFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* point,
    unsigned int point_num_alloc,
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
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedFocalAndExtraFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      point,
      point_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar