#include "kernel_opencv_split_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
      r91, r92, r93, r94, r95, r96, r97;

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
    r0 = -9.99999999999999955e-07;
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
    r7 = fmaf(r8, r30, r7);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r31,
                       r32,
                       r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r15 * r17;
    r34 = r34 * r23;
    r35 = r16 * r18;
    r35 = fmaf(r27, r35, r34);
    r36 = r15 * r15;
    r36 = r36 * r27;
    r37 = 1.00000000000000000e+00;
    r38 = r16 * r16;
    r38 = fmaf(r27, r38, r37);
    r39 = r36 + r38;
    r40 = r16 * r17;
    r40 = r40 * r23;
    r41 = r15 * r18;
    r41 = fmaf(r23, r41, r40);
    r42 = r23 * r25;
    r43 = r23 * r26;
    r44 = r21 * r43;
    r42 = fmaf(r28, r42, r44);
    r45 = r27 * r26;
    r45 = r45 * r26;
    r46 = r37 + r45;
    r47 = r27 * r25;
    r47 = r47 * r25;
    r46 = r46 + r47;
    r7 = fmaf(r31, r35, r7);
    r7 = fmaf(r33, r39, r7);
    r7 = fmaf(r32, r41, r7);
    r7 = fmaf(r9, r42, r7);
    r7 = fmaf(r10, r46, r7);
    r48 = r27 * r21;
    r48 = r48 * r21;
    r49 = r37 + r48;
    r49 = r49 + r45;
    r5 = fmaf(r8, r49, r5);
    r45 = r25 * r43;
    r50 = fmaf(r21, r29, r45);
    r24 = fmaf(r28, r43, r24);
    r51 = r16 * r18;
    r51 = fmaf(r23, r51, r34);
    r34 = r17 * r18;
    r52 = r15 * r16;
    r52 = r52 * r23;
    r34 = fmaf(r27, r34, r52);
    r53 = r17 * r17;
    r53 = r27 * r53;
    r38 = r53 + r38;
    r5 = fmaf(r9, r50, r5);
    r5 = fmaf(r10, r24, r5);
    r5 = fmaf(r33, r51, r5);
    r5 = fmaf(r32, r34, r5);
    r5 = fmaf(r31, r38, r5);
    r54 = r21 * r23;
    r54 = fmaf(r28, r54, r45);
    r6 = fmaf(r8, r54, r6);
    r45 = r17 * r18;
    r45 = fmaf(r23, r45, r52);
    r53 = r37 + r53;
    r53 = r53 + r36;
    r36 = r15 * r18;
    r36 = fmaf(r27, r36, r40);
    r44 = fmaf(r25, r29, r44);
    r48 = r37 + r48;
    r48 = r48 + r47;
    r6 = fmaf(r31, r45, r6);
    r6 = fmaf(r32, r53, r6);
    r6 = fmaf(r33, r36, r6);
    r6 = fmaf(r10, r44, r6);
    r6 = fmaf(r9, r48, r6);
    r33 = fmaf(r6, r6, r5 * r5);
    r32 = fmaf(r7, r7, r33);
    r31 = rsqrtf(r32);
    r47 = r7 * r31;
    r40 = copysign(1.0, r47);
    r40 = fmaf(r0, r40, r47);
    r0 = acosf(r40);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r47,
                                         r52);
    r55 = r52 * r0;
    r56 = r0 * r0;
    r57 = r56 * r56;
    r58 = r57 * r57;
    r55 = fmaf(r58, r55, r0);
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r58,
                                         r59,
                                         r60,
                                         r61);
    r62 = r60 * r0;
    r55 = fmaf(r56, r62, r55);
    r63 = r0 * r56;
    r63 = r63 * r63;
    r63 = r0 * r63;
    r64 = r61 * r0;
    r55 = fmaf(r57, r64, r55);
    r55 = fmaf(r47, r63, r55);
    r64 = 9.99999999999999955e-07;
    r63 = sqrtf(r33);
    r63 = r64 + r63;
    r64 = 1.0 / r63;
    r62 = r55 * r64;
    r57 = r58 * r5;
    r2 = fmaf(r62, r57, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r59 * r6;
    r3 = fmaf(r62, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r65 = fmaf(r2, r2, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r65);
  if (global_thread_idx < problem_size) {
    r65 = r23 * r28;
    r66 = 5.00000000000000000e-01;
    r67 = fmaf(r66, r20, r66 * r19);
    r68 = -5.00000000000000000e-01;
    r69 = r14 * r66;
    r67 = fmaf(r68, r22, r67);
    r67 = fmaf(r17, r69, r67);
    r70 = r11 * r18;
    r71 = r14 * r15;
    r71 = fmaf(r68, r71, r68 * r70);
    r70 = r13 * r16;
    r71 = fmaf(r68, r70, r71);
    r72 = r12 * r17;
    r71 = fmaf(r66, r72, r71);
    r65 = fmaf(r71, r43, r67 * r65);
    r72 = r23 * r25;
    r70 = r12 * r18;
    r73 = r13 * r15;
    r73 = fmaf(r66, r73, r68 * r70);
    r70 = r14 * r16;
    r73 = fmaf(r68, r70, r73);
    r74 = r11 * r17;
    r73 = fmaf(r68, r74, r73);
    r74 = r21 * r23;
    r70 = r11 * r15;
    r75 = r12 * r16;
    r75 = fmaf(r68, r75, r68 * r70);
    r70 = r13 * r17;
    r75 = fmaf(r68, r70, r75);
    r75 = fmaf(r18, r69, r75);
    r74 = r74 * r75;
    r72 = fmaf(r73, r72, r74);
    r65 = r65 + r72;
    r70 = r23 * r25;
    r70 = r70 * r67;
    r76 = r27 * r21;
    r76 = fmaf(r71, r76, r70);
    r77 = r75 * r43;
    r76 = r76 + r77;
    r76 = fmaf(r73, r29, r76);
    r76 = fmaf(r9, r76, r10 * r65);
    r65 = r26 * r67;
    r78 = -4.00000000000000000e+00;
    r65 = r65 * r78;
    r79 = r21 * r73;
    r80 = r78 * r79;
    r81 = r65 + r80;
    r76 = fmaf(r8, r81, r76);
    r81 = r58 * r76;
    r82 = r23 * r5;
    r83 = r23 * r6;
    r84 = r27 * r25;
    r85 = r75 * r29;
    r84 = fmaf(r71, r84, r85);
    r86 = r21 * r23;
    r87 = r73 * r43;
    r86 = fmaf(r67, r86, r87);
    r84 = r84 + r86;
    r88 = r25 * r78;
    r89 = r75 * r88;
    r80 = r80 + r89;
    r80 = fmaf(r9, r80, r10 * r84);
    r84 = r23 * r28;
    r84 = fmaf(r73, r84, r70);
    r70 = r21 * r23;
    r70 = fmaf(r71, r70, r77);
    r84 = r84 + r70;
    r80 = fmaf(r8, r84, r80);
    r83 = fmaf(r80, r83, r76 * r82);
    r82 = r83 * r57;
    r55 = r68 * r55;
    r33 = rsqrtf(r33);
    r63 = r63 * r63;
    r63 = 1.0 / r63;
    r55 = r55 * r33;
    r55 = r55 * r63;
    r82 = fmaf(r55, r82, r62 * r81);
    r81 = r64 * r57;
    r63 = r7 * r68;
    r33 = r32 * r32;
    r33 = r32 * r33;
    r33 = rsqrtf(r33);
    r63 = r63 * r33;
    r33 = r23 * r7;
    r32 = r23 * r25;
    r32 = r32 * r71;
    r84 = r23 * r28;
    r84 = r84 * r75;
    r77 = r32 + r84;
    r86 = r86 + r77;
    r90 = r27 * r26;
    r67 = fmaf(r67, r29, r71 * r90);
    r67 = r67 + r72;
    r67 = fmaf(r8, r67, r9 * r86);
    r89 = r65 + r89;
    r67 = fmaf(r10, r89, r67);
    r33 = fmaf(r67, r33, r83);
    r67 = fmaf(r67, r31, r33 * r63);
    r33 = r4 * r67;
    r40 = r40 * r40;
    r40 = fmaf(r4, r40, r37);
    r40 = rsqrtf(r40);
    r37 = -7.00000000000000000e+00;
    r37 = r47 * r37;
    r47 = r40 * r56;
    r89 = r47 * r56;
    r65 = r89 * r56;
    r37 = r37 * r65;
    r33 = fmaf(r67, r37, r40 * r33);
    r86 = -3.00000000000000000e+00;
    r86 = r60 * r86;
    r86 = r86 * r47;
    r47 = -9.00000000000000000e+00;
    r47 = r52 * r47;
    r47 = r47 * r65;
    r47 = r47 * r56;
    r56 = -5.00000000000000000e+00;
    r56 = r61 * r56;
    r56 = r56 * r89;
    r33 = fmaf(r67, r86, r33);
    r33 = fmaf(r67, r47, r33);
    r33 = fmaf(r67, r56, r33);
    r82 = fmaf(r33, r81, r82);
    r89 = r59 * r80;
    r65 = r33 * r64;
    r65 = fmaf(r1, r65, r62 * r89);
    r89 = r1 * r55;
    r65 = fmaf(r83, r89, r65);
    r90 = r23 * r6;
    r91 = r27 * r25;
    r91 = fmaf(r73, r91, r74);
    r74 = r11 * r18;
    r92 = r13 * r16;
    r92 = fmaf(r66, r92, r66 * r74);
    r74 = r12 * r17;
    r92 = fmaf(r68, r74, r92);
    r92 = fmaf(r15, r69, r92);
    r74 = r92 * r43;
    r93 = r14 * r17;
    r19 = fmaf(r68, r19, r68 * r93);
    r19 = fmaf(r68, r20, r19);
    r19 = fmaf(r66, r22, r19);
    r91 = r91 + r74;
    r91 = fmaf(r19, r29, r91);
    r22 = r23 * r28;
    r22 = fmaf(r23, r79, r92 * r22);
    r20 = r23 * r25;
    r20 = r20 * r75;
    r93 = fmaf(r19, r43, r20);
    r22 = r22 + r93;
    r22 = fmaf(r8, r22, r10 * r91);
    r91 = r21 * r78;
    r91 = r91 * r92;
    r94 = r19 * r88;
    r95 = r91 + r94;
    r22 = fmaf(r9, r95, r22);
    r95 = r23 * r5;
    r87 = r84 + r87;
    r84 = r21 * r23;
    r84 = r84 * r19;
    r96 = r23 * r25;
    r96 = fmaf(r92, r96, r84);
    r87 = r87 + r96;
    r97 = r26 * r75;
    r97 = r97 * r78;
    r91 = r91 + r97;
    r91 = fmaf(r8, r91, r10 * r87);
    r92 = fmaf(r92, r29, r27 * r79);
    r92 = r92 + r93;
    r91 = fmaf(r9, r92, r91);
    r95 = fmaf(r91, r95, r22 * r90);
    r90 = r95 * r57;
    r92 = r27 * r26;
    r92 = fmaf(r73, r92, r85);
    r92 = r92 + r96;
    r96 = r23 * r28;
    r96 = fmaf(r19, r96, r74);
    r96 = r96 + r72;
    r96 = fmaf(r9, r96, r8 * r92);
    r94 = r97 + r94;
    r96 = fmaf(r10, r94, r96);
    r94 = r23 * r7;
    r94 = fmaf(r96, r94, r95);
    r94 = fmaf(r94, r63, r96 * r31);
    r96 = fmaf(r94, r37, r94 * r47);
    r97 = r4 * r94;
    r96 = fmaf(r40, r97, r96);
    r96 = fmaf(r94, r56, r96);
    r96 = fmaf(r94, r86, r96);
    r90 = fmaf(r96, r81, r55 * r90);
    r97 = r58 * r91;
    r90 = fmaf(r62, r97, r90);
    r97 = r96 * r64;
    r97 = fmaf(r1, r97, r95 * r89);
    r92 = r59 * r22;
    r97 = fmaf(r62, r92, r97);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r82,
                                          r65,
                                          r90,
                                          r97);
    r92 = r26 * r71;
    r92 = r92 * r78;
    r72 = r12 * r18;
    r74 = r13 * r15;
    r74 = fmaf(r68, r74, r66 * r72);
    r72 = r11 * r17;
    r74 = fmaf(r66, r72, r74);
    r74 = fmaf(r16, r69, r74);
    r88 = r74 * r88;
    r69 = r92 + r88;
    r72 = r21 * r23;
    r72 = r72 * r74;
    r20 = r20 + r72;
    r66 = r27 * r26;
    r20 = fmaf(r19, r66, r20);
    r20 = fmaf(r71, r29, r20);
    r20 = fmaf(r8, r20, r10 * r69);
    r69 = r23 * r25;
    r66 = r23 * r28;
    r66 = fmaf(r74, r66, r19 * r69);
    r66 = r66 + r70;
    r20 = fmaf(r9, r66, r20);
    r66 = r23 * r7;
    r69 = r23 * r6;
    r43 = r74 * r43;
    r84 = r84 + r43;
    r84 = r84 + r77;
    r77 = r27 * r25;
    r29 = fmaf(r74, r29, r19 * r77);
    r29 = r29 + r70;
    r29 = fmaf(r10, r29, r8 * r84);
    r75 = r21 * r75;
    r75 = r75 * r78;
    r88 = r75 + r88;
    r29 = fmaf(r9, r88, r29);
    r88 = r23 * r5;
    r78 = r27 * r21;
    r78 = fmaf(r19, r78, r32);
    r78 = r78 + r85;
    r78 = r78 + r43;
    r75 = r92 + r75;
    r75 = fmaf(r8, r75, r9 * r78);
    r8 = r23 * r28;
    r8 = fmaf(r71, r8, r72);
    r8 = r8 + r93;
    r75 = fmaf(r10, r8, r75);
    r88 = fmaf(r75, r88, r29 * r69);
    r66 = fmaf(r20, r66, r88);
    r66 = fmaf(r66, r63, r20 * r31);
    r20 = r4 * r66;
    r20 = fmaf(r66, r86, r40 * r20);
    r20 = fmaf(r66, r37, r20);
    r20 = fmaf(r66, r56, r20);
    r20 = fmaf(r66, r47, r20);
    r69 = r88 * r57;
    r69 = fmaf(r55, r69, r20 * r81);
    r8 = r58 * r75;
    r69 = fmaf(r62, r8, r69);
    r8 = r20 * r64;
    r8 = fmaf(r1, r8, r88 * r89);
    r10 = r59 * r29;
    r8 = fmaf(r62, r10, r8);
    r10 = r23 * r38;
    r93 = r23 * r45;
    r93 = fmaf(r6, r93, r5 * r10);
    r10 = r93 * r57;
    r72 = r58 * r38;
    r72 = fmaf(r62, r72, r55 * r10);
    r10 = r23 * r35;
    r10 = fmaf(r7, r10, r93);
    r10 = fmaf(r10, r63, r35 * r31);
    r71 = r4 * r10;
    r71 = fmaf(r10, r37, r40 * r71);
    r71 = fmaf(r10, r47, r71);
    r71 = fmaf(r10, r56, r71);
    r71 = fmaf(r10, r86, r71);
    r72 = fmaf(r71, r81, r72);
    r78 = r59 * r45;
    r78 = fmaf(r62, r78, r93 * r89);
    r9 = r71 * r64;
    r78 = fmaf(r1, r9, r78);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r69,
                                          r8,
                                          r72,
                                          r78);
    r9 = r23 * r34;
    r92 = r23 * r53;
    r92 = fmaf(r6, r92, r5 * r9);
    r9 = r92 * r57;
    r43 = r23 * r41;
    r43 = fmaf(r7, r43, r92);
    r43 = fmaf(r41, r31, r43 * r63);
    r85 = r4 * r43;
    r85 = fmaf(r43, r47, r40 * r85);
    r85 = fmaf(r43, r37, r85);
    r85 = fmaf(r43, r86, r85);
    r85 = fmaf(r43, r56, r85);
    r9 = fmaf(r85, r81, r55 * r9);
    r32 = r58 * r34;
    r9 = fmaf(r62, r32, r9);
    r32 = r85 * r64;
    r32 = fmaf(r92, r89, r1 * r32);
    r19 = r59 * r53;
    r32 = fmaf(r62, r19, r32);
    r19 = r58 * r51;
    r84 = r23 * r51;
    r70 = r23 * r36;
    r70 = fmaf(r6, r70, r5 * r84);
    r84 = r70 * r57;
    r84 = fmaf(r55, r84, r62 * r19);
    r19 = r23 * r39;
    r19 = fmaf(r7, r19, r70);
    r19 = fmaf(r39, r31, r19 * r63);
    r74 = fmaf(r19, r37, r19 * r56);
    r77 = r4 * r19;
    r74 = fmaf(r40, r77, r74);
    r74 = fmaf(r19, r86, r74);
    r74 = fmaf(r19, r47, r74);
    r84 = fmaf(r74, r81, r84);
    r77 = r74 * r64;
    r77 = fmaf(r70, r89, r1 * r77);
    r68 = r59 * r36;
    r77 = fmaf(r62, r68, r77);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r32,
                                          r84,
                                          r77);
    r68 = r4 * r2;
    r3 = r4 * r3;
    r68 = fmaf(r65, r3, r82 * r68);
    r73 = r4 * r2;
    r73 = fmaf(r97, r3, r90 * r73);
    r79 = r4 * r2;
    r79 = fmaf(r8, r3, r69 * r79);
    r87 = r4 * r2;
    r87 = fmaf(r78, r3, r72 * r87);
    WriteSum4<float, float>((float*)inout_shared, r68, r73, r79, r87);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = r4 * r2;
    r87 = fmaf(r32, r3, r9 * r87);
    r79 = r4 * r2;
    r79 = fmaf(r77, r3, r84 * r79);
    WriteSum2<float, float>((float*)inout_shared, r87, r79);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fmaf(r65, r65, r82 * r82);
    r87 = fmaf(r90, r90, r97 * r97);
    r73 = fmaf(r8, r8, r69 * r69);
    r68 = fmaf(r78, r78, r72 * r72);
    WriteSum4<float, float>((float*)inout_shared, r79, r87, r73, r68);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r32, r32, r9 * r9);
    r73 = fmaf(r77, r77, r84 * r84);
    WriteSum2<float, float>((float*)inout_shared, r68, r73);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r82, r90, r65 * r97);
    r68 = fmaf(r65, r8, r82 * r69);
    r87 = fmaf(r65, r78, r82 * r72);
    r79 = fmaf(r65, r32, r82 * r9);
    WriteSum4<float, float>((float*)inout_shared, r73, r68, r87, r79);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fmaf(r65, r77, r82 * r84);
    r82 = fmaf(r97, r8, r90 * r69);
    r79 = fmaf(r90, r72, r97 * r78);
    r87 = fmaf(r97, r32, r90 * r9);
    WriteSum4<float, float>((float*)inout_shared, r65, r82, r79, r87);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = fmaf(r90, r84, r97 * r77);
    r97 = fmaf(r8, r78, r69 * r72);
    r87 = fmaf(r69, r9, r8 * r32);
    r8 = fmaf(r8, r77, r69 * r84);
    WriteSum4<float, float>((float*)inout_shared, r90, r97, r87, r8);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r72, r9, r78 * r32);
    r72 = fmaf(r72, r84, r78 * r77);
    r84 = fmaf(r9, r84, r32 * r77);
    WriteSum3<float, float>((float*)inout_shared, r8, r72, r84);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r23 * r30;
    r72 = r23 * r49;
    r8 = r23 * r54;
    r8 = fmaf(r6, r8, r5 * r72);
    r84 = fmaf(r7, r84, r8);
    r84 = fmaf(r84, r63, r30 * r31);
    r72 = r4 * r84;
    r72 = fmaf(r84, r86, r40 * r72);
    r72 = fmaf(r84, r37, r72);
    r72 = fmaf(r84, r47, r72);
    r72 = fmaf(r84, r56, r72);
    r9 = r8 * r57;
    r9 = fmaf(r55, r9, r72 * r81);
    r77 = r58 * r49;
    r9 = fmaf(r62, r77, r9);
    r77 = r72 * r64;
    r77 = fmaf(r1, r77, r8 * r89);
    r32 = r59 * r54;
    r77 = fmaf(r62, r32, r77);
    r32 = r23 * r48;
    r78 = r23 * r50;
    r78 = fmaf(r5, r78, r6 * r32);
    r32 = r78 * r57;
    r87 = r58 * r50;
    r87 = fmaf(r62, r87, r55 * r32);
    r32 = r23 * r42;
    r32 = fmaf(r7, r32, r78);
    r32 = fmaf(r42, r31, r32 * r63);
    r97 = r4 * r32;
    r97 = fmaf(r32, r47, r40 * r97);
    r97 = fmaf(r32, r37, r97);
    r97 = fmaf(r32, r86, r97);
    r97 = fmaf(r32, r56, r97);
    r87 = fmaf(r97, r81, r87);
    r90 = r59 * r48;
    r90 = fmaf(r62, r90, r78 * r89);
    r69 = r97 * r64;
    r90 = fmaf(r1, r69, r90);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r77,
                                          r87,
                                          r90);
    r69 = r23 * r46;
    r79 = r23 * r24;
    r82 = r23 * r44;
    r82 = fmaf(r6, r82, r5 * r79);
    r69 = fmaf(r7, r69, r82);
    r63 = fmaf(r69, r63, r46 * r31);
    r69 = r4 * r63;
    r37 = fmaf(r63, r37, r40 * r69);
    r37 = fmaf(r63, r56, r37);
    r37 = fmaf(r63, r47, r37);
    r37 = fmaf(r63, r86, r37);
    r86 = r58 * r24;
    r86 = fmaf(r62, r86, r37 * r81);
    r81 = r82 * r57;
    r86 = fmaf(r55, r81, r86);
    r81 = r37 * r64;
    r81 = fmaf(r1, r81, r82 * r89);
    r89 = r59 * r44;
    r81 = fmaf(r62, r89, r81);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r86,
                                          r81);
    r89 = r4 * r2;
    r89 = fmaf(r77, r3, r9 * r89);
    r62 = r4 * r2;
    r62 = fmaf(r90, r3, r87 * r62);
    r1 = r4 * r2;
    r3 = fmaf(r81, r3, r86 * r1);
    WriteSum3<float, float>((float*)inout_shared, r89, r62, r3);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r9, r9, r77 * r77);
    r62 = fmaf(r87, r87, r90 * r90);
    r89 = fmaf(r81, r81, r86 * r86);
    WriteSum3<float, float>((float*)inout_shared, r3, r62, r89);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = fmaf(r77, r90, r9 * r87);
    r9 = fmaf(r9, r86, r77 * r81);
    r81 = fmaf(r90, r81, r87 * r86);
    WriteSum3<float, float>((float*)inout_shared, r89, r9, r81);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedFocalAndExtraFixedPrincipalPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  OpencvSplitFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                                      1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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