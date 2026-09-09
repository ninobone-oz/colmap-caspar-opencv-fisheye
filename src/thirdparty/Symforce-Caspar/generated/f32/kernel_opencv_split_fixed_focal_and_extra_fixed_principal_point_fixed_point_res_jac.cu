#include "kernel_opencv_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90;

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
    r19 = r12 * r15;
    r20 = fmaf(r13, r18, r19);
    r21 = r14 * r17;
    r22 = r11 * r16;
    r20 = r20 + r21;
    r20 = fmaf(r4, r22, r20);
    r23 = 2.00000000000000000e+00;
    r24 = r20 * r23;
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
    r43 = r20 * r42;
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
    r45 = r27 * r20;
    r45 = r45 * r20;
    r41 = r36 + r45;
    r41 = r41 + r44;
    r41 = fmaf(r8, r41, r5);
    r5 = r25 * r42;
    r44 = fmaf(r20, r29, r5);
    r24 = fmaf(r28, r42, r24);
    r47 = r16 * r18;
    r47 = fmaf(r23, r47, r33);
    r33 = r17 * r18;
    r48 = r15 * r16;
    r48 = r48 * r23;
    r33 = fmaf(r27, r33, r48);
    r49 = r17 * r17;
    r49 = r49 * r27;
    r37 = r49 + r37;
    r41 = fmaf(r9, r44, r41);
    r41 = fmaf(r10, r24, r41);
    r41 = fmaf(r32, r47, r41);
    r41 = fmaf(r31, r33, r41);
    r41 = fmaf(r7, r37, r41);
    r24 = r20 * r23;
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
    r57 = r4 * r2;
    r58 = r23 * r28;
    r59 = 5.00000000000000000e-01;
    r60 = r18 * r59;
    r61 = fmaf(r59, r19, r13 * r60);
    r62 = -5.00000000000000000e-01;
    r61 = fmaf(r62, r22, r61);
    r61 = fmaf(r59, r21, r61);
    r63 = r11 * r18;
    r64 = r14 * r15;
    r64 = fmaf(r62, r64, r62 * r63);
    r63 = r13 * r16;
    r64 = fmaf(r62, r63, r64);
    r65 = r12 * r17;
    r64 = fmaf(r59, r65, r64);
    r58 = fmaf(r64, r42, r61 * r58);
    r65 = r23 * r25;
    r63 = r12 * r18;
    r66 = r13 * r15;
    r66 = fmaf(r59, r66, r62 * r63);
    r63 = r14 * r16;
    r66 = fmaf(r62, r63, r66);
    r67 = r11 * r17;
    r66 = fmaf(r62, r67, r66);
    r67 = r20 * r23;
    r63 = r11 * r15;
    r68 = r12 * r16;
    r68 = fmaf(r62, r68, r62 * r63);
    r63 = r13 * r17;
    r68 = fmaf(r62, r63, r68);
    r68 = fmaf(r14, r60, r68);
    r67 = r67 * r68;
    r65 = fmaf(r66, r65, r67);
    r58 = r58 + r65;
    r63 = r23 * r25;
    r63 = r63 * r61;
    r69 = r27 * r20;
    r69 = fmaf(r64, r69, r63);
    r70 = r68 * r42;
    r69 = r69 + r70;
    r69 = fmaf(r66, r29, r69);
    r69 = fmaf(r9, r69, r10 * r58);
    r58 = r26 * r61;
    r71 = -4.00000000000000000e+00;
    r58 = r58 * r71;
    r72 = r20 * r66;
    r73 = r71 * r72;
    r74 = r58 + r73;
    r69 = fmaf(r8, r74, r69);
    r74 = r44 * r69;
    r75 = r23 * r41;
    r76 = r23 * r24;
    r77 = r27 * r25;
    r78 = r68 * r29;
    r77 = fmaf(r64, r77, r78);
    r79 = r20 * r23;
    r80 = r66 * r42;
    r79 = fmaf(r61, r79, r80);
    r77 = r77 + r79;
    r81 = r25 * r71;
    r82 = r68 * r81;
    r73 = r82 + r73;
    r73 = fmaf(r9, r73, r10 * r77);
    r77 = r23 * r28;
    r77 = fmaf(r66, r77, r63);
    r63 = r20 * r23;
    r63 = fmaf(r64, r63, r70);
    r77 = r77 + r63;
    r73 = fmaf(r8, r77, r73);
    r76 = fmaf(r73, r76, r69 * r75);
    r75 = r76 * r5;
    r39 = r62 * r39;
    r45 = rsqrtf(r45);
    r54 = r54 * r54;
    r54 = 1.0 / r54;
    r39 = r39 * r45;
    r39 = r39 * r54;
    r75 = fmaf(r39, r75, r53 * r74);
    r74 = r56 * r5;
    r54 = r30 * r62;
    r45 = r43 * r43;
    r45 = r43 * r45;
    r45 = rsqrtf(r45);
    r54 = r54 * r45;
    r45 = r23 * r30;
    r43 = r23 * r25;
    r43 = r43 * r64;
    r77 = r23 * r28;
    r77 = r77 * r68;
    r70 = r43 + r77;
    r79 = r79 + r70;
    r83 = r27 * r26;
    r61 = fmaf(r61, r29, r64 * r83);
    r61 = r61 + r65;
    r61 = fmaf(r8, r61, r9 * r79);
    r82 = r58 + r82;
    r61 = fmaf(r10, r82, r61);
    r45 = fmaf(r61, r45, r76);
    r61 = fmaf(r61, r32, r45 * r54);
    r45 = r4 * r61;
    r7 = r7 * r7;
    r7 = fmaf(r4, r7, r36);
    r7 = rsqrtf(r7);
    r36 = -7.00000000000000000e+00;
    r36 = r31 * r36;
    r55 = r7 * r55;
    r82 = r55 * r48;
    r36 = r36 * r82;
    r45 = fmaf(r61, r36, r7 * r45);
    r58 = -3.00000000000000000e+00;
    r58 = r51 * r58;
    r58 = r58 * r7;
    r58 = r58 * r48;
    r51 = -9.00000000000000000e+00;
    r51 = r46 * r51;
    r51 = r51 * r82;
    r51 = r51 * r48;
    r48 = -5.00000000000000000e+00;
    r48 = r52 * r48;
    r48 = r48 * r55;
    r45 = fmaf(r61, r58, r45);
    r45 = fmaf(r61, r51, r45);
    r45 = fmaf(r61, r48, r45);
    r75 = fmaf(r45, r74, r75);
    r3 = r4 * r3;
    r55 = r50 * r73;
    r82 = r45 * r56;
    r82 = fmaf(r1, r82, r53 * r55);
    r55 = r1 * r39;
    r82 = fmaf(r76, r55, r82);
    r57 = fmaf(r82, r3, r75 * r57);
    r79 = r4 * r2;
    r83 = r23 * r24;
    r84 = r27 * r25;
    r84 = fmaf(r66, r84, r67);
    r67 = r14 * r15;
    r85 = r13 * r16;
    r85 = fmaf(r59, r85, r59 * r67);
    r67 = r12 * r17;
    r85 = fmaf(r62, r67, r85);
    r85 = fmaf(r11, r60, r85);
    r67 = r85 * r42;
    r86 = r13 * r18;
    r19 = fmaf(r62, r19, r62 * r86);
    r19 = fmaf(r59, r22, r19);
    r19 = fmaf(r62, r21, r19);
    r84 = r84 + r67;
    r84 = fmaf(r19, r29, r84);
    r21 = r23 * r28;
    r21 = fmaf(r23, r72, r85 * r21);
    r22 = r23 * r25;
    r22 = r22 * r68;
    r86 = fmaf(r19, r42, r22);
    r21 = r21 + r86;
    r21 = fmaf(r8, r21, r10 * r84);
    r84 = r20 * r71;
    r84 = r84 * r85;
    r87 = r19 * r81;
    r88 = r84 + r87;
    r21 = fmaf(r9, r88, r21);
    r88 = r23 * r41;
    r80 = r77 + r80;
    r77 = r20 * r23;
    r77 = r77 * r19;
    r89 = r23 * r25;
    r89 = fmaf(r85, r89, r77);
    r80 = r80 + r89;
    r90 = r26 * r68;
    r90 = r90 * r71;
    r84 = r84 + r90;
    r84 = fmaf(r8, r84, r10 * r80);
    r85 = fmaf(r85, r29, r27 * r72);
    r85 = r85 + r86;
    r84 = fmaf(r9, r85, r84);
    r88 = fmaf(r84, r88, r21 * r83);
    r83 = r88 * r5;
    r85 = r27 * r26;
    r85 = fmaf(r66, r85, r78);
    r85 = r85 + r89;
    r89 = r23 * r28;
    r89 = fmaf(r19, r89, r67);
    r89 = r89 + r65;
    r89 = fmaf(r9, r89, r8 * r85);
    r87 = r90 + r87;
    r89 = fmaf(r10, r87, r89);
    r87 = r23 * r30;
    r87 = fmaf(r89, r87, r88);
    r87 = fmaf(r87, r54, r89 * r32);
    r89 = fmaf(r87, r36, r87 * r51);
    r90 = r4 * r87;
    r89 = fmaf(r7, r90, r89);
    r89 = fmaf(r87, r48, r89);
    r89 = fmaf(r87, r58, r89);
    r83 = fmaf(r89, r74, r39 * r83);
    r90 = r44 * r84;
    r83 = fmaf(r53, r90, r83);
    r90 = r89 * r56;
    r90 = fmaf(r1, r90, r88 * r55);
    r85 = r50 * r21;
    r90 = fmaf(r53, r85, r90);
    r79 = fmaf(r90, r3, r83 * r79);
    r85 = r4 * r2;
    r65 = r26 * r64;
    r65 = r65 * r71;
    r67 = r13 * r15;
    r66 = r14 * r16;
    r66 = fmaf(r59, r66, r62 * r67);
    r67 = r11 * r17;
    r66 = fmaf(r59, r67, r66);
    r66 = fmaf(r12, r60, r66);
    r81 = r66 * r81;
    r60 = r65 + r81;
    r67 = r20 * r23;
    r67 = r67 * r66;
    r22 = r22 + r67;
    r59 = r27 * r26;
    r22 = fmaf(r19, r59, r22);
    r22 = fmaf(r64, r29, r22);
    r22 = fmaf(r8, r22, r10 * r60);
    r60 = r23 * r25;
    r59 = r23 * r28;
    r59 = fmaf(r66, r59, r19 * r60);
    r59 = r59 + r63;
    r22 = fmaf(r9, r59, r22);
    r59 = r23 * r30;
    r60 = r23 * r24;
    r42 = r66 * r42;
    r77 = r77 + r42;
    r77 = r77 + r70;
    r70 = r27 * r25;
    r29 = fmaf(r66, r29, r19 * r70);
    r29 = r29 + r63;
    r29 = fmaf(r10, r29, r8 * r77);
    r68 = r20 * r68;
    r68 = r68 * r71;
    r81 = r68 + r81;
    r29 = fmaf(r9, r81, r29);
    r81 = r23 * r41;
    r71 = r27 * r20;
    r71 = fmaf(r19, r71, r43);
    r71 = r71 + r78;
    r71 = r71 + r42;
    r68 = r65 + r68;
    r68 = fmaf(r8, r68, r9 * r71);
    r8 = r23 * r28;
    r8 = fmaf(r64, r8, r67);
    r8 = r8 + r86;
    r68 = fmaf(r10, r8, r68);
    r81 = fmaf(r68, r81, r29 * r60);
    r59 = fmaf(r22, r59, r81);
    r59 = fmaf(r59, r54, r22 * r32);
    r22 = r4 * r59;
    r22 = fmaf(r59, r58, r7 * r22);
    r22 = fmaf(r59, r36, r22);
    r22 = fmaf(r59, r48, r22);
    r22 = fmaf(r59, r51, r22);
    r60 = r81 * r5;
    r60 = fmaf(r39, r60, r22 * r74);
    r8 = r44 * r68;
    r60 = fmaf(r53, r8, r60);
    r8 = r22 * r56;
    r8 = fmaf(r1, r8, r81 * r55);
    r10 = r50 * r29;
    r8 = fmaf(r53, r10, r8);
    r85 = fmaf(r8, r3, r60 * r85);
    r10 = r4 * r2;
    r86 = r23 * r37;
    r67 = r23 * r6;
    r67 = fmaf(r24, r67, r41 * r86);
    r86 = r67 * r5;
    r64 = r44 * r37;
    r64 = fmaf(r53, r64, r39 * r86);
    r86 = r23 * r34;
    r86 = fmaf(r30, r86, r67);
    r86 = fmaf(r86, r54, r34 * r32);
    r71 = r4 * r86;
    r71 = fmaf(r86, r36, r7 * r71);
    r71 = fmaf(r86, r51, r71);
    r71 = fmaf(r86, r48, r71);
    r71 = fmaf(r86, r58, r71);
    r64 = fmaf(r71, r74, r64);
    r9 = r50 * r6;
    r9 = fmaf(r53, r9, r67 * r55);
    r65 = r71 * r56;
    r9 = fmaf(r1, r65, r9);
    r10 = fmaf(r9, r3, r64 * r10);
    WriteSum4<float, float>((float*)inout_shared, r57, r79, r85, r10);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r4 * r2;
    r85 = r23 * r33;
    r79 = r23 * r49;
    r79 = fmaf(r24, r79, r41 * r85);
    r85 = r79 * r5;
    r57 = r23 * r40;
    r57 = fmaf(r30, r57, r79);
    r57 = fmaf(r40, r32, r57 * r54);
    r65 = r4 * r57;
    r65 = fmaf(r57, r51, r7 * r65);
    r65 = fmaf(r57, r36, r65);
    r65 = fmaf(r57, r58, r65);
    r65 = fmaf(r57, r48, r65);
    r85 = fmaf(r65, r74, r39 * r85);
    r42 = r44 * r33;
    r85 = fmaf(r53, r42, r85);
    r42 = r65 * r56;
    r42 = fmaf(r79, r55, r1 * r42);
    r78 = r50 * r49;
    r42 = fmaf(r53, r78, r42);
    r10 = fmaf(r42, r3, r85 * r10);
    r78 = r4 * r2;
    r43 = r44 * r47;
    r19 = r23 * r47;
    r77 = r23 * r35;
    r77 = fmaf(r24, r77, r41 * r19);
    r19 = r77 * r5;
    r19 = fmaf(r39, r19, r53 * r43);
    r43 = r23 * r38;
    r43 = fmaf(r30, r43, r77);
    r32 = fmaf(r38, r32, r43 * r54);
    r36 = fmaf(r32, r36, r32 * r48);
    r48 = r4 * r32;
    r36 = fmaf(r7, r48, r36);
    r36 = fmaf(r32, r58, r36);
    r36 = fmaf(r32, r51, r36);
    r19 = fmaf(r36, r74, r19);
    r74 = r36 * r56;
    r55 = fmaf(r77, r55, r1 * r74);
    r74 = r50 * r35;
    r55 = fmaf(r53, r74, r55);
    r3 = fmaf(r55, r3, r19 * r78);
    WriteSum2<float, float>((float*)inout_shared, r10, r3);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r82, r82, r75 * r75);
    r10 = fmaf(r83, r83, r90 * r90);
    r78 = fmaf(r8, r8, r60 * r60);
    r74 = fmaf(r9, r9, r64 * r64);
    WriteSum4<float, float>((float*)inout_shared, r3, r10, r78, r74);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r42, r42, r85 * r85);
    r78 = fmaf(r55, r55, r19 * r19);
    WriteSum2<float, float>((float*)inout_shared, r74, r78);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fmaf(r75, r83, r82 * r90);
    r74 = fmaf(r82, r8, r75 * r60);
    r10 = fmaf(r82, r9, r75 * r64);
    r3 = fmaf(r82, r42, r75 * r85);
    WriteSum4<float, float>((float*)inout_shared, r78, r74, r10, r3);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fmaf(r82, r55, r75 * r19);
    r75 = fmaf(r90, r8, r83 * r60);
    r3 = fmaf(r83, r64, r90 * r9);
    r10 = fmaf(r90, r42, r83 * r85);
    WriteSum4<float, float>((float*)inout_shared, r82, r75, r3, r10);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fmaf(r83, r19, r90 * r55);
    r90 = fmaf(r8, r9, r60 * r64);
    r10 = fmaf(r60, r85, r8 * r42);
    r8 = fmaf(r8, r55, r60 * r19);
    WriteSum4<float, float>((float*)inout_shared, r83, r90, r10, r8);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r64, r85, r9 * r42);
    r64 = fmaf(r64, r19, r9 * r55);
    r19 = fmaf(r85, r19, r42 * r55);
    WriteSum3<float, float>((float*)inout_shared, r8, r64, r19);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
}

void OpencvSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar