#include "kernel_opencv_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvResJacFirstKernel(float* pose,
                            unsigned int pose_num_alloc,
                            SharedIndex* pose_indices,
                            float* sensor_from_rig,
                            unsigned int sensor_from_rig_num_alloc,
                            float* calib,
                            unsigned int calib_num_alloc,
                            SharedIndex* calib_indices,
                            float* point,
                            unsigned int point_num_alloc,
                            SharedIndex* point_indices,
                            float* pixel,
                            unsigned int pixel_num_alloc,
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
    r7 = fmaf(r10, r28, r7);
    r31 = fmaf(r18, r19, r15 * r22);
    r32 = r16 * r21;
    r31 = fmaf(r6, r32, r31);
    r31 = fmaf(r17, r20, r31);
    r32 = 2.00000000000000000e+00;
    r33 = r32 * r29;
    r34 = r31 * r33;
    r35 = fmaf(r16, r20, r15 * r19);
    r35 = fmaf(r17, r21, r35);
    r35 = fmaf(r6, r35, r18 * r22);
    r36 = r14 * r35;
    r37 = fmaf(r25, r36, r34);
    r38 = r25 * r32;
    r38 = r38 * r31;
    r39 = fmaf(r35, r33, r38);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r40,
                       r41,
                       r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r19 * r21;
    r43 = r43 * r32;
    r44 = r20 * r22;
    r44 = fmaf(r32, r44, r43);
    r45 = r21 * r22;
    r46 = r19 * r20;
    r46 = r46 * r32;
    r45 = fmaf(r14, r45, r46);
    r47 = r21 * r21;
    r47 = r14 * r47;
    r48 = r20 * r20;
    r48 = fmaf(r14, r48, r13);
    r49 = r47 + r48;
    r7 = fmaf(r11, r37, r7);
    r7 = fmaf(r12, r39, r7);
    r7 = fmaf(r42, r44, r7);
    r7 = fmaf(r41, r45, r7);
    r7 = fmaf(r40, r49, r7);
    r50 = r25 * r32;
    r50 = fmaf(r35, r50, r34);
    r8 = fmaf(r10, r50, r8);
    r34 = r21 * r22;
    r34 = fmaf(r32, r34, r46);
    r47 = r13 + r47;
    r46 = r19 * r19;
    r46 = r46 * r14;
    r47 = r47 + r46;
    r51 = r20 * r21;
    r51 = r51 * r32;
    r52 = r19 * r22;
    r52 = fmaf(r14, r52, r51);
    r53 = r25 * r33;
    r54 = fmaf(r31, r36, r53);
    r27 = r13 + r27;
    r55 = r14 * r31;
    r55 = r55 * r31;
    r27 = r27 + r55;
    r8 = fmaf(r40, r34, r8);
    r8 = fmaf(r41, r47, r8);
    r8 = fmaf(r42, r52, r8);
    r8 = fmaf(r12, r54, r8);
    r8 = fmaf(r11, r27, r8);
    r56 = fmaf(r8, r8, r7 * r7);
    r57 = sqrtf(r56);
    r57 = r2 + r57;
    r2 = 1.0 / r57;
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r58,
                       r59,
                       r60,
                       r61);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r62 = r58 * r7;
    r63 = r2 * r62;
    r64 = -9.99999999999999955e-07;
    r38 = fmaf(r29, r36, r38);
    r9 = fmaf(r10, r38, r9);
    r65 = r20 * r22;
    r65 = fmaf(r14, r65, r43);
    r48 = r46 + r48;
    r46 = r19 * r22;
    r46 = fmaf(r32, r46, r51);
    r51 = r32 * r31;
    r51 = fmaf(r35, r51, r53);
    r30 = r13 + r30;
    r30 = r30 + r55;
    r9 = fmaf(r40, r65, r9);
    r9 = fmaf(r42, r48, r9);
    r9 = fmaf(r41, r46, r9);
    r9 = fmaf(r11, r51, r9);
    r9 = fmaf(r12, r30, r9);
    r41 = fmaf(r9, r9, r56);
    r42 = rsqrtf(r41);
    r40 = r9 * r42;
    r55 = copysign(1.0, r40);
    r55 = fmaf(r64, r55, r40);
    r64 = acosf(r55);
    r40 = r64 * r64;
    r53 = r40 * r40;
    r43 = r64 * r53;
    r66 = fmaf(r61, r43, r64);
    r67 = r64 * r40;
    r68 = r67 * r67;
    r69 = r64 * r68;
    r70 = r53 * r53;
    r64 = r64 * r70;
    r66 = fmaf(r60, r67, r66);
    r66 = fmaf(r0, r69, r66);
    r66 = fmaf(r1, r64, r66);
    r4 = fmaf(r66, r63, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r66 * r2;
    r71 = r59 * r8;
    r5 = fmaf(r71, r3, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r3 = fmaf(r4, r4, r5 * r5);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    r3 = r58 * r66;
    r72 = r32 * r35;
    r73 = 5.00000000000000000e-01;
    r74 = fmaf(r73, r24, r73 * r23);
    r75 = -5.00000000000000000e-01;
    r76 = r18 * r73;
    r74 = fmaf(r75, r26, r74);
    r74 = fmaf(r21, r76, r74);
    r77 = r15 * r22;
    r78 = r18 * r19;
    r78 = fmaf(r75, r78, r75 * r77);
    r77 = r17 * r20;
    r78 = fmaf(r75, r77, r78);
    r79 = r16 * r21;
    r78 = fmaf(r73, r79, r78);
    r72 = fmaf(r78, r33, r74 * r72);
    r79 = r32 * r31;
    r77 = r16 * r22;
    r80 = r17 * r19;
    r80 = fmaf(r73, r80, r75 * r77);
    r77 = r18 * r20;
    r80 = fmaf(r75, r77, r80);
    r81 = r15 * r21;
    r80 = fmaf(r75, r81, r80);
    r81 = r25 * r32;
    r77 = r15 * r19;
    r82 = r16 * r20;
    r82 = fmaf(r75, r82, r75 * r77);
    r77 = r17 * r21;
    r82 = fmaf(r75, r77, r82);
    r82 = fmaf(r22, r76, r82);
    r81 = r81 * r82;
    r79 = fmaf(r80, r79, r81);
    r72 = r72 + r79;
    r77 = r32 * r31;
    r77 = r77 * r74;
    r83 = r14 * r25;
    r83 = fmaf(r78, r83, r77);
    r84 = r82 * r33;
    r83 = r83 + r84;
    r83 = fmaf(r80, r36, r83);
    r83 = fmaf(r11, r83, r12 * r72);
    r72 = r29 * r74;
    r85 = -4.00000000000000000e+00;
    r72 = r72 * r85;
    r86 = r25 * r80;
    r87 = r85 * r86;
    r88 = r72 + r87;
    r83 = fmaf(r10, r88, r83);
    r3 = r3 * r83;
    r88 = r32 * r7;
    r89 = r32 * r8;
    r90 = r14 * r31;
    r91 = r82 * r36;
    r90 = fmaf(r78, r90, r91);
    r92 = r25 * r32;
    r93 = r80 * r33;
    r92 = fmaf(r74, r92, r93);
    r90 = r90 + r92;
    r94 = r31 * r85;
    r95 = r82 * r94;
    r87 = r87 + r95;
    r87 = fmaf(r11, r87, r12 * r90);
    r90 = r32 * r35;
    r90 = fmaf(r80, r90, r77);
    r77 = r25 * r32;
    r77 = fmaf(r78, r77, r84);
    r90 = r90 + r77;
    r87 = fmaf(r10, r90, r87);
    r89 = fmaf(r87, r89, r83 * r88);
    r56 = rsqrtf(r56);
    r56 = r75 * r56;
    r57 = r57 * r57;
    r57 = 1.0 / r57;
    r88 = r66 * r57;
    r56 = r56 * r88;
    r83 = r89 * r56;
    r3 = fmaf(r62, r83, r2 * r3);
    r90 = -5.00000000000000000e+00;
    r90 = r61 * r90;
    r55 = r55 * r55;
    r55 = fmaf(r6, r55, r13);
    r55 = rsqrtf(r55);
    r90 = r90 * r55;
    r90 = r90 * r53;
    r53 = r9 * r75;
    r61 = r41 * r41;
    r61 = r41 * r61;
    r61 = rsqrtf(r61);
    r53 = r53 * r61;
    r61 = r32 * r9;
    r41 = r32 * r31;
    r41 = r41 * r78;
    r84 = r32 * r35;
    r84 = r84 * r82;
    r96 = r41 + r84;
    r92 = r92 + r96;
    r97 = r14 * r29;
    r74 = fmaf(r74, r36, r78 * r97);
    r74 = r74 + r79;
    r74 = fmaf(r10, r74, r11 * r92);
    r95 = r72 + r95;
    r74 = fmaf(r12, r95, r74);
    r61 = fmaf(r74, r61, r89);
    r74 = fmaf(r74, r42, r61 * r53);
    r61 = -9.00000000000000000e+00;
    r61 = r1 * r61;
    r61 = r61 * r55;
    r61 = r61 * r70;
    r1 = fmaf(r74, r61, r74 * r90);
    r89 = -3.00000000000000000e+00;
    r89 = r60 * r89;
    r89 = r89 * r55;
    r89 = r89 * r40;
    r40 = -7.00000000000000000e+00;
    r40 = r0 * r40;
    r40 = r40 * r55;
    r40 = r40 * r68;
    r0 = r6 * r74;
    r1 = fmaf(r55, r0, r1);
    r1 = fmaf(r74, r89, r1);
    r1 = fmaf(r74, r40, r1);
    r3 = fmaf(r1, r63, r3);
    r0 = r59 * r66;
    r0 = r0 * r87;
    r83 = fmaf(r71, r83, r2 * r0);
    r0 = r1 * r2;
    r83 = fmaf(r71, r0, r83);
    r0 = r14 * r29;
    r0 = fmaf(r80, r0, r91);
    r87 = r25 * r32;
    r60 = r18 * r21;
    r23 = fmaf(r75, r23, r75 * r60);
    r23 = fmaf(r75, r24, r23);
    r23 = fmaf(r73, r26, r23);
    r87 = r87 * r23;
    r26 = r32 * r31;
    r24 = r15 * r22;
    r60 = r17 * r20;
    r60 = fmaf(r73, r60, r73 * r24);
    r24 = r16 * r21;
    r60 = fmaf(r75, r24, r60);
    r60 = fmaf(r19, r76, r60);
    r26 = fmaf(r60, r26, r87);
    r0 = r0 + r26;
    r24 = r32 * r35;
    r95 = r60 * r33;
    r24 = fmaf(r23, r24, r95);
    r24 = r24 + r79;
    r24 = fmaf(r11, r24, r10 * r0);
    r0 = r29 * r82;
    r0 = r0 * r85;
    r79 = r23 * r94;
    r72 = r0 + r79;
    r24 = fmaf(r12, r72, r24);
    r72 = r32 * r9;
    r92 = r32 * r8;
    r97 = r14 * r31;
    r97 = fmaf(r80, r97, r81);
    r97 = r97 + r95;
    r97 = fmaf(r23, r36, r97);
    r95 = r32 * r35;
    r95 = fmaf(r32, r86, r60 * r95);
    r81 = r32 * r31;
    r81 = r81 * r82;
    r80 = fmaf(r23, r33, r81);
    r95 = r95 + r80;
    r95 = fmaf(r10, r95, r12 * r97);
    r97 = r25 * r85;
    r97 = r97 * r60;
    r79 = r97 + r79;
    r95 = fmaf(r11, r79, r95);
    r79 = r32 * r7;
    r93 = r84 + r93;
    r93 = r93 + r26;
    r97 = r0 + r97;
    r97 = fmaf(r10, r97, r12 * r93);
    r60 = fmaf(r60, r36, r14 * r86);
    r60 = r60 + r80;
    r97 = fmaf(r11, r60, r97);
    r79 = fmaf(r97, r79, r95 * r92);
    r72 = fmaf(r24, r72, r79);
    r72 = fmaf(r72, r53, r24 * r42);
    r24 = r6 * r72;
    r24 = fmaf(r72, r40, r55 * r24);
    r24 = fmaf(r72, r90, r24);
    r24 = fmaf(r72, r89, r24);
    r24 = fmaf(r72, r61, r24);
    r92 = r58 * r66;
    r92 = r92 * r97;
    r92 = fmaf(r2, r92, r24 * r63);
    r97 = r79 * r62;
    r92 = fmaf(r56, r97, r92);
    r97 = r59 * r66;
    r97 = r97 * r95;
    r95 = r24 * r2;
    r95 = fmaf(r71, r95, r2 * r97);
    r97 = r79 * r71;
    r95 = fmaf(r56, r97, r95);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r3,
                                          r83,
                                          r92,
                                          r95);
    r97 = r58 * r66;
    r60 = r14 * r25;
    r60 = fmaf(r23, r60, r41);
    r41 = r16 * r22;
    r86 = r17 * r19;
    r86 = fmaf(r75, r86, r73 * r41);
    r41 = r15 * r21;
    r86 = fmaf(r73, r41, r86);
    r86 = fmaf(r20, r76, r86);
    r33 = r86 * r33;
    r60 = r60 + r91;
    r60 = r60 + r33;
    r82 = r25 * r82;
    r82 = r82 * r85;
    r91 = r29 * r78;
    r91 = r91 * r85;
    r85 = r82 + r91;
    r85 = fmaf(r10, r85, r11 * r60);
    r60 = r25 * r32;
    r60 = r60 * r86;
    r76 = r32 * r35;
    r76 = fmaf(r78, r76, r60);
    r76 = r76 + r80;
    r85 = fmaf(r12, r76, r85);
    r97 = r97 * r85;
    r94 = r86 * r94;
    r91 = r91 + r94;
    r60 = r81 + r60;
    r81 = r14 * r29;
    r60 = fmaf(r23, r81, r60);
    r60 = fmaf(r78, r36, r60);
    r60 = fmaf(r10, r60, r12 * r91);
    r91 = r32 * r31;
    r78 = r32 * r35;
    r78 = fmaf(r86, r78, r23 * r91);
    r78 = r78 + r77;
    r60 = fmaf(r11, r78, r60);
    r78 = r32 * r9;
    r91 = r32 * r8;
    r33 = r87 + r33;
    r33 = r33 + r96;
    r96 = r14 * r31;
    r36 = fmaf(r86, r36, r23 * r96);
    r36 = r36 + r77;
    r36 = fmaf(r12, r36, r10 * r33);
    r94 = r82 + r94;
    r36 = fmaf(r11, r94, r36);
    r94 = r32 * r7;
    r94 = fmaf(r85, r94, r36 * r91);
    r78 = fmaf(r60, r78, r94);
    r78 = fmaf(r78, r53, r60 * r42);
    r60 = r6 * r78;
    r60 = fmaf(r78, r40, r55 * r60);
    r60 = fmaf(r78, r89, r60);
    r60 = fmaf(r78, r90, r60);
    r60 = fmaf(r78, r61, r60);
    r97 = fmaf(r60, r63, r2 * r97);
    r91 = r94 * r62;
    r97 = fmaf(r56, r91, r97);
    r91 = r94 * r71;
    r85 = r60 * r2;
    r85 = fmaf(r71, r85, r56 * r91);
    r91 = r59 * r66;
    r91 = r91 * r36;
    r85 = fmaf(r2, r91, r85);
    r91 = r32 * r49;
    r36 = r32 * r34;
    r36 = fmaf(r8, r36, r7 * r91);
    r91 = r36 * r62;
    r11 = r58 * r49;
    r11 = r11 * r66;
    r11 = fmaf(r2, r11, r56 * r91);
    r91 = r32 * r65;
    r91 = fmaf(r9, r91, r36);
    r91 = fmaf(r91, r53, r65 * r42);
    r82 = fmaf(r91, r90, r91 * r40);
    r12 = r6 * r91;
    r82 = fmaf(r55, r12, r82);
    r82 = fmaf(r91, r89, r82);
    r82 = fmaf(r91, r61, r82);
    r11 = fmaf(r82, r63, r11);
    r12 = r82 * r2;
    r33 = r59 * r34;
    r33 = r33 * r66;
    r33 = fmaf(r2, r33, r71 * r12);
    r12 = r36 * r71;
    r33 = fmaf(r56, r12, r33);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r97,
                                          r85,
                                          r11,
                                          r33);
    r12 = r58 * r45;
    r12 = r12 * r66;
    r10 = r32 * r45;
    r77 = r32 * r47;
    r77 = fmaf(r8, r77, r7 * r10);
    r10 = r77 * r62;
    r10 = fmaf(r56, r10, r2 * r12);
    r12 = r32 * r46;
    r12 = fmaf(r9, r12, r77);
    r12 = fmaf(r46, r42, r12 * r53);
    r86 = r6 * r12;
    r86 = fmaf(r12, r40, r55 * r86);
    r86 = fmaf(r12, r90, r86);
    r86 = fmaf(r12, r89, r86);
    r86 = fmaf(r12, r61, r86);
    r10 = fmaf(r86, r63, r10);
    r96 = r59 * r47;
    r96 = r96 * r66;
    r23 = r77 * r71;
    r23 = fmaf(r56, r23, r2 * r96);
    r96 = r86 * r2;
    r23 = fmaf(r71, r96, r23);
    r96 = r32 * r44;
    r87 = r32 * r52;
    r87 = fmaf(r8, r87, r7 * r96);
    r96 = r87 * r62;
    r81 = r58 * r44;
    r81 = r81 * r66;
    r81 = fmaf(r2, r81, r56 * r96);
    r96 = r32 * r48;
    r96 = fmaf(r9, r96, r87);
    r96 = fmaf(r48, r42, r96 * r53);
    r76 = fmaf(r96, r89, r96 * r40);
    r80 = r6 * r96;
    r76 = fmaf(r55, r80, r76);
    r76 = fmaf(r96, r90, r76);
    r76 = fmaf(r96, r61, r76);
    r81 = fmaf(r76, r63, r81);
    r80 = r76 * r2;
    r41 = r59 * r52;
    r41 = r41 * r66;
    r41 = fmaf(r2, r41, r71 * r80);
    r80 = r87 * r71;
    r41 = fmaf(r56, r80, r41);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r10,
                                          r23,
                                          r81,
                                          r41);
    r80 = r6 * r5;
    r4 = r6 * r4;
    r80 = fmaf(r3, r4, r83 * r80);
    r73 = r6 * r5;
    r73 = fmaf(r92, r4, r95 * r73);
    r75 = r6 * r5;
    r75 = fmaf(r97, r4, r85 * r75);
    r93 = r6 * r5;
    r93 = fmaf(r11, r4, r33 * r93);
    WriteSum4<float, float>((float*)inout_shared, r80, r73, r75, r93);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r93 = r6 * r5;
    r93 = fmaf(r10, r4, r23 * r93);
    r75 = r6 * r5;
    r75 = fmaf(r81, r4, r41 * r75);
    WriteSum2<float, float>((float*)inout_shared, r93, r75);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r3, r3, r83 * r83);
    r93 = fmaf(r92, r92, r95 * r95);
    r73 = fmaf(r97, r97, r85 * r85);
    r80 = fmaf(r33, r33, r11 * r11);
    WriteSum4<float, float>((float*)inout_shared, r75, r93, r73, r80);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fmaf(r10, r10, r23 * r23);
    r73 = fmaf(r81, r81, r41 * r41);
    WriteSum2<float, float>((float*)inout_shared, r80, r73);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r83, r95, r3 * r92);
    r80 = fmaf(r83, r85, r3 * r97);
    r93 = fmaf(r3, r11, r83 * r33);
    r75 = fmaf(r83, r23, r3 * r10);
    WriteSum4<float, float>((float*)inout_shared, r73, r80, r93, r75);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fmaf(r83, r41, r3 * r81);
    r3 = fmaf(r95, r85, r92 * r97);
    r75 = fmaf(r95, r33, r92 * r11);
    r93 = fmaf(r95, r23, r92 * r10);
    WriteSum4<float, float>((float*)inout_shared, r83, r3, r75, r93);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = fmaf(r95, r41, r92 * r81);
    r92 = fmaf(r97, r11, r85 * r33);
    r93 = fmaf(r85, r23, r97 * r10);
    r97 = fmaf(r97, r81, r85 * r41);
    WriteSum4<float, float>((float*)inout_shared, r95, r92, r93, r97);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r97 = fmaf(r11, r10, r33 * r23);
    r33 = fmaf(r33, r41, r11 * r81);
    r81 = fmaf(r10, r81, r23 * r41);
    WriteSum3<float, float>((float*)inout_shared, r97, r33, r81);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r7 * r66;
    r81 = r81 * r2;
    r33 = r8 * r66;
    r33 = r33 * r2;
    r97 = r67 * r63;
    r10 = r2 * r67;
    r10 = r10 * r71;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r81,
                                          r33,
                                          r97,
                                          r10);
    r41 = r43 * r63;
    r23 = r2 * r43;
    r23 = r23 * r71;
    r11 = r69 * r63;
    r93 = r2 * r69;
    r93 = r93 * r71;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r41,
                                          r23,
                                          r11,
                                          r93);
    r92 = r64 * r63;
    r95 = r2 * r71;
    r95 = r95 * r64;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r92,
                                          r95);
    r85 = r7 * r66;
    r85 = r85 * r2;
    r85 = r85 * r4;
    r75 = r6 * r8;
    r75 = r75 * r66;
    r75 = r75 * r5;
    r75 = r75 * r2;
    r3 = r6 * r5;
    r3 = r3 * r2;
    r3 = r3 * r67;
    r83 = r63 * r4;
    r3 = fmaf(r67, r83, r71 * r3);
    r80 = r6 * r5;
    r80 = r80 * r2;
    r80 = r80 * r43;
    r80 = fmaf(r43, r83, r71 * r80);
    WriteSum4<float, float>((float*)inout_shared, r85, r75, r3, r80);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r6 * r5;
    r3 = r6 * r5;
    r3 = r3 * r2;
    r3 = r3 * r69;
    r3 = fmaf(r69, r83, r71 * r3);
    r75 = r6 * r5;
    r75 = r75 * r2;
    r75 = r75 * r71;
    r83 = fmaf(r64, r83, r64 * r75);
    WriteSum4<float, float>((float*)inout_shared, r3, r83, r4, r80);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r7 * r7;
    r80 = r80 * r66;
    r80 = r80 * r88;
    r83 = r8 * r8;
    r83 = r83 * r66;
    r83 = r83 * r88;
    r3 = r57 * r62;
    r3 = r3 * r62;
    r57 = r59 * r57;
    r75 = r8 * r71;
    r57 = r57 * r75;
    r85 = fmaf(r68, r57, r68 * r3);
    r73 = r43 * r43;
    r73 = fmaf(r73, r57, r73 * r3);
    WriteSum4<float, float>((float*)inout_shared, r80, r83, r85, r73);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r69 * r69;
    r85 = fmaf(r85, r57, r85 * r3);
    r83 = r64 * r64;
    r83 = fmaf(r57, r83, r3 * r83);
    WriteSum4<float, float>((float*)inout_shared, r85, r83, r13, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = 0.00000000000000000e+00;
    r83 = r7 * r67;
    r83 = r83 * r62;
    r83 = r83 * r88;
    r80 = r7 * r43;
    r80 = r80 * r62;
    r80 = r80 * r88;
    r0 = r7 * r69;
    r0 = r0 * r62;
    r0 = r0 * r88;
    WriteSum4<float, float>((float*)inout_shared, r13, r83, r80, r0);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r7 * r62;
    r0 = r0 * r88;
    r0 = r0 * r64;
    r67 = r67 * r88;
    r67 = r67 * r75;
    WriteSum4<float, float>((float*)inout_shared, r0, r81, r13, r67);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r43 * r88;
    r43 = r43 * r75;
    r69 = r69 * r88;
    r69 = r69 * r75;
    r64 = r88 * r64;
    r64 = r64 * r75;
    WriteSum4<float, float>((float*)inout_shared, r43, r69, r64, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r70, r57, r70 * r3);
    r68 = r68 * r68;
    r68 = fmaf(r68, r57, r68 * r3);
    WriteSum4<float, float>((float*)inout_shared, r33, r64, r73, r68);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r97, r10, r68, r85);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = r70 * r70;
    r70 = fmaf(r57, r70, r3 * r70);
    WriteSum4<float, float>((float*)inout_shared, r41, r23, r70, r11);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r93, r92, r95, r13);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r32 * r38;
    r95 = r32 * r28;
    r92 = r32 * r50;
    r92 = fmaf(r8, r92, r7 * r95);
    r13 = fmaf(r9, r13, r92);
    r13 = fmaf(r13, r53, r38 * r42);
    r95 = r6 * r13;
    r95 = fmaf(r13, r61, r55 * r95);
    r95 = fmaf(r13, r89, r95);
    r95 = fmaf(r13, r90, r95);
    r95 = fmaf(r13, r40, r95);
    r93 = r58 * r28;
    r93 = r93 * r66;
    r93 = fmaf(r2, r93, r95 * r63);
    r11 = r92 * r62;
    r93 = fmaf(r56, r11, r93);
    r11 = r59 * r50;
    r11 = r11 * r66;
    r70 = r95 * r2;
    r70 = fmaf(r71, r70, r2 * r11);
    r11 = r92 * r71;
    r70 = fmaf(r56, r11, r70);
    r11 = r58 * r37;
    r11 = r11 * r66;
    r23 = r32 * r27;
    r41 = r32 * r37;
    r41 = fmaf(r7, r41, r8 * r23);
    r23 = r41 * r62;
    r23 = fmaf(r56, r23, r2 * r11);
    r11 = r32 * r51;
    r11 = fmaf(r9, r11, r41);
    r11 = fmaf(r51, r42, r11 * r53);
    r57 = r6 * r11;
    r57 = fmaf(r11, r89, r55 * r57);
    r57 = fmaf(r11, r90, r57);
    r57 = fmaf(r11, r40, r57);
    r57 = fmaf(r11, r61, r57);
    r23 = fmaf(r57, r63, r23);
    r3 = r41 * r71;
    r85 = r57 * r2;
    r85 = fmaf(r71, r85, r56 * r3);
    r3 = r59 * r27;
    r3 = r3 * r66;
    r85 = fmaf(r2, r3, r85);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r93,
                                          r70,
                                          r23,
                                          r85);
    r3 = r32 * r39;
    r68 = r32 * r54;
    r68 = fmaf(r8, r68, r7 * r3);
    r3 = r68 * r62;
    r10 = r58 * r39;
    r10 = r10 * r66;
    r10 = fmaf(r2, r10, r56 * r3);
    r3 = r32 * r30;
    r3 = fmaf(r9, r3, r68);
    r53 = fmaf(r3, r53, r30 * r42);
    r3 = r6 * r53;
    r90 = fmaf(r53, r90, r55 * r3);
    r90 = fmaf(r53, r89, r90);
    r90 = fmaf(r53, r40, r90);
    r90 = fmaf(r53, r61, r90);
    r10 = fmaf(r90, r63, r10);
    r63 = r59 * r54;
    r63 = r63 * r66;
    r61 = r90 * r2;
    r61 = fmaf(r71, r61, r2 * r63);
    r63 = r68 * r71;
    r61 = fmaf(r56, r63, r61);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r10,
                                          r61);
    r63 = r6 * r5;
    r63 = fmaf(r93, r4, r70 * r63);
    r56 = r6 * r5;
    r56 = fmaf(r23, r4, r85 * r56);
    r40 = r6 * r5;
    r4 = fmaf(r10, r4, r61 * r40);
    WriteSum3<float, float>((float*)inout_shared, r63, r56, r4);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r93, r93, r70 * r70);
    r56 = fmaf(r23, r23, r85 * r85);
    r63 = fmaf(r61, r61, r10 * r10);
    WriteSum3<float, float>((float*)inout_shared, r4, r56, r63);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r93, r23, r70 * r85);
    r70 = fmaf(r70, r61, r93 * r10);
    r10 = fmaf(r23, r10, r85 * r61);
    WriteSum3<float, float>((float*)inout_shared, r63, r70, r10);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvResJacFirst(float* pose,
                       unsigned int pose_num_alloc,
                       SharedIndex* pose_indices,
                       float* sensor_from_rig,
                       unsigned int sensor_from_rig_num_alloc,
                       float* calib,
                       unsigned int calib_num_alloc,
                       SharedIndex* calib_indices,
                       float* point,
                       unsigned int point_num_alloc,
                       SharedIndex* point_indices,
                       float* pixel,
                       unsigned int pixel_num_alloc,
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
  OpencvResJacFirstKernel<<<n_blocks, 1024>>>(pose,
                                              pose_num_alloc,
                                              pose_indices,
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