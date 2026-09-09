#include "kernel_opencv_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60;

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
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r12,
                                         r13,
                                         r14);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r15, r16, r17, r18);
    r19 = fmaf(r11, r16, r14 * r17);
    r20 = r12 * r15;
    r19 = fmaf(r4, r20, r19);
    r19 = fmaf(r13, r18, r19);
    r20 = 2.00000000000000000e+00;
    r21 = r19 * r20;
    r22 = fmaf(r11, r18, r14 * r15);
    r23 = r13 * r16;
    r22 = fmaf(r4, r23, r22);
    r22 = fmaf(r12, r17, r22);
    r21 = r21 * r22;
    r23 = r11 * r17;
    r23 = fmaf(r4, r23, r14 * r16);
    r23 = fmaf(r12, r18, r23);
    r23 = fmaf(r13, r15, r23);
    r24 = -2.00000000000000000e+00;
    r25 = fmaf(r12, r16, r11 * r15);
    r25 = fmaf(r13, r17, r25);
    r25 = fmaf(r4, r25, r14 * r18);
    r18 = r24 * r25;
    r26 = fmaf(r23, r18, r21);
    r7 = fmaf(r8, r26, r7);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r27, r28, r29);
    r30 = r11 * r13;
    r30 = r30 * r20;
    r31 = r12 * r14;
    r32 = fmaf(r24, r31, r30);
    r33 = r11 * r11;
    r33 = r24 * r33;
    r34 = 1.00000000000000000e+00;
    r35 = r12 * r12;
    r35 = fmaf(r24, r35, r34);
    r36 = r33 + r35;
    r37 = r12 * r13;
    r37 = r37 * r20;
    r38 = r11 * r14;
    r38 = fmaf(r20, r38, r37);
    r39 = r20 * r22;
    r40 = r20 * r23;
    r41 = r19 * r40;
    r39 = fmaf(r25, r39, r41);
    r42 = r24 * r23;
    r42 = r42 * r23;
    r23 = r34 + r42;
    r43 = r22 * r22;
    r43 = r24 * r43;
    r23 = r23 + r43;
    r7 = fmaf(r27, r32, r7);
    r7 = fmaf(r29, r36, r7);
    r7 = fmaf(r28, r38, r7);
    r7 = fmaf(r9, r39, r7);
    r7 = fmaf(r10, r23, r7);
    r38 = r19 * r19;
    r38 = r24 * r38;
    r36 = r34 + r38;
    r36 = r36 + r42;
    r5 = fmaf(r8, r36, r5);
    r42 = r22 * r40;
    r32 = fmaf(r19, r18, r42);
    r40 = fmaf(r25, r40, r21);
    r31 = fmaf(r20, r31, r30);
    r30 = r13 * r14;
    r21 = r11 * r12;
    r21 = r21 * r20;
    r30 = fmaf(r24, r30, r21);
    r44 = r13 * r13;
    r44 = r24 * r44;
    r35 = r44 + r35;
    r5 = fmaf(r9, r32, r5);
    r5 = fmaf(r10, r40, r5);
    r5 = fmaf(r29, r31, r5);
    r5 = fmaf(r28, r30, r5);
    r5 = fmaf(r27, r35, r5);
    r35 = r19 * r20;
    r35 = fmaf(r25, r35, r42);
    r8 = fmaf(r8, r35, r6);
    r6 = r13 * r14;
    r6 = fmaf(r20, r6, r21);
    r44 = r34 + r44;
    r44 = r44 + r33;
    r33 = r11 * r14;
    r33 = fmaf(r24, r33, r37);
    r18 = fmaf(r22, r18, r41);
    r38 = r34 + r38;
    r38 = r38 + r43;
    r8 = fmaf(r27, r6, r8);
    r8 = fmaf(r28, r44, r8);
    r8 = fmaf(r29, r33, r8);
    r8 = fmaf(r10, r18, r8);
    r8 = fmaf(r9, r38, r8);
    r9 = fmaf(r8, r8, r5 * r5);
    r10 = fmaf(r7, r7, r9);
    r33 = rsqrtf(r10);
    r29 = r7 * r33;
    r44 = copysign(1.0, r29);
    r44 = fmaf(r0, r44, r29);
    r0 = acosf(r44);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r29,
                                         r28);
    r6 = r28 * r0;
    r27 = r0 * r0;
    r43 = r27 * r27;
    r41 = r43 * r43;
    r6 = fmaf(r41, r6, r0);
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r41,
                                         r37,
                                         r24,
                                         r21);
    r42 = r0 * r27;
    r25 = r29 * r0;
    r30 = r0 * r27;
    r30 = r30 * r30;
    r6 = fmaf(r30, r25, r6);
    r30 = r21 * r0;
    r6 = fmaf(r43, r30, r6);
    r6 = fmaf(r24, r42, r6);
    r30 = 9.99999999999999955e-07;
    r25 = sqrtf(r9);
    r25 = r30 + r25;
    r30 = 1.0 / r25;
    r42 = r6 * r30;
    r43 = r41 * r5;
    r2 = fmaf(r42, r43, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r37 * r8;
    r3 = fmaf(r42, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r31 = fmaf(r2, r2, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r31);
  if (global_thread_idx < problem_size) {
    r31 = r4 * r2;
    r45 = r30 * r43;
    r46 = -5.00000000000000000e-01;
    r47 = r7 * r46;
    r48 = r10 * r10;
    r48 = r10 * r48;
    r48 = rsqrtf(r48);
    r47 = r47 * r48;
    r48 = r20 * r26;
    r10 = r20 * r36;
    r49 = r20 * r35;
    r49 = fmaf(r8, r49, r5 * r10);
    r48 = fmaf(r7, r48, r49);
    r48 = fmaf(r48, r47, r26 * r33);
    r10 = r4 * r48;
    r44 = r44 * r44;
    r44 = fmaf(r4, r44, r34);
    r44 = rsqrtf(r44);
    r34 = -3.00000000000000000e+00;
    r34 = r24 * r34;
    r34 = r34 * r44;
    r34 = r34 * r27;
    r10 = fmaf(r48, r34, r44 * r10);
    r24 = -7.00000000000000000e+00;
    r24 = r29 * r24;
    r50 = r27 * r27;
    r50 = r44 * r50;
    r51 = r50 * r27;
    r24 = r24 * r51;
    r52 = -9.00000000000000000e+00;
    r52 = r28 * r52;
    r52 = r52 * r51;
    r52 = r52 * r27;
    r27 = -5.00000000000000000e+00;
    r27 = r21 * r27;
    r27 = r27 * r50;
    r10 = fmaf(r48, r24, r10);
    r10 = fmaf(r48, r52, r10);
    r10 = fmaf(r48, r27, r10);
    r50 = r49 * r43;
    r6 = r46 * r6;
    r9 = rsqrtf(r9);
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r6 = r6 * r9;
    r6 = r6 * r25;
    r50 = fmaf(r6, r50, r10 * r45);
    r25 = r41 * r36;
    r50 = fmaf(r42, r25, r50);
    r3 = r4 * r3;
    r25 = r1 * r6;
    r9 = r10 * r30;
    r9 = fmaf(r1, r9, r49 * r25);
    r46 = r37 * r35;
    r9 = fmaf(r42, r46, r9);
    r31 = fmaf(r9, r3, r50 * r31);
    r46 = r4 * r2;
    r51 = r20 * r38;
    r53 = r20 * r32;
    r53 = fmaf(r5, r53, r8 * r51);
    r51 = r53 * r43;
    r54 = r41 * r32;
    r54 = fmaf(r42, r54, r6 * r51);
    r51 = r20 * r39;
    r51 = fmaf(r7, r51, r53);
    r51 = fmaf(r39, r33, r51 * r47);
    r55 = r4 * r51;
    r55 = fmaf(r51, r52, r44 * r55);
    r55 = fmaf(r51, r24, r55);
    r55 = fmaf(r51, r34, r55);
    r55 = fmaf(r51, r27, r55);
    r54 = fmaf(r55, r45, r54);
    r56 = r37 * r38;
    r56 = fmaf(r42, r56, r53 * r25);
    r57 = r55 * r30;
    r56 = fmaf(r1, r57, r56);
    r46 = fmaf(r56, r3, r54 * r46);
    r57 = r4 * r2;
    r58 = r20 * r23;
    r59 = r20 * r40;
    r60 = r20 * r18;
    r60 = fmaf(r8, r60, r5 * r59);
    r58 = fmaf(r7, r58, r60);
    r47 = fmaf(r58, r47, r23 * r33);
    r58 = r4 * r47;
    r24 = fmaf(r47, r24, r44 * r58);
    r24 = fmaf(r47, r27, r24);
    r24 = fmaf(r47, r52, r24);
    r24 = fmaf(r47, r34, r24);
    r34 = r41 * r40;
    r34 = fmaf(r42, r34, r24 * r45);
    r45 = r60 * r43;
    r34 = fmaf(r6, r45, r34);
    r45 = r24 * r30;
    r45 = fmaf(r1, r45, r60 * r25);
    r25 = r37 * r18;
    r45 = fmaf(r42, r25, r45);
    r3 = fmaf(r45, r3, r34 * r57);
    WriteSum3<float, float>((float*)inout_shared, r31, r46, r3);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r50, r50, r9 * r9);
    r46 = fmaf(r54, r54, r56 * r56);
    r31 = fmaf(r45, r45, r34 * r34);
    WriteSum3<float, float>((float*)inout_shared, r3, r46, r31);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r9, r56, r50 * r54);
    r50 = fmaf(r50, r34, r9 * r45);
    r45 = fmaf(r56, r45, r54 * r34);
    WriteSum3<float, float>((float*)inout_shared, r31, r50, r45);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  OpencvSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_point_njtr,
              out_point_njtr_num_alloc,
              out_point_precond_diag,
              out_point_precond_diag_num_alloc,
              out_point_precond_tril,
              out_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar