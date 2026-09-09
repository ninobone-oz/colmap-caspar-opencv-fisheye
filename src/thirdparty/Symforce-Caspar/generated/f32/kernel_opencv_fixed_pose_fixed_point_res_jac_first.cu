#include "kernel_opencv_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvFixedPoseFixedPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;
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
    r2 = -9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r8,
                                         r9);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11, r12);
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
    r21 = fmaf(r6, r22, r21);
    r21 = fmaf(r15, r20, r21);
    r22 = 2.00000000000000000e+00;
    r23 = r21 * r22;
    r24 = fmaf(r13, r20, r16 * r17);
    r25 = r15 * r18;
    r24 = fmaf(r6, r25, r24);
    r24 = fmaf(r14, r19, r24);
    r23 = r23 * r24;
    r25 = r13 * r19;
    r25 = fmaf(r6, r25, r16 * r18);
    r25 = fmaf(r14, r20, r25);
    r25 = fmaf(r15, r17, r25);
    r26 = -2.00000000000000000e+00;
    r27 = fmaf(r14, r18, r13 * r17);
    r27 = fmaf(r15, r19, r27);
    r27 = fmaf(r6, r27, r16 * r20);
    r20 = r26 * r27;
    r28 = fmaf(r25, r20, r23);
    r28 = fmaf(r10, r28, r9);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r9, r29, r30);
    r31 = r13 * r15;
    r31 = r31 * r22;
    r32 = r14 * r16;
    r33 = fmaf(r26, r32, r31);
    r34 = r13 * r13;
    r34 = r26 * r34;
    r35 = 1.00000000000000000e+00;
    r36 = r14 * r14;
    r36 = fmaf(r26, r36, r35);
    r37 = r34 + r36;
    r38 = r14 * r15;
    r38 = r38 * r22;
    r39 = r13 * r16;
    r39 = fmaf(r22, r39, r38);
    r40 = r22 * r24;
    r41 = r22 * r25;
    r42 = r21 * r41;
    r40 = fmaf(r27, r40, r42);
    r43 = r26 * r25;
    r43 = r43 * r25;
    r25 = r35 + r43;
    r44 = r24 * r24;
    r44 = r26 * r44;
    r25 = r25 + r44;
    r28 = fmaf(r9, r33, r28);
    r28 = fmaf(r30, r37, r28);
    r28 = fmaf(r29, r39, r28);
    r28 = fmaf(r11, r40, r28);
    r28 = fmaf(r12, r25, r28);
    r25 = r21 * r21;
    r25 = r26 * r25;
    r40 = r35 + r25;
    r40 = r40 + r43;
    r40 = fmaf(r10, r40, r7);
    r7 = r24 * r41;
    r43 = fmaf(r21, r20, r7);
    r41 = fmaf(r27, r41, r23);
    r32 = fmaf(r22, r32, r31);
    r31 = r15 * r16;
    r23 = r13 * r14;
    r23 = r23 * r22;
    r31 = fmaf(r26, r31, r23);
    r39 = r15 * r15;
    r39 = r26 * r39;
    r36 = r39 + r36;
    r40 = fmaf(r11, r43, r40);
    r40 = fmaf(r12, r41, r40);
    r40 = fmaf(r30, r32, r40);
    r40 = fmaf(r29, r31, r40);
    r40 = fmaf(r9, r36, r40);
    r36 = r21 * r22;
    r36 = fmaf(r27, r36, r7);
    r36 = fmaf(r10, r36, r8);
    r10 = r15 * r16;
    r10 = fmaf(r22, r10, r23);
    r39 = r35 + r39;
    r39 = r39 + r34;
    r34 = r13 * r16;
    r34 = fmaf(r26, r34, r38);
    r20 = fmaf(r24, r20, r42);
    r25 = r35 + r25;
    r25 = r25 + r44;
    r36 = fmaf(r9, r10, r36);
    r36 = fmaf(r29, r39, r36);
    r36 = fmaf(r30, r34, r36);
    r36 = fmaf(r12, r20, r36);
    r36 = fmaf(r11, r25, r36);
    r25 = fmaf(r36, r36, r40 * r40);
    r11 = fmaf(r28, r28, r25);
    r11 = rsqrtf(r11);
    r11 = r28 * r11;
    r28 = copysign(1.0, r11);
    r28 = fmaf(r2, r28, r11);
    r28 = acosf(r28);
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r2,
                       r11,
                       r20,
                       r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r28 * r28;
    r30 = r34 * r34;
    r39 = r28 * r30;
    r12 = fmaf(r12, r39, r28);
    r34 = r28 * r34;
    r29 = r34 * r34;
    r10 = r28 * r29;
    r30 = r30 * r30;
    r28 = r28 * r30;
    r12 = fmaf(r20, r34, r12);
    r12 = fmaf(r0, r10, r12);
    r12 = fmaf(r1, r28, r12);
    r1 = 9.99999999999999955e-07;
    r25 = sqrtf(r25);
    r25 = r1 + r25;
    r1 = 1.0 / r25;
    r0 = r12 * r1;
    r20 = r2 * r40;
    r4 = fmaf(r20, r0, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r12 * r1;
    r0 = r11 * r36;
    r5 = fmaf(r0, r3, r5);
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
    r3 = r40 * r12;
    r9 = r6 * r1;
    r3 = r3 * r4;
    r3 = r3 * r9;
    r44 = r36 * r12;
    r42 = r5 * r9;
    r44 = r44 * r42;
    r42 = r0 * r42;
    r38 = r4 * r20;
    r38 = r38 * r9;
    r9 = fmaf(r34, r38, r34 * r42);
    r26 = fmaf(r39, r38, r39 * r42);
    WriteSum4<float, float>((float*)inout_shared, r3, r44, r9, r26);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r6 * r4;
    r5 = r6 * r5;
    r6 = fmaf(r10, r38, r10 * r42);
    r38 = fmaf(r28, r38, r28 * r42);
    WriteSum4<float, float>((float*)inout_shared, r6, r38, r4, r5);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r40 * r40;
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r4 = r12 * r12;
    r4 = r25 * r4;
    r5 = r5 * r4;
    r38 = r36 * r36;
    r38 = r38 * r4;
    r4 = r40 * r25;
    r4 = r4 * r20;
    r2 = r2 * r4;
    r25 = r36 * r25;
    r25 = r25 * r0;
    r11 = r11 * r25;
    r6 = fmaf(r29, r11, r29 * r2);
    r42 = r39 * r39;
    r42 = fmaf(r42, r11, r42 * r2);
    WriteSum4<float, float>((float*)inout_shared, r5, r38, r6, r42);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r10 * r10;
    r6 = fmaf(r6, r11, r6 * r2);
    r38 = r28 * r28;
    r38 = fmaf(r11, r38, r2 * r38);
    WriteSum4<float, float>((float*)inout_shared, r6, r38, r35, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = 0.00000000000000000e+00;
    r38 = r12 * r34;
    r38 = r38 * r4;
    r5 = r12 * r39;
    r5 = r5 * r4;
    r26 = r12 * r10;
    r26 = r26 * r4;
    WriteSum4<float, float>((float*)inout_shared, r35, r38, r5, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r12 * r28;
    r26 = r26 * r4;
    r40 = r40 * r12;
    r40 = r40 * r1;
    r4 = r12 * r34;
    r4 = r4 * r25;
    WriteSum4<float, float>((float*)inout_shared, r26, r40, r35, r4);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r12 * r39;
    r4 = r4 * r25;
    r40 = r12 * r10;
    r40 = r40 * r25;
    r26 = r12 * r28;
    r26 = r26 * r25;
    WriteSum4<float, float>((float*)inout_shared, r4, r40, r26, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r36 * r12;
    r36 = r36 * r1;
    r26 = fmaf(r30, r11, r30 * r2);
    r29 = r29 * r29;
    r29 = fmaf(r29, r11, r29 * r2);
    WriteSum4<float, float>((float*)inout_shared, r36, r26, r42, r29);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r1 * r34;
    r42 = r42 * r20;
    r34 = r1 * r34;
    r34 = r34 * r0;
    WriteSum4<float, float>((float*)inout_shared, r42, r34, r29, r6);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r1 * r39;
    r6 = r6 * r20;
    r39 = r1 * r39;
    r39 = r39 * r0;
    r29 = r1 * r10;
    r29 = r29 * r20;
    r30 = r30 * r30;
    r30 = fmaf(r11, r30, r2 * r30);
    WriteSum4<float, float>((float*)inout_shared, r6, r39, r30, r29);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r1 * r10;
    r10 = r10 * r0;
    r29 = r1 * r28;
    r29 = r29 * r20;
    r28 = r1 * r28;
    r28 = r28 * r0;
    WriteSum4<float, float>((float*)inout_shared, r10, r29, r28, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvFixedPoseFixedPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  OpencvFixedPoseFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar