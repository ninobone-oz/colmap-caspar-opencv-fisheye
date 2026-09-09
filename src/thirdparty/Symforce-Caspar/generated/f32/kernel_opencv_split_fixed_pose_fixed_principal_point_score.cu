#include "kernel_opencv_split_fixed_pose_fixed_principal_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedPrincipalPointScoreKernel(
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
        float* const out_rTr,
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
      r46, r47;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r4, r1);
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r1,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r18, r19, r20, r21);
    r22 = r14 * r20;
    r22 = fmaf(r4, r22, r17 * r19);
    r22 = fmaf(r15, r21, r22);
    r22 = fmaf(r16, r18, r22);
    r23 = 2.00000000000000000e+00;
    r24 = fmaf(r14, r21, r17 * r18);
    r25 = r16 * r19;
    r24 = fmaf(r4, r25, r24);
    r24 = fmaf(r15, r20, r24);
    r25 = r23 * r24;
    r26 = r22 * r25;
    r27 = fmaf(r14, r19, r17 * r20);
    r28 = r15 * r18;
    r27 = fmaf(r4, r28, r27);
    r27 = fmaf(r16, r21, r27);
    r28 = fmaf(r15, r19, r14 * r18);
    r28 = fmaf(r16, r20, r28);
    r28 = fmaf(r4, r28, r17 * r21);
    r21 = r27 * r28;
    r29 = fmaf(r23, r21, r26);
    r29 = fmaf(r11, r29, r9);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r9, r30, r31);
    r32 = r14 * r15;
    r32 = r32 * r23;
    r33 = r16 * r17;
    r33 = fmaf(r23, r33, r32);
    r34 = -2.00000000000000000e+00;
    r35 = r16 * r16;
    r35 = r34 * r35;
    r36 = 1.00000000000000000e+00;
    r37 = r14 * r14;
    r37 = fmaf(r34, r37, r36);
    r38 = r35 + r37;
    r39 = r15 * r16;
    r39 = r39 * r23;
    r40 = r17 * r34;
    r41 = fmaf(r14, r40, r39);
    r42 = r23 * r27;
    r42 = r42 * r22;
    r43 = r24 * r34;
    r43 = fmaf(r28, r43, r42);
    r44 = r24 * r24;
    r44 = r44 * r34;
    r45 = r36 + r44;
    r46 = r27 * r27;
    r46 = r46 * r34;
    r45 = r45 + r46;
    r29 = fmaf(r9, r33, r29);
    r29 = fmaf(r30, r38, r29);
    r29 = fmaf(r31, r41, r29);
    r29 = fmaf(r13, r43, r29);
    r29 = fmaf(r12, r45, r29);
    r45 = r5 * r29;
    r43 = -9.99999999999999955e-07;
    r41 = r34 * r22;
    r27 = r27 * r25;
    r41 = fmaf(r28, r41, r27);
    r41 = fmaf(r11, r41, r10);
    r10 = r14 * r16;
    r10 = r10 * r23;
    r38 = fmaf(r15, r40, r10);
    r33 = r15 * r15;
    r33 = r34 * r33;
    r37 = r33 + r37;
    r47 = r14 * r17;
    r47 = fmaf(r23, r47, r39);
    r25 = fmaf(r28, r25, r42);
    r44 = r36 + r44;
    r42 = r22 * r22;
    r42 = r34 * r42;
    r44 = r44 + r42;
    r41 = fmaf(r9, r38, r41);
    r41 = fmaf(r31, r37, r41);
    r41 = fmaf(r30, r47, r41);
    r41 = fmaf(r12, r25, r41);
    r41 = fmaf(r13, r44, r41);
    r46 = r36 + r46;
    r46 = r46 + r42;
    r46 = fmaf(r11, r46, r8);
    r21 = fmaf(r34, r21, r26);
    r26 = r23 * r22;
    r26 = fmaf(r28, r26, r27);
    r27 = r15 * r17;
    r27 = fmaf(r23, r27, r10);
    r40 = fmaf(r16, r40, r32);
    r35 = r36 + r35;
    r35 = r35 + r33;
    r46 = fmaf(r12, r21, r46);
    r46 = fmaf(r13, r26, r46);
    r46 = fmaf(r31, r27, r46);
    r46 = fmaf(r30, r40, r46);
    r46 = fmaf(r9, r35, r46);
    r35 = fmaf(r29, r29, r46 * r46);
    r9 = fmaf(r41, r41, r35);
    r9 = rsqrtf(r9);
    r9 = r41 * r9;
    r41 = copysign(1.0, r9);
    r41 = fmaf(r43, r41, r9);
    r41 = acosf(r41);
  };
  LoadShared<2, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r43,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r41 * r41;
    r30 = r40 * r40;
    r27 = r30 * r30;
    r27 = r41 * r27;
    r27 = fmaf(r9, r27, r41);
    r9 = r6 * r41;
    r27 = fmaf(r40, r9, r27);
    r31 = r43 * r41;
    r40 = r41 * r40;
    r40 = r40 * r40;
    r27 = fmaf(r40, r31, r27);
    r40 = r7 * r41;
    r27 = fmaf(r30, r40, r27);
    r40 = 9.99999999999999955e-07;
    r35 = sqrtf(r35);
    r35 = r40 + r35;
    r35 = 1.0 / r35;
    r35 = r27 * r35;
    r3 = fmaf(r35, r45, r3);
    r4 = fmaf(r2, r4, r0);
    r2 = r1 * r46;
    r4 = fmaf(r35, r2, r4);
    r4 = fmaf(r4, r4, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedPrincipalPointScore(
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
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPoseFixedPrincipalPointScoreKernel<<<n_blocks, 1024>>>(
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
      out_rTr,
      problem_size);
}

}  // namespace caspar