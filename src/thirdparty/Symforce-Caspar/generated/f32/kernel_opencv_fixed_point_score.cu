#include "kernel_opencv_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvFixedPointScoreKernel(float* pose,
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
                                float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49;
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
    r5 = fmaf(r5, r6, r3);
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r3,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r16,
                       r17,
                       r18,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r20,
                                         r21,
                                         r22,
                                         r23);
    r24 = r18 * r20;
    r24 = fmaf(r6, r24, r17 * r23);
    r24 = fmaf(r19, r21, r24);
    r24 = fmaf(r16, r22, r24);
    r25 = 2.00000000000000000e+00;
    r26 = fmaf(r19, r20, r16 * r23);
    r27 = r17 * r22;
    r26 = fmaf(r6, r27, r26);
    r26 = fmaf(r18, r21, r26);
    r27 = r25 * r26;
    r28 = r24 * r27;
    r29 = fmaf(r17, r20, r18 * r23);
    r30 = r16 * r21;
    r29 = fmaf(r6, r30, r29);
    r29 = fmaf(r19, r22, r29);
    r30 = fmaf(r17, r21, r16 * r20);
    r30 = fmaf(r18, r22, r30);
    r30 = fmaf(r6, r30, r19 * r23);
    r19 = r29 * r30;
    r31 = fmaf(r25, r19, r28);
    r31 = fmaf(r13, r31, r11);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r32,
                       r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r20 * r21;
    r34 = r34 * r25;
    r35 = r22 * r23;
    r35 = fmaf(r25, r35, r34);
    r36 = -2.00000000000000000e+00;
    r37 = r22 * r22;
    r37 = r36 * r37;
    r38 = 1.00000000000000000e+00;
    r39 = r20 * r20;
    r39 = fmaf(r36, r39, r38);
    r40 = r37 + r39;
    r41 = r21 * r22;
    r41 = r41 * r25;
    r42 = r23 * r36;
    r43 = fmaf(r20, r42, r41);
    r44 = r25 * r29;
    r44 = r44 * r24;
    r45 = r26 * r36;
    r45 = fmaf(r30, r45, r44);
    r46 = r26 * r26;
    r46 = r46 * r36;
    r47 = r38 + r46;
    r48 = r29 * r29;
    r48 = r48 * r36;
    r47 = r47 + r48;
    r31 = fmaf(r11, r35, r31);
    r31 = fmaf(r32, r40, r31);
    r31 = fmaf(r33, r43, r31);
    r31 = fmaf(r15, r45, r31);
    r31 = fmaf(r14, r47, r31);
    r47 = r7 * r31;
    r45 = -9.99999999999999955e-07;
    r43 = r36 * r24;
    r29 = r29 * r27;
    r43 = fmaf(r30, r43, r29);
    r43 = fmaf(r13, r43, r12);
    r12 = r20 * r22;
    r12 = r12 * r25;
    r40 = fmaf(r21, r42, r12);
    r35 = r21 * r21;
    r35 = r36 * r35;
    r39 = r35 + r39;
    r49 = r20 * r23;
    r49 = fmaf(r25, r49, r41);
    r27 = fmaf(r30, r27, r44);
    r46 = r38 + r46;
    r44 = r24 * r24;
    r44 = r36 * r44;
    r46 = r46 + r44;
    r43 = fmaf(r11, r40, r43);
    r43 = fmaf(r33, r39, r43);
    r43 = fmaf(r32, r49, r43);
    r43 = fmaf(r14, r27, r43);
    r43 = fmaf(r15, r46, r43);
    r48 = r38 + r48;
    r48 = r48 + r44;
    r48 = fmaf(r13, r48, r10);
    r19 = fmaf(r36, r19, r28);
    r28 = r25 * r24;
    r28 = fmaf(r30, r28, r29);
    r29 = r21 * r23;
    r29 = fmaf(r25, r29, r12);
    r42 = fmaf(r22, r42, r34);
    r37 = r38 + r37;
    r37 = r37 + r35;
    r48 = fmaf(r14, r19, r48);
    r48 = fmaf(r15, r28, r48);
    r48 = fmaf(r33, r29, r48);
    r48 = fmaf(r32, r42, r48);
    r48 = fmaf(r11, r37, r48);
    r37 = fmaf(r31, r31, r48 * r48);
    r11 = fmaf(r43, r43, r37);
    r11 = rsqrtf(r11);
    r11 = r43 * r11;
    r43 = copysign(1.0, r11);
    r43 = fmaf(r45, r43, r11);
    r43 = acosf(r43);
    r45 = r43 * r43;
    r11 = r45 * r45;
    r42 = r43 * r11;
    r42 = fmaf(r9, r42, r43);
    r9 = r8 * r43;
    r42 = fmaf(r45, r9, r42);
    r32 = r0 * r43;
    r45 = r43 * r45;
    r45 = r45 * r45;
    r42 = fmaf(r45, r32, r42);
    r45 = r1 * r43;
    r11 = r11 * r11;
    r42 = fmaf(r11, r45, r42);
    r45 = 9.99999999999999955e-07;
    r37 = sqrtf(r37);
    r37 = r45 + r37;
    r37 = 1.0 / r37;
    r37 = r42 * r37;
    r5 = fmaf(r37, r47, r5);
    r6 = fmaf(r4, r6, r2);
    r4 = r3 * r48;
    r6 = fmaf(r37, r4, r6);
    r6 = fmaf(r6, r6, r5 * r5);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r6);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvFixedPointScore(float* pose,
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
                           float* const out_rTr,
                           size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPointScoreKernel<<<n_blocks, 1024>>>(pose,
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
                                                  out_rTr,
                                                  problem_size);
}

}  // namespace caspar