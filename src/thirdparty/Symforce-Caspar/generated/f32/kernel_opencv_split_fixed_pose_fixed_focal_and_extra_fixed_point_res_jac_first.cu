#include "kernel_opencv_split_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedFocalAndExtraFixedPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
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
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r11, r12, r13);
    r14 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r19, r20, r21, r22);
    r23 = r15 * r21;
    r23 = fmaf(r4, r23, r18 * r20);
    r23 = fmaf(r16, r22, r23);
    r23 = fmaf(r17, r19, r23);
    r24 = r14 * r23;
    r24 = r24 * r23;
    r25 = 1.00000000000000000e+00;
    r26 = fmaf(r15, r20, r18 * r21);
    r27 = r16 * r19;
    r26 = fmaf(r4, r27, r26);
    r26 = fmaf(r17, r22, r26);
    r27 = r26 * r26;
    r27 = fmaf(r14, r27, r25);
    r28 = r24 + r27;
    r28 = fmaf(r11, r28, r8);
    r8 = fmaf(r15, r22, r18 * r19);
    r29 = r17 * r20;
    r8 = fmaf(r4, r29, r8);
    r8 = fmaf(r16, r21, r8);
    r29 = 2.00000000000000000e+00;
    r30 = r29 * r23;
    r31 = r8 * r30;
    r32 = fmaf(r16, r20, r15 * r19);
    r32 = fmaf(r17, r21, r32);
    r32 = fmaf(r4, r32, r18 * r22);
    r22 = r14 * r32;
    r33 = fmaf(r26, r22, r31);
    r34 = r26 * r29;
    r34 = r34 * r8;
    r35 = fmaf(r32, r30, r34);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r36, r37, r38);
    r39 = r15 * r17;
    r39 = r39 * r29;
    r40 = r16 * r18;
    r41 = fmaf(r29, r40, r39);
    r42 = r17 * r18;
    r43 = r15 * r16;
    r43 = r43 * r29;
    r42 = fmaf(r14, r42, r43);
    r44 = r17 * r17;
    r44 = r14 * r44;
    r45 = r25 + r44;
    r46 = r16 * r16;
    r46 = r46 * r14;
    r45 = r45 + r46;
    r28 = fmaf(r12, r33, r28);
    r28 = fmaf(r13, r35, r28);
    r28 = fmaf(r38, r41, r28);
    r28 = fmaf(r37, r42, r28);
    r28 = fmaf(r36, r45, r28);
    r45 = r0 * r28;
    r42 = -9.99999999999999955e-07;
    r23 = fmaf(r23, r22, r34);
    r23 = fmaf(r11, r23, r10);
    r40 = fmaf(r14, r40, r39);
    r46 = r25 + r46;
    r39 = r15 * r15;
    r39 = r14 * r39;
    r46 = r46 + r39;
    r10 = r16 * r17;
    r10 = r10 * r29;
    r34 = r15 * r18;
    r34 = fmaf(r29, r34, r10);
    r41 = r29 * r8;
    r30 = r26 * r30;
    r41 = fmaf(r32, r41, r30);
    r24 = r25 + r24;
    r35 = r8 * r8;
    r35 = r14 * r35;
    r24 = r24 + r35;
    r23 = fmaf(r36, r40, r23);
    r23 = fmaf(r38, r46, r23);
    r23 = fmaf(r37, r34, r23);
    r23 = fmaf(r12, r41, r23);
    r23 = fmaf(r13, r24, r23);
    r24 = r26 * r29;
    r24 = fmaf(r32, r24, r31);
    r24 = fmaf(r11, r24, r9);
    r11 = r17 * r18;
    r11 = fmaf(r29, r11, r43);
    r44 = r25 + r44;
    r44 = r44 + r39;
    r39 = r15 * r18;
    r39 = fmaf(r14, r39, r10);
    r22 = fmaf(r8, r22, r30);
    r27 = r35 + r27;
    r24 = fmaf(r36, r11, r24);
    r24 = fmaf(r37, r44, r24);
    r24 = fmaf(r38, r39, r24);
    r24 = fmaf(r13, r22, r24);
    r24 = fmaf(r12, r27, r24);
    r27 = fmaf(r24, r24, r28 * r28);
    r12 = fmaf(r23, r23, r27);
    r12 = rsqrtf(r12);
    r12 = r23 * r12;
    r23 = copysign(1.0, r12);
    r23 = fmaf(r42, r23, r12);
    r23 = acosf(r23);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r42,
                                         r12);
    r22 = r23 * r23;
    r13 = r22 * r22;
    r39 = r13 * r13;
    r39 = r23 * r39;
    r39 = fmaf(r12, r39, r23);
    r12 = r6 * r23;
    r39 = fmaf(r22, r12, r39);
    r38 = r42 * r23;
    r22 = r23 * r22;
    r22 = r22 * r22;
    r39 = fmaf(r22, r38, r39);
    r22 = r7 * r23;
    r39 = fmaf(r13, r22, r39);
    r22 = 9.99999999999999955e-07;
    r27 = sqrtf(r27);
    r27 = r22 + r27;
    r27 = 1.0 / r27;
    r27 = r39 * r27;
    r2 = fmaf(r27, r45, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r5 * r24;
    r3 = fmaf(r27, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fmaf(r2, r2, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    r2 = r4 * r2;
    r3 = r4 * r3;
    WriteSum2<float, float>((float*)inout_shared, r2, r3);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r25, r25);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedFocalAndExtraFixedPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  OpencvSplitFixedPoseFixedFocalAndExtraFixedPointResJacFirstKernel<<<n_blocks,
                                                                      1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar