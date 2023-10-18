import time
import numpy as np
from tinygrad.helpers import dtypes, getenv, prod
from tinygrad.runtime.ops_hip import RawHIPBuffer, HIPProgram

# AMD_LOG_LEVEL=3 ./MIOpenDriver gemm --iter 1000 --time 1 --a_w 2048 --a_h 2048 --b_w 2048
# 5.5: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# 5.6: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# gets ~100
# hipExtModuleLaunchKernel ( 0x0x16ccde0, 2048, 16, 1, 128, 1, 1,
# 161.60 us = 106.31 TFLOPS
# with --batch_count 8 / 1.258128 ms / (8*2048*2048*2048*2)/(1.258128)*1e-9 / 109.24 TFLOPS

# we only get ~53
# KY=2 KX=2 N=2048 python3 extra/gemm/hip_matmul.py
#   4194304    324.76 us, would be  52899.88 GFLOPS matmul, 154.98 GB/s

N = getenv("N", 2048)
KX = getenv("KX", 4)
KY = getenv("KY", 4)
assert N%(16*KX) == 0, f"N must be multiple of {16*KX}"
assert N%(16*KY) == 0, f"N must be multiple of {16*KY}"
FLOPS = N*N*N*2
BW = N*N*3*4

a = RawHIPBuffer(N*N, dtypes.float32)

nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
b = RawHIPBuffer.fromCPU(nb)
c = RawHIPBuffer.fromCPU(nc)

prog = HIPProgram("test", f"""
#define F32
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
extern "C" __global__ void __launch_bounds__ (32, 1) test(float* c, __half* a, __half* b) {{
  const int gx = blockIdx.x;
  const int gy = blockIdx.y;

  const int lIdx = threadIdx.x;
  const int lane = lIdx%16;

  c += gx*{KX*16}*{N} + gy*{KY*16} + (lIdx/16)*{N} + lane;
  a += gx*{KX*16}*{N};
  b += gy*{KY*16};

  // KX&KY must be >= 2
  // Double buffers
  __shared__ half16 a_lds[16*{KX}*2];
  __shared__ half16 b_lds[16*{KY}*2];

  auto *a_lds_ptr_0 = &a_lds[0];
  auto *a_lds_ptr_1 = &a_lds[16*{KX}];

  auto *b_lds_ptr_0 = &b_lds[0];
  auto *b_lds_ptr_1 = &b_lds[16*{KX}];

  half16 a_frag;
  half16 b_frag;
  #ifdef F32
    float8 c_frag[{KY}][{KX}] = {{}};
  #else
    half16 c_frag[{KY}][{KX}] = {{}};
  #endif

  // Global mem reader
  half16 a_gr[{KX}/2];
  half16 b_gr[{KY}/2];

  // Prefetch for k = 0
  for(int kk = 0; kk < {KX}/2;kk++) {{
    for (int ele = 0; ele < 16; ++ele) {{
      a_gr[kk][ele] = a[(0+ele) + (threadIdx.x/16 +2*kk)*{16*N} + {N}*(threadIdx.x%16)];
    }}
  }}

  for (int ele = 0; ele < 16; ++ele) {{
    for(int kk = 0; kk < {KY}/2;kk++) {{
      b_gr[kk][ele] = b[(0+ele)*{N} + (threadIdx.x/16 + 2*kk)*16 + threadIdx.x%16];
    }}
  }}

  for (int ele = 0; ele < 16; ++ele) {{
    for(int kk = 0; kk < {KX}/2;kk++) {{
      a_lds_ptr_0[threadIdx.x + 32*kk][ele] = a_gr[kk][ele];
    }}
  }}

  for (int ele = 0; ele < 16; ++ele) {{
    for(int kk = 0; kk < {KY}/2;kk++) {{
      b_lds_ptr_0[threadIdx.x+32*kk][ele] = b_gr[kk][ele];
    }}
  }}

  __syncthreads();

  for (int k = 0; k < {N-16}; k += 16) {{
    // Prefetch for k+16
    for(int kk = 0; kk < {KX}/2;kk++) {{
      for (int ele = 0; ele < 16; ++ele) {{
        a_gr[kk][ele] = a[(k+16+ele) + (threadIdx.x/16 +2*kk)*{16*N} + {N}*(threadIdx.x%16)];
      }}
    }}

    for(int kk = 0; kk < {KY}/2;kk++) {{
      for (int ele = 0; ele < 16; ++ele) {{
        b_gr[kk][ele] = b[(k+16+ele)*{N} + (threadIdx.x/16 + 2*kk)*16 + threadIdx.x%16];
      }}
    }}

    for (int y = 0; y < {KY}; y++) {{
      for (int ele = 0; ele < 16; ++ele) {{
        b_frag[ele] = b_lds_ptr_0[y*16 + lane][ele];
      }}

      for (int x = 0; x < {KX}; x++) {{
        for (int ele = 0; ele < 16; ++ele) {{
          a_frag[ele] = a_lds_ptr_0[x*16 + lane][ele];
        }}
        #ifdef F32
          c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag[y][x]);
        #else
          c_frag[y][x] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag[y][x], false);
        #endif
      }}
    }}

    for (int ele = 0; ele < 16; ++ele) {{
      for(int kk = 0; kk < {KX}/2;kk++) {{
        a_lds_ptr_1[threadIdx.x+32*kk][ele] = a_gr[kk][ele];
      }}
    }}
    for (int ele = 0; ele < 16; ++ele) {{
      for(int kk = 0; kk < {KX}/2;kk++) {{
        b_lds_ptr_1[threadIdx.x+32*kk][ele] = b_gr[kk][ele];
      }}
    }}

    __syncthreads();

    auto *a_tmp = a_lds_ptr_0;
    a_lds_ptr_0 = a_lds_ptr_1;
    a_lds_ptr_1 = a_tmp;

    auto *b_tmp = b_lds_ptr_0;
    b_lds_ptr_0 = b_lds_ptr_1;
    b_lds_ptr_1 = b_tmp;

  }}

  for (int y = 0; y < {KY}; y++) {{
    for (int x = 0; x < {KX}; x++) {{
      for (int ele = 0; ele < 16; ++ele) {{
        a_frag[ele] = a_lds_ptr_0[x*16 + lane][ele];
      }}

      for (int ele = 0; ele < 16; ++ele) {{
        b_frag[ele] = b_lds_ptr_0[y*16 + lane][ele];
      }}

      #ifdef F32
        c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag[y][x]);
      #else
        c_frag[y][x] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag[y][x], false);
      #endif
    }}
  }}

  for (int ele = 0; ele < 8; ++ele) {{
    for (int y = 0; y < {KY}; y++) {{
      for (int x = 0; x < {KX}; x++) {{
        #ifdef F32
          c[ele*{2*N} + y*16 + x*{16*N}] = c_frag[y][x][ele];
        #else
          c[ele*{2*N} + y*16 + x*{16*N}] = c_frag[y][x][ele*2];
        #endif
      }}
    }}
  }}
}}""")

def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  ret = time.perf_counter() - st # NOTE: et doesn't contain the launch overhead
  #print(f"{ret*1e6:.2f} us")
  return et

global_size, local_size = [N//(KX*16), N//(KY*16), 1], [32, 1, 1]
print("global/local size", global_size, local_size, f"local_size:{prod(local_size)} total_size:{prod(global_size+local_size)}")
tm = min([timeit(lambda: prog(global_size, local_size, a, b, c, wait=True)) for _ in range(1000)])
na = a.toCPU().reshape(N,N)
comp = nb.astype(np.float32) @ nc.astype(np.float32)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
np.testing.assert_allclose(na, comp, atol=1e-2, rtol=1e-2)
