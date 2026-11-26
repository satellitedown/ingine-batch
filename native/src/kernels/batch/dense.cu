#include "kernels/batch/dense.h"
#include "kernels/draft/rng.h"

#include <stdexcept>
#include <string>

namespace ingine::kernels::batch {
namespace {

constexpr int H = prefill::kHidden;
constexpr int ROW_T = H / 4;  // 576 threads, float4 each
constexpr int ROW_W = ROW_T / 32;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
  return std::uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (std::uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// ---- the head: L[M, N] = A[M, K] . B[N, K]^T (FP32), and kb::sample's choice of a token per row
//
// Warp tile (MF * 16) x (NF * 8), WM x WN warps. Both operands are K-contiguous rows; a stage holds TM rows of A and
// TN rows of B, 64 elements (128 bytes = 8 chunks of 16 bytes, whole lines from DRAM) each, chunk c of row r at
// chunk c ^ (r & 7) so the 8 rows of an ldmatrix hit distinct banks. cp.async in STAGES stages (rows past M / N
// zero-filled), one barrier per stage, four k16 MMA steps per stage. The row tiles of A are the fastest grid
// dimension, so the CTAs that share B's rows run together and B streams from DRAM once. Epilogue: L to C (when
// given), and per row the tile's best (key, v) (pick_key; ties to the lowest v) to the partials.

constexpr int BK = 64;

struct HeadArgs {
  const __nv_bfloat16* A;
  const __nv_bfloat16* B;
  float* C;  // nullptr: L not stored
  int lda, ldb, ldc, M, N, K;
  const float* temps;
  std::uint64_t seed;
  int2* partials;  // [M, ntiles]: (key bits, v)
  int ntiles;
};

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// kb::sample's operations (batch.cu sample_kernel, raw logits): s_v = 30 tanh(bf16(l) / 30), plus for T > 0
// s_v / T + Gumbel noise of (seed, row, v)
__device__ __forceinline__ std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ std::uint64_t row_key(std::uint64_t seed, int row) {
  return mix64(seed ^ (std::uint64_t(row) * 0xD1B54A32D192ED03ull));
}
__device__ __forceinline__ float pick_key(float l, bool greedy, float inv_t, std::uint64_t rkey, int v) {
  float key = 30.f * tanhf(bf16r(l) / 30.f);
  if (!greedy) {
    key = key * inv_t + draft::gumbel_from_bits(mix64(rkey ^ std::uint64_t(v)));
  }
  return key;
}
__device__ __forceinline__ bool better(float a, int av, float b, int bv) { return a > b || (a == b && av < bv); }
template <int LANES>
__device__ __forceinline__ void lanes_best(float& k, int& v) {  // over lane bits below LANES
#pragma unroll
  for (int o = 1; o < LANES; o <<= 1) {
    const float ok = __shfl_xor_sync(0xffffffffu, k, o);
    const int ov = __shfl_xor_sync(0xffffffffu, v, o);
    if (better(ok, ov, k, v)) k = ok, v = ov;
  }
}

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int MF_, int NF_, int WM_, int WN_, int STAGES_>
struct MmaCfg {
  static constexpr int MF = MF_, NF = NF_, WM = WM_, WN = WN_, STAGES = STAGES_;
  static constexpr int TM = WM * MF * 16, TN = WN * NF * 8, NT = WM * WN * 32;
  static constexpr int A_BYTES = TM * BK * 2, STAGE = (TM + TN) * BK * 2, SMEM = STAGES * STAGE;
  static_assert(NF % 2 == 0 && SMEM <= 96 * 1024);
};

template <class C>
__global__ void __launch_bounds__(C::NT) head_kernel(const HeadArgs p) {
  extern __shared__ __align__(128) std::uint8_t smem[];
  constexpr int MF = C::MF, NF = C::NF, TM = C::TM, TN = C::TN, NT = C::NT, STAGES = C::STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int mtile = blockIdx.x, ntile = blockIdx.y;
  const __nv_bfloat16* A = p.A + std::size_t(mtile) * TM * p.lda;
  const __nv_bfloat16* B = p.B + std::size_t(ntile) * TN * p.ldb;
  const int m_rows = min(TM, p.M - mtile * TM), n_rows = min(TN, p.N - ntile * TN);
  const int nk = p.K / BK;

  auto load_a = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE;
    for (int i = tid; i < TM * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < m_rows;
      cp_async16(smem_addr(base + swz(r, c)), A + std::size_t(ok ? r : 0) * p.lda + kt * BK + c * 8, ok);
    }
  };
  auto load_b = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE + C::A_BYTES;
    for (int i = tid; i < TN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < n_rows;
      cp_async16(smem_addr(base + swz(r, c)), B + std::size_t(ok ? r : 0) * p.ldb + kt * BK + c * 8, ok);
    }
  };

  float acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // the weights do not depend on the previous kernel (programmatic dependent launch): their first stages load
  // before waiting for it
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s)
    if (s < nk) load_b(s, s);
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_a(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < nk) load_a(next % STAGES, next), load_b(next % STAGES, next);
    cp_commit();
    const std::uint32_t a_base = smem_addr(smem + (kt % STAGES) * C::STAGE), b_base = a_base + C::A_BYTES;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      std::uint32_t af[MF][4], bf[NF][2];
#pragma unroll
      for (int i = 0; i < MF; ++i)
        ldsm_x4(af[i], a_base + swz(warp_m * MF * 16 + i * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < NF / 2; ++j) {
        std::uint32_t r[4];
        ldsm_x4(r, b_base + swz(warp_n * NF * 8 + j * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1)));
        bf[2 * j][0] = r[0], bf[2 * j][1] = r[1], bf[2 * j + 1][0] = r[2], bf[2 * j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) mma_bf16(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
  }
  cp_wait<0>();
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");

  // thread holds rows lane / 4 (+ 8), columns 2 (lane % 4) (+ 1) of each m16 x n8 block
  float best_k[MF][2];
  int best_v[MF][2];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp_m * MF * 16 + i * 16 + (lane >> 2) + h * 8, grow = mtile * TM + row;
      best_k[i][h] = -INFINITY, best_v[i][h] = 0x7fffffff;
      if (row >= m_rows) continue;
      const float t = p.temps[grow];
      const bool greedy = !(t > 0.f);
      const float inv_t = greedy ? 1.f : 1.f / t;
      const std::uint64_t rkey = row_key(p.seed, grow);
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const int col = warp_n * NF * 8 + j * 8 + (lane & 3) * 2;
        if (col >= n_rows) continue;
        if (p.C)
          *reinterpret_cast<float2*>(p.C + std::size_t(grow) * p.ldc + ntile * TN + col) =
              make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
#pragma unroll
        for (int e = 0; e < 2; ++e) {  // N even: col + 1 < n_rows
          const int v = ntile * TN + col + e;
          const float key = pick_key(acc[i][j][2 * h + e], greedy, inv_t, rkey, v);
          if (better(key, v, best_k[i][h], best_v[i][h])) best_k[i][h] = key, best_v[i][h] = v;
        }
      }
    }
  // the tile's best per row: the row's 4 lanes, then the WN warps of its warp row (in the stages' shared memory,
  // once every warp is past its last MMA: no static shared memory, two CTAs per SM)
  __syncthreads();
  auto red_k = reinterpret_cast<float(*)[C::WN]>(smem);
  auto red_v = reinterpret_cast<int(*)[C::WN]>(smem + TM * C::WN * sizeof(float));
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      lanes_best<4>(best_k[i][h], best_v[i][h]);
      const int row = warp_m * MF * 16 + i * 16 + (lane >> 2) + h * 8;
      if ((lane & 3) == 0) red_k[row][warp_n] = best_k[i][h], red_v[row][warp_n] = best_v[i][h];
    }
  __syncthreads();
  for (int row = tid; row < m_rows; row += NT) {
    float k = red_k[row][0];
    int v = red_v[row][0];
    for (int w = 1; w < C::WN; ++w)
      if (better(red_k[row][w], red_v[row][w], k, v)) k = red_k[row][w], v = red_v[row][w];
    p.partials[std::size_t(mtile * TM + row) * p.ntiles + ntile] = make_int2(__float_as_int(k), v);
  }
}

// one CTA per row: the best of the row's tiles
constexpr int PICK_T = 256;
__global__ void __launch_bounds__(PICK_T) head_pick_kernel(const int2* __restrict__ partials, int ntiles,
                                                           std::int32_t* __restrict__ out) {
  const int2* const p = partials + std::size_t(blockIdx.x) * ntiles;
  float k = -INFINITY;
  int v = 0x7fffffff;
  for (int i = threadIdx.x; i < ntiles; i += PICK_T) {
    const int2 x = p[i];
    if (better(__int_as_float(x.x), x.y, k, v)) k = __int_as_float(x.x), v = x.y;
  }
  lanes_best<32>(k, v);
  __shared__ float s_k[PICK_T / 32];
  __shared__ int s_v[PICK_T / 32];
  if ((threadIdx.x & 31) == 0) s_k[threadIdx.x >> 5] = k, s_v[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < PICK_T / 32; ++w)
      if (better(s_k[w], s_v[w], s_k[0], s_v[0])) s_k[0] = s_k[w], s_v[0] = s_v[w];
    out[blockIdx.x] = s_v[0];
  }
}

// configs: 64 x 128 (8 warps), 32 x 128 (8 warps, up to 32 rows); two stages, so that two CTAs share an SM (one
// computes while the other waits)
constexpr int HEAD_TN = 128;
using Cfg64 = MmaCfg<2, 4, 2, 4, 2>;
using Cfg32 = MmaCfg<1, 4, 2, 4, 2>;
static_assert(Cfg64::TN == HEAD_TN && Cfg32::TN == HEAD_TN);

template <class C>
void launch(const HeadArgs& a, cudaStream_t stream) {
  static const bool attr = [] {
    return cudaFuncSetAttribute(head_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess;
  }();
  if (!attr) throw std::runtime_error("batch head_sample: shared memory attribute");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((a.M + C::TM - 1) / C::TM, a.ntiles);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute pdl;
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &pdl;
  cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, head_kernel<C>, a) != cudaSuccess)
    throw std::runtime_error("batch head_sample: launch failed");
}

// prefill gate_apply_kernel (same operations), gate blockIdx.y, f / g rows at fg + (g * T + t) * 2H (+ H); the rows
// indexed in parameter space (no local copy)
__global__ void __launch_bounds__(ROW_T)
    gate_apply_kernel(const __grid_constant__ prefill::GateRows rows, const float* __restrict__ fg, int T,
                      const float* __restrict__ norm_w, std::uint16_t* __restrict__ out, std::size_t out_stride) {
  const int t = blockIdx.x, g = blockIdx.y, c = threadIdx.x * 4;
  const std::size_t off = std::size_t(t) * H + c;
  const float* f = fg + (std::size_t(g) * T + t) * 2 * H + c;
  const float4 h4 = *reinterpret_cast<const float4*>(rows.h[g] + off);
  const float4 m4 = *reinterpret_cast<const float4*>(rows.msg[g] + off);
  const float4 f4 = *reinterpret_cast<const float4*>(f), g4 = *reinterpret_cast<const float4*>(f + H);
  const float hv[4] = {h4.x, h4.y, h4.z, h4.w}, mv[4] = {m4.x, m4.y, m4.z, m4.w};
  const float fv[4] = {f4.x, f4.y, f4.z, f4.w}, gv[4] = {g4.x, g4.y, g4.z, g4.w};
  float y[4];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // (1 - forget) * hidden + gain * message
    y[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[i])), hv[i]), __fmul_rn(tanhf(gv[i]), mv[i]));
    ss += y[i] * y[i];
  }
  // block sum as prefill's block_sum: warp sums, then the warps' sums in order
  __shared__ float red[ROW_W];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  ss = 0.f;
  for (int w = 0; w < ROW_W; ++w) ss += red[w];
  const float r = rsqrtf(ss * (1.f / H) + kEps);
  const float4 w4 = *reinterpret_cast<const float4*>(norm_w + std::size_t(g) * H + c);
  *reinterpret_cast<uint2*>(out + g * out_stride + off) =
      make_uint2(pack_bf16x2(y[0] * r * w4.x, y[1] * r * w4.y), pack_bf16x2(y[2] * r * w4.z, y[3] * r * w4.w));
}

}  // namespace

std::size_t head_partials(int M, int N) { return std::size_t(M) * ((N + HEAD_TN - 1) / HEAD_TN); }

void head_sample(const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, int M, int N, int K,
                 const float* temps, std::uint64_t seed, float* logits, int ldc, int2* partials, std::int32_t* out,
                 cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K % BK || lda % 8 || ldb % 8 || N % 2 || (logits && ldc % 2))
    throw std::invalid_argument("batch head_sample: unsupported shape");
  const HeadArgs a{A, B, logits, lda, ldb, ldc, M, N, K, temps, seed, partials, (N + HEAD_TN - 1) / HEAD_TN};
  if (M <= 32) launch<Cfg32>(a, stream);
  else launch<Cfg64>(a, stream);
  head_pick_kernel<<<M, PICK_T, 0, stream>>>(partials, a.ntiles, out);
}

void gate_apply(const prefill::GateRows& rows, int G, const float* fg, int T, const float* norm_w, __nv_bfloat16* out,
                std::size_t out_stride, cudaStream_t stream) {
  gate_apply_kernel<<<dim3(T, G), ROW_T, 0, stream>>>(rows, fg, T, norm_w, reinterpret_cast<std::uint16_t*>(out),
                                                       out_stride);
}

}  // namespace ingine::kernels::batch
