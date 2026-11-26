// The head with the token choice fused (kernels::batch::head_sample) against kernels::batch::sample over the logits
// it stores: the same token for every row, greedy and sampled, with and without storing the logits. Ties: duplicated
// weight rows (equal logits in different tiles) and saturated rows (30 tanh(l / 30) == 30 for many v).
#include "common/test_util.h"
#include "kernels/batch/batch.h"
#include "kernels/batch/dense.h"

#include <cstdio>
#include <iterator>
#include <vector>

using namespace ingine::test;
namespace kb = ingine::kernels::batch;

namespace {

int g_failures = 0;

void run(int M, int N, int K, cudaStream_t stream) {
  Rng r(std::uint64_t(M) * 7919u + N + K);
  const int ldc = (N + 7) / 8 * 8;
  std::vector<__nv_bfloat16> a(std::size_t(M) * K), b(std::size_t(N) * K);
  for (int i = 0; i < M; ++i) {
    // rows 3 mod 5: scaled so that most logits saturate the softcap
    const double scale = i % 5 == 3 ? 64.0 : 0.5;
    for (int k = 0; k < K; ++k) a[std::size_t(i) * K + k] = float_to_bf16(float(r.normal() * scale));
  }
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) b[std::size_t(j) * K + k] = float_to_bf16(float(r.normal()));
  // duplicated rows: v = 1000 + 7 t copies row 7 t + 3 (other tiles), the last row copies row 0
  for (int t = 0; 1000 + 7 * t < N; ++t)
    for (int k = 0; k < K; ++k) b[std::size_t(1000 + 7 * t) * K + k] = b[std::size_t(7 * t + 3) * K + k];
  for (int k = 0; k < K; ++k) b[std::size_t(N - 1) * K + k] = b[k];
  std::vector<float> temps(M);
  for (int i = 0; i < M; ++i) temps[i] = i % 3 == 0 ? 0.f : i % 3 == 1 ? -1.f : float(0.25 + r.uniform());
  DeviceArray<__nv_bfloat16> da(a), db(b);
  DeviceArray<float> dt(temps), logits(std::size_t(M) * ldc);
  DeviceArray<int2> partials(kb::head_partials(M, N));
  DeviceArray<std::int32_t> fused(M), unstored(M), ref(M);
  for (const std::uint64_t seed : {std::uint64_t(1), std::uint64_t(0x9E3779B97F4A7C15ull)}) {
    kb::head_sample(da.get(), K, db.get(), K, M, N, K, dt.get(), seed, logits.get(), ldc, partials.get(), fused.get(),
                    stream);
    kb::sample(logits.get(), ldc, M, N, true, dt.get(), seed, ref.get(), stream);
    kb::head_sample(da.get(), K, db.get(), K, M, N, K, dt.get(), seed, nullptr, ldc, partials.get(), unstored.get(),
                    stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const std::vector<std::int32_t> hf = fused.download(), hu = unstored.download(), hr = ref.download();
    int diff = 0, ties = 0;
    for (int i = 0; i < M; ++i) {
      if (hf[i] != hr[i] || hu[i] != hr[i]) ++diff;
      if (i % 5 == 3 && temps[i] <= 0.f) ++ties;
    }
    std::printf("%s M %d N %d K %d seed %llx: %d of %d rows differ from sample (%d saturated greedy rows)\n",
                diff ? "FAIL" : "PASS", M, N, K, (unsigned long long)seed, diff, M, ties);
    if (diff) ++g_failures;
  }
}

// Regression test for 24-bit Gumbel rounding to u = 1.0f (+inf noise): with a single dominant token (+30 vs -30,
// gap 60, p(other) < 1e-20), every sampled draw at T = 1 must select token 0 across both head_sample and sample.
void test_no_inf_gumbel(cudaStream_t stream) {
  constexpr int M = 256, N = 200058, K = 64;
  constexpr int ldc = (N + 7) / 8 * 8;
  std::vector<__nv_bfloat16> a(std::size_t(M) * K, float_to_bf16(0.f));
  std::vector<__nv_bfloat16> b(std::size_t(N) * K, float_to_bf16(0.f));
  for (int i = 0; i < M; ++i) a[std::size_t(i) * K] = float_to_bf16(1.f);
  b[0] = float_to_bf16(1000.f);  // 30 * tanh(1000 / 30) = +30
  for (int j = 1; j < N; ++j) b[std::size_t(j) * K] = float_to_bf16(-1000.f);  // -30
  std::vector<float> temps(M, 1.f);
  std::vector<float> capped(std::size_t(M) * ldc, -30.f);
  for (int i = 0; i < M; ++i) capped[std::size_t(i) * ldc] = 30.f;

  DeviceArray<__nv_bfloat16> da(a), db(b);
  DeviceArray<float> dt(temps), logits(std::size_t(M) * ldc), dcapped(capped);
  DeviceArray<int2> partials(kb::head_partials(M, N));
  DeviceArray<std::int32_t> fused(M), unstored(M), ref_raw(M), ref_capped(M);

  int bad_fused = 0, bad_unstored = 0, bad_raw = 0, bad_capped = 0;
  constexpr std::uint64_t seeds[] = {1ull, 2ull, 3ull, 42ull, 0x9E3779B97F4A7C15ull};
  for (const std::uint64_t seed : seeds) {
    kb::head_sample(da.get(), K, db.get(), K, M, N, K, dt.get(), seed, logits.get(), ldc, partials.get(), fused.get(),
                    stream);
    kb::head_sample(da.get(), K, db.get(), K, M, N, K, dt.get(), seed, nullptr, ldc, partials.get(), unstored.get(),
                    stream);
    kb::sample(logits.get(), ldc, M, N, true, dt.get(), seed, ref_raw.get(), stream);
    kb::sample(dcapped.get(), ldc, M, N, false, dt.get(), seed, ref_capped.get(), stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto hf = fused.download(), hu = unstored.download(), hr = ref_raw.download(), hc = ref_capped.download();
    for (int i = 0; i < M; ++i) {
      if (hf[i] != 0) ++bad_fused;
      if (hu[i] != 0) ++bad_unstored;
      if (hr[i] != 0) ++bad_raw;
      if (hc[i] != 0) ++bad_capped;
    }
  }
  const int total = M * int(std::size(seeds));
  const int bad = bad_fused + bad_unstored + bad_raw + bad_capped;
  std::printf("%s no-inf Gumbel tail (%d draws, N=%d): bad fused=%d unstored=%d sample_raw=%d sample_capped=%d\n",
              bad ? "FAIL" : "PASS", total, N, bad_fused, bad_unstored, bad_raw, bad_capped);
  if (bad) ++g_failures;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  run(7, 5000, 128, stream);       // 32-row tiles
  run(40, 20002, 256, stream);     // 64-row tiles, a partial one
  run(130, 200058, 64, stream);    // the vocabulary: a partial last column tile
  test_no_inf_gumbel(stream);
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
