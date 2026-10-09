// Copyright 2026 OpenArm 实验 / Apache-2.0
// 增益一致性验收: 打印 C++ PreviewMPC 增益, 由
// real/right/mpc_track/check_gain_parity.py 与 Python mpc_core.py 比对。
// 用法: mpc_gain_dump <N> <dt> <w_p> <w_v> <w_a>   (参数缺省 25 0.01 1000 200 1)
#include "openarm_mpc_controller/preview_mpc.hpp"

#include <cstdlib>
#include <cstdio>

using openarm_mpc_controller::PreviewMPC;

int main(int argc, char ** argv)
{
  const int N = argc > 1 ? std::atoi(argv[1]) : 25;
  const double dt = argc > 2 ? std::atof(argv[2]) : 0.01;
  const double w_p = argc > 3 ? std::atof(argv[3]) : 1000.0;
  const double w_v = argc > 4 ? std::atof(argv[4]) : 200.0;
  const double w_a = argc > 5 ? std::atof(argv[5]) : 1.0;

  PreviewMPC mpc(7, N, dt, w_p, w_v, w_a);
  std::printf("N %d dt %.17g w_p %.17g w_v %.17g w_a %.17g\n", N, dt, w_p,
              w_v, w_a);
  for (int j = 0; j < 7; ++j) {
    const auto & K = mpc.gains(j);
    std::printf("K %d", j);
    for (int i = 0; i < K.size(); ++i) {
      std::printf(" %.17g", K[i]);
    }
    std::printf("\n");
  }
  return 0;
}
