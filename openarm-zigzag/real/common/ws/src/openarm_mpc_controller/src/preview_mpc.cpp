// Copyright 2026 OpenArm 实验 / Apache-2.0
#include "openarm_mpc_controller/preview_mpc.hpp"

#include <algorithm>

namespace openarm_mpc_controller
{

PreviewMPC::PreviewMPC(int n_joints, int N, double dt,
                       double w_p, double w_v, double w_a)
: n_joints_(n_joints), N_(N), dt_(dt)
{
  // 精确 ZOH 双积分离散化(与 mpc_core.py 逐行对应):
  //   p_j = q0 + j·h·v0 + h² Σ_{m<j} (j-m-0.5) a_m
  //   v_j = v0 + h  Σ_{m<j}          a_m
  const double h2 = dt * dt;
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd Lv = Eigen::MatrixXd::Zero(N, N);
  for (int j = 0; j < N; ++j) {
    for (int m = 0; m < N; ++m) {
      if (m < j) {
        L(j, m) = h2 * (j - m - 0.5);
        Lv(j, m) = dt;
      }
    }
  }
  Eigen::MatrixXd M(2 * N, N);
  M.topRows(N) = L;
  M.bottomRows(N) = Lv;

  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(2 * N, 2 * N);
  for (int j = 0; j < N; ++j) {
    W(j, j) = w_p;
    W(N + j, N + j) = w_v;
  }
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(N, N);
  for (int j = 0; j < N; ++j) {
    R(j, j) = w_a;
  }

  // Kfull = (MᵀWM+R)⁻¹ MᵀW; 首行 = a0 对自由误差 e 的增益
  Eigen::MatrixXd Kfull =
    (M.transpose() * W * M + R).ldlt().solve(M.transpose() * W);
  K_.reserve(static_cast<size_t>(n_joints));
  for (int i = 0; i < n_joints; ++i) {
    K_.push_back(Kfull.row(0));
  }
}

double PreviewMPC::step(int joint, double q, double dq,
                        const double * ref, double a_max) const
{
  const int N = N_;
  const double h = dt_;
  // 自由预测误差 e(2N): [p_j - r_j ; dq - ṙ_j]
  Eigen::VectorXd e(2 * N);
  for (int j = 1; j <= N; ++j) {
    e(j - 1) = q + j * h * dq - ref[j];
    const double dr = (ref[j] - ref[j - 1]) / h;
    e(N + j - 1) = dq - dr;
  }
  const double a = -K_[static_cast<size_t>(joint)].dot(e);
  return std::max(-a_max, std::min(a_max, a));
}

}  // namespace openarm_mpc_controller
