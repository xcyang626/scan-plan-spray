// Copyright 2026 OpenArm 实验 / Apache-2.0
// 闭式预览 MPC 数学核 —— real/right/mpc_track/mpc_core.py PreviewMPC 的
// C++ 移植, 数学逐点一致(验收: tools/mpc_gain_dump + check_gain_parity.py)。
//
// 模型(每关节独立):  q' = dq;  dq' = a        (ZOH 双积分离散)
// 代价(N 步):        Σ w_p(q_j - r_j)² + w_v(dq_j - ṙ_j)² + w_a·a_j²
// 求解:              闭式线性增益 K(建时预计算, 无 QP 求解器依赖)
// 约束处理(调用方):  a 饱和 ±a_max + 预测位置净空盒投影
// 设计依据: docs/MPC控制器插件_线路一改造设计.md §三
#ifndef OPENARM_MPC_CONTROLLER__PREVIEW_MPC_HPP_
#define OPENARM_MPC_CONTROLLER__PREVIEW_MPC_HPP_

#include <vector>

#include <Eigen/Dense>

namespace openarm_mpc_controller
{

class PreviewMPC
{
public:
  /// n_joints 个关节独立求解(增益逐关节相同, 但保留逐关节接口以对齐 Python 版)。
  PreviewMPC(int n_joints, int N, double dt,
             double w_p, double w_v, double w_a);

  int horizon() const {return N_;}
  double dt() const {return dt_;}

  /// 单关节一步。ref_window: (N+1,) 参考位置窗 [r_0 .. r_N]。
  /// 速度参考取窗内有限差分(移动参考自动前馈)。返回饱和后的 a0。
  double step(int joint, double q, double dq,
              const double * ref_window, double a_max) const;

  /// 调试/一致性验收: 该关节增益行 (2N,)。
  const Eigen::VectorXd & gains(int joint) const
  {
    return K_[static_cast<size_t>(joint)];
  }

private:
  int n_joints_;
  int N_;
  double dt_;
  std::vector<Eigen::VectorXd> K_;
};

}  // namespace openarm_mpc_controller

#endif  // OPENARM_MPC_CONTROLLER__PREVIEW_MPC_HPP_
