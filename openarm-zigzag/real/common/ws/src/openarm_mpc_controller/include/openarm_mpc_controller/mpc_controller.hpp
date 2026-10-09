// Copyright 2026 OpenArm 实验 / Apache-2.0
// 线路一: MPC 控制器插件 —— FollowJointTrajectory 进, position 指令出。
// 以 right_joint_trajectory_controller 槽位名注册(MoveIt/脚本零修改),
// 数学与 real/right/mpc_track/mpc_core.py 逐点一致。
// 设计/行为规格: docs/MPC控制器插件_线路一改造设计.md
#ifndef OPENARM_MPC_CONTROLLER__MPC_CONTROLLER_HPP_
#define OPENARM_MPC_CONTROLLER__MPC_CONTROLLER_HPP_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "controller_interface/controller_interface.hpp"
#include "openarm_mpc_controller/preview_mpc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_server_goal_handle.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

namespace openarm_mpc_controller
{

class OpenArmMpcController : public controller_interface::ControllerInterface
{
public:
  using FollowJTrajAction = control_msgs::action::FollowJointTrajectory;
  using GoalHandle = rclcpp_action::ServerGoalHandle<FollowJTrajAction>;
  using RtGoalHandle = realtime_tools::RealtimeServerGoalHandle<FollowJTrajAction>;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration()
  const override;
  controller_interface::InterfaceConfiguration state_interface_configuration()
  const override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // ---- 打包结构(须先于使用处声明) ----
  struct Traj
  {
    std::vector<double> times;               // 各点相对起点秒, 递增, 首点=0
    std::vector<std::vector<double>> points; // [point][joint], 内部关节序
    double duration{0.0};
  };
  using TrajPtr = std::shared_ptr<const Traj>;
  /// RealtimeBuffer 载荷: 轨迹 + 代号(防"完结轨迹被再采纳")
  struct TrajSlot
  {
    TrajPtr traj;
    uint64_t gen{0};
  };
  struct GhSlot
  {
    std::shared_ptr<RtGoalHandle> gh;
    uint64_t gen{0};
  };
  /// RT → 非 RT(起点校验)的测量缓存, 互斥锁保护(读写都极短)
  struct Meas
  {
    std::vector<double> pos;
    std::vector<double> vel;
    bool valid{false};
  };

  // ---- action 回调(executor 线程, 非 RT) ----
  rclcpp_action::GoalResponse on_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const FollowJTrajAction::Goal> goal);
  rclcpp_action::CancelResponse on_cancel(const std::shared_ptr<GoalHandle> gh);
  void on_accepted(const std::shared_ptr<GoalHandle> gh);

  /// 内部打包轨迹(已换到内部关节序)。返回 false 时 why 给出拒绝原因。
  bool prepare_traj(
    const FollowJTrajAction::Goal & goal, TrajPtr & out,
    std::string & why) const;

  /// RT 侧终结 goal(success/abort/cancel): 置结果并立即 runNonRealtime 冲发
  void finish_goal_(bool success, const std::string & reason);

  /// 周期冲发 RealtimeServerGoalHandle 队列(响应/结果真正发出在这里)
  void flush_goal_();

  /// 轨迹第 j 关节在 tt 秒的线性插值(clamp 到两端)。
  double sample_at_(const Traj & t, size_t j, double tt) const;

  // ---- 参数(yaml 注入) ----
  std::vector<std::string> joint_names_;
  int n_joints_{7};
  int N_{25};
  double dt_{0.01};
  double w_p_{1000.0};
  double w_v_{200.0};
  double w_a_{1.0};
  double a_max_{1.2};
  double speed_scale_{1.0};
  double clearance_deg_{3.0};
  double start_tol_{0.035};       // 起点校验(rad, ≈2°)
  double goal_tol_{0.20};         // 结束判定(rad, 对齐旧 JTC 最松 goal 容差)
  double abort_dev_{0.40};        // 测量-指令偏差保护(rad, 对齐旧 JTC 最松 trajectory 容差)
  std::string safety_yaml_;       // 软限位单一权威来源 real_safety.yaml

  // ---- 数学 + 净空盒 ----
  std::unique_ptr<PreviewMPC> mpc_;
  std::vector<double> lo_box_, hi_box_;
  std::vector<double> ref_win_;   // 复用缓冲, 避免周期内堆分配

  // ---- RT 侧状态(仅 update() 触碰) ----
  std::vector<double> q_model_, v_model_, q_cmd_;
  std::vector<double> meas_pos_, meas_vel_;
  TrajPtr traj_;                  // 当前活跃轨迹
  uint64_t traj_gen_{0};
  rclcpp::Time traj_start_{0, 0, RCL_SYSTEM_TIME};
  bool traj_running_{false};
  double traj_duration_{0.0};

  // ---- 非 RT → RT 交接(RealtimeBuffer) ----
  realtime_tools::RealtimeBuffer<TrajSlot> traj_box_;
  realtime_tools::RealtimeBuffer<GhSlot> gh_box_;
  std::atomic<bool> cancel_req_{false};
  std::atomic<int> watchdog_state_{0};   // 0=未知/OK, 1=TRIPPED/RETURNING
  std::atomic<bool> estop_latch_{false}; // 置位后拒绝新 goal, on_activate 清零
  std::atomic<bool> goal_busy_{false};

  // ---- RT → 非 RT 测量缓存 ----
  mutable std::mutex meas_mutex_;
  Meas meas_cache_;

  // 非 RT 侧待发(单线程 executor, on_goal→on_accepted 顺序执行);
  // next_gen_ 为纯非 RT 代号计数器(与 RT 侧 traj_gen_ 不共享写)
  TrajPtr pending_traj_;
  uint64_t next_gen_{1};

  rclcpp_action::Server<FollowJTrajAction>::SharedPtr action_server_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_safety_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_estop_;
};

}  // namespace openarm_mpc_controller

#endif  // OPENARM_MPC_CONTROLLER__MPC_CONTROLLER_HPP_
