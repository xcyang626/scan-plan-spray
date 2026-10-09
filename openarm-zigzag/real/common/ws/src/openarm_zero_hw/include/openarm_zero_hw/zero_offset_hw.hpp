#pragma once

// Copyright 2025 Enactic, Inc. / Apache-2.0
// 基于官方 openarm_hardware/openarm_simple_hardware.hpp 修改。
// 修改目的(真机安全部署, 详见 real/DEPLOY_指令.md):
//   1. 增加 per-joint zero 偏移 d: 电机反馈系 M 与 URDF 关节系 U 的换算,
//      urdf = motor - d, motor = urdf + d (d 来自实机标定 zero 文件,
//      由 config/gen_real_config.py 生成后写入 ros2_control 参数 d1..d7)。
//      官方驱动直接把电机角度当 URDF 关节值, 对标定过的机器人会完全错位。
//   2. on_activate 不再自动 return_to_zero(官方会在 2s 内以 kp=70 大增益
//      插值到电机零位, 对未对齐机器人是危险动作), 改为使能后锁定当前位置。

#include <chrono>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "openarm_zero_hw/visibility_control.h"
#include "rclcpp/macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace openarm_zero_hw {

class ZeroOffsetHW : public hardware_interface::SystemInterface {
 public:
  ZeroOffsetHW();

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_init(
      const hardware_interface::HardwareInfo& info) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces()
      override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces()
      override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

  hardware_interface::CallbackReturn on_shutdown(
      const rclcpp_lifecycle::State& previous_state) override;

  hardware_interface::return_type read(const rclcpp::Time& time,
                                       const rclcpp::Duration& period) override;

  hardware_interface::return_type write(
      const rclcpp::Time& time, const rclcpp::Duration& period) override;

 private:
  static constexpr size_t ARM_DOF = 7;

  // 官方 V10 电机配置: DM8009x2 / DM4340x2 / DM4310x3
  const std::vector<openarm::damiao_motor::MotorType> MOTOR_TYPES_ = {
      openarm::damiao_motor::MotorType::DM8009,
      openarm::damiao_motor::MotorType::DM8009,
      openarm::damiao_motor::MotorType::DM4340,
      openarm::damiao_motor::MotorType::DM4340,
      openarm::damiao_motor::MotorType::DM4310,
      openarm::damiao_motor::MotorType::DM4310,
      openarm::damiao_motor::MotorType::DM4310};

  const std::vector<uint32_t> SEND_CAN_IDS_ = {0x01, 0x02, 0x03, 0x04,
                                               0x05, 0x06, 0x07};
  const std::vector<uint32_t> RECV_CAN_IDS_ = {0x11, 0x12, 0x13, 0x14,
                                               0x15, 0x16, 0x17};

  // 配置
  std::string can_interface_;
  std::string arm_prefix_;
  bool can_fd_;
  std::vector<double> kp_;
  std::vector<double> kd_;
  // 夹爪 ESC8(hand=hold): 使能后原地 MIT 锁定, 不暴露 ROS 命令接口
  bool hand_hold_{false};
  double hand_kp_{0.0};
  double hand_kd_{0.0};
  double hand_q0_{0.0};

  // 软件积分器(重力补偿): write() 中对 电机系误差积分, 以 tau 前馈
  // 叠加进 MIT 指令, 消除纯位置控制的稳态重力下垂(带钳位防饱和)。
  std::vector<double> integ_;
  std::vector<double> err_f_;      // EMA 平滑后的误差
  std::vector<double> ki_;
  std::vector<double> integ_clamp_;

  // 注入通道(2026-09-18 完全分离 + 无残留): 方案由 gen_real_config 写入
  // 的 mpc_topic 参数决定(指向 RL 或 MPC 话题之一, verify_deploy 审计),
  // 驱动只订阅这一个 —— 方案隔离在配置层完成, 运行时不存在双源。上升沿
  // (从无到有)以当前实际位置重置基线 → 上一会话残留结构性清除(16:09
  // 启动踢跳事故根因); 流断 → fallback_slew_ 限速滑回 JTC(看门狗回位
  // 无缝衔接, 闭锁至追上)。消息: 7 项=位置 | 14=位置+前馈 | 21=位置+
  // 速度+前馈。钳位: 绝对±6.5/0.2rad、vel_clamp_、ff_clamp_。
  struct InjectSrc {
    std::vector<double> cmd, vel, ff;
    rclcpp::Time stamp{};
    // 跨线程(订阅回调线程写 / RT write 读)标志用 atomic; 与原始实现
    // (mpc_fresh_ atomic + stamp 普通读)保持同等级别的同步
    std::atomic<bool> got{false};     // 曾收到消息
    std::atomic<bool> active{false};  // 上一周期被采用(上升沿检测)
  };
  InjectSrc inj_;
  std::string inj_name_{"?"};          // 日志用: 由话题判定 RL/MPC
  rclcpp::Node::SharedPtr mpc_node_;   // 注入节点(提供时钟与订阅)
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr inj_sub_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr mpc_exec_;
  std::thread mpc_spin_;
  std::vector<double> ff_clamp_;
  std::vector<double> vel_clamp_;
  double inj_fresh_s_{0.2};
  std::vector<double> last_cmd_urdf_;  // 最后下发指令(断流滑行基线)
  double fallback_slew_;
  bool fallback_active_{false};   // 滑行闭锁: 追上 JTC 目标(<0.02rad)才解除
  bool prev_use_inj_{false};      // 上周期是否有注入源(断流沿检测)
  std::string last_src_{"init"};  // 上周期源(日志)
  // 电机系→URDF 系综合偏移(rad): urdf = motor - d
  std::vector<double> d_;
  // 通道映射: 第 i 个 URDF 关节使用第 channel_perm_[i] 台电机(0 基 ESC 序),
  // channel_sign_[i] = ±1, 电机正方向与模型关节正方向的关系
  std::vector<size_t> channel_perm_;
  std::vector<double> channel_sign_;

  std::unique_ptr<openarm::can::socket::OpenArm> openarm_;
  std::vector<std::string> joint_names_;

  std::vector<double> pos_commands_;
  std::vector<double> vel_commands_;
  std::vector<double> tau_commands_;
  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> tau_states_;
};

}  // namespace openarm_zero_hw
