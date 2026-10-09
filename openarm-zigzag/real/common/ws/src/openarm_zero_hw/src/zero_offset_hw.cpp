// Copyright 2025 Enactic, Inc. / Apache-2.0
// 基于官方 openarm_hardware/openarm_simple_hardware.cpp 修改, 修改点见头文件注释。
// 与官方实现的关键行为差异:
//   read():  pos = motor - d          (官方: pos = motor)
//   write(): mit_target = cmd + d     (官方: mit_target = cmd)
//   on_activate(): 使能后锁定当前位置 (官方: 使能后 2s 插值回电机零位——危险)
//   夹爪: hand=hold 时第 8 电机(ESC8=夹爪 DM4310, send 0x08/recv 0x18)加入
//         电机表, 使能后原地锁定(hand_kp/hand_kd), 不暴露 ROS 命令接口;
//         hand=false 时不接触 ESC8

#include "openarm_zero_hw/zero_offset_hw.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

namespace openarm_zero_hw {

ZeroOffsetHW::ZeroOffsetHW() = default;

hardware_interface::CallbackReturn ZeroOffsetHW::on_init(
    const hardware_interface::HardwareInfo& info) {
  if (hardware_interface::SystemInterface::on_init(info) !=
      CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  auto get_param = [&](const std::string& key, const std::string& def) {
    auto it = info.hardware_parameters.find(key);
    return it != info.hardware_parameters.end() ? it->second : def;
  };
  auto to_bool = [](const std::string& v) {
    std::string s = v;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "true";
  };

  can_interface_ = get_param("can_interface", "can0");
  arm_prefix_ = get_param("arm_prefix", "left_");
  can_fd_ = to_bool(get_param("can_fd", "true"));

  const std::string def_kp = "70,70,70,60,10,10,10";
  const std::string def_kd = "2.75,2.5,2.0,2.0,0.7,0.6,0.5";
  auto parse_list = [](const std::string& s, size_t n) {
    std::vector<double> out(n, 0.0);
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t comma = s.find(',', start);
      out[i] = std::stod(s.substr(start, comma - start));
      start = (comma == std::string::npos) ? s.size() : comma + 1;
    }
    return out;
  };
  // 增益三级回退: ① gen_real_config.py 写入的逐关节 kp1..kp7/kd1..kd7
  // (标准接口); ② 逗号列表 kp/kd; ③ 官方默认。逐关节参数必须 7 个齐全
  // 才算命中, 防止部分缺失造成静默错位。
  auto read_gain = [&](const std::string& base, const std::string& def_list) {
    std::vector<double> out;
    bool all_single = true;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      auto it = info.hardware_parameters.find(base + std::to_string(i + 1));
      if (it == info.hardware_parameters.end()) {
        all_single = false;
        break;
      }
      out.push_back(std::stod(it->second));
    }
    if (all_single) {
      RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                  "增益 %s: 逐关节 %s1..%s%zu 命中", base.c_str(),
                  base.c_str(), base.c_str(), ARM_DOF);
      return out;
    }
    RCLCPP_WARN(rclcpp::get_logger("ZeroOffsetHW"),
                "增益 %s: 逐关节参数不全, 回退 %s 列表/内置默认",
                base.c_str(), base.c_str());
    return parse_list(get_param(base, def_list), ARM_DOF);
  };
  kp_ = read_gain("kp", def_kp);
  kd_ = read_gain("kd", def_kd);

  // 软件积分器(重力补偿): tau_ff = ki * ∫(目标-实际)dt, 电机系叠加在
  // MIT tau 上。钳位限制补偿力矩上限; 误差>0.5rad 时清零防积分饱和;
  // ki=0 即关闭(退化为纯 MIT 位置控制)。
  // 2026-09-04 调优: ki J1/J2 24→12, 抑制 2Hz 肩部极限环抖动(实测数据)。
  integ_ = std::vector<double>(ARM_DOF, 0.0);
  err_f_ = std::vector<double>(ARM_DOF, 0.0);
  ki_ = parse_list(get_param("ki", "12,12,12,10,8,8,8"), ARM_DOF);
  integ_clamp_ = parse_list(get_param("integ_clamp", "10,10,6,8,3,3,3"),
                            ARM_DOF);

  // 通道映射(默认恒等 1..7): channel_map = 每个_urdf_关节对应的电机
  // ESC 序号(1 基, 必须为不重复排列); channel_sign = 每通道 ±1, 电机
  // 正方向与模型关节正方向的关系。肩部双电机装配顺序与模型相反的
  // 整机, 由 gen_real_config.py 按实测调查写入实际映射。
  channel_perm_.resize(ARM_DOF);
  channel_sign_ = parse_list(get_param("channel_sign", "1,1,1,1,1,1,1"),
                             ARM_DOF);
  {
    std::vector<double> map1 =
        parse_list(get_param("channel_map", "1,2,3,4,5,6,7"), ARM_DOF);
    std::vector<bool> used(ARM_DOF, false);
    for (size_t i = 0; i < ARM_DOF; ++i) {
      const int ch = static_cast<int>(map1[i]);
      if (ch < 1 || ch > static_cast<int>(ARM_DOF) || used[ch - 1]) {
        RCLCPP_ERROR(rclcpp::get_logger("ZeroOffsetHW"),
                     "channel_map 非法: 必须是 1..%zu 的不重复排列",
                     ARM_DOF);
        return CallbackReturn::ERROR;
      }
      if (channel_sign_[i] != 1.0 && channel_sign_[i] != -1.0) {
        RCLCPP_ERROR(rclcpp::get_logger("ZeroOffsetHW"),
                     "channel_sign 只允许 ±1");
        return CallbackReturn::ERROR;
      }
      used[ch - 1] = true;
      channel_perm_[i] = static_cast<size_t>(ch - 1);
    }
    std::string map_str;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      map_str += std::to_string(channel_perm_[i] + 1);
      if (i + 1 < ARM_DOF) map_str += ",";
    }
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                "通道映射(ESC 序号按 URDF 关节序): [%s]", map_str.c_str());
  }

  // d1..d7 必须逐个显式提供(缺任何一个都是配置错误, 拒绝启动)
  d_.resize(ARM_DOF);
  for (size_t i = 0; i < ARM_DOF; ++i) {
    auto it = info.hardware_parameters.find("d" + std::to_string(i + 1));
    if (it == info.hardware_parameters.end()) {
      RCLCPP_ERROR(rclcpp::get_logger("ZeroOffsetHW"),
                   "缺少参数 d%zu (zero 偏移), 真机配置必须由 "
                   "gen_real_config.py 生成", i + 1);
      return CallbackReturn::ERROR;
    }
    d_[i] = std::stod(it->second);
  }

  // hand: false=不接触 ESC8(旧版行为); true/hold=第 8 电机(ESC8=夹爪
  // DM4310, send 0x08/recv 0x18)使能并原地锁定, 增益默认取标定
  // grip_hold_kp/kd(10.0/0.9, 标定程序实测值)。不暴露 ROS 命令接口,
  // JTC/MoveIt 仍只管 7 关节。
  const std::string hand_v = get_param("hand", "false");
  if (hand_v == "true" || hand_v == "hold") {
    hand_hold_ = true;
    hand_kp_ = std::stod(get_param("hand_kp", "10.0"));
    hand_kd_ = std::stod(get_param("hand_kd", "0.9"));
  } else if (hand_v != "false") {
    RCLCPP_ERROR(rclcpp::get_logger("ZeroOffsetHW"),
                 "hand 参数只支持 true/hold/false, 收到: %s",
                 hand_v.c_str());
    return CallbackReturn::ERROR;
  }

  joint_names_.clear();
  for (size_t i = 1; i <= ARM_DOF; ++i) {
    joint_names_.push_back("openarm_" + arm_prefix_ + "joint" +
                           std::to_string(i));
  }
  size_t expected = ARM_DOF;
  if (info.joints.size() != expected) {
    RCLCPP_ERROR(rclcpp::get_logger("ZeroOffsetHW"),
                 "URDF 中有 %zu 个 ros2_control 关节, 期望 %zu",
                 info.joints.size(), expected);
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
              "CAN=%s can_fd=%s prefix=%s", can_interface_.c_str(),
              can_fd_ ? "on" : "off", arm_prefix_.c_str());
  {
    std::string ki_s, cl_s;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      ki_s += std::to_string(ki_[i]) + (i + 1 < ARM_DOF ? "," : "");
      cl_s += std::to_string(integ_clamp_[i]) + (i + 1 < ARM_DOF ? "," : "");
    }
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                "积分器 ki=[%s] clamp=[%s] Nm", ki_s.c_str(), cl_s.c_str());
  }
  for (size_t i = 0; i < ARM_DOF; ++i) {
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                "%s: d=%.4f rad kp=%.2f kd=%.3f", joint_names_[i].c_str(),
                d_[i], kp_[i], kd_[i]);
  }

  openarm_ =
      std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);
  auto motor_types = MOTOR_TYPES_;
  auto send_ids = SEND_CAN_IDS_;
  auto recv_ids = RECV_CAN_IDS_;
  if (hand_hold_) {
    motor_types.push_back(openarm::damiao_motor::MotorType::DM4310);
    send_ids.push_back(0x08);
    recv_ids.push_back(0x18);
  }
  openarm_->init_arm_motors(motor_types, send_ids, recv_ids);

  pos_commands_.resize(ARM_DOF, 0.0);
  vel_commands_.resize(ARM_DOF, 0.0);
  tau_commands_.resize(ARM_DOF, 0.0);
  pos_states_.resize(ARM_DOF, 0.0);
  vel_states_.resize(ARM_DOF, 0.0);
  tau_states_.resize(ARM_DOF, 0.0);

  // 注入通道(2026-09-18 完全分离 + 无残留): 单话题(方案由 mpc_topic
  // 参数决定, gen_real_config --scheme 写入; verify_deploy 审计),
  // 上升沿以当前实际位置重置基线 → 上会话残留自动清除。
  inj_.cmd.assign(ARM_DOF, 0.0);
  inj_.vel.assign(ARM_DOF, 0.0);
  inj_.ff.assign(ARM_DOF, 0.0);
  ff_clamp_ = parse_list(get_param("ff_clamp", "15,15,20,20,5,5,5"), ARM_DOF);
  vel_clamp_ = parse_list(get_param("vel_clamp",
                                    "1.5,1.5,1.5,1.5,1.5,1.5,1.5"),
                          ARM_DOF);
  last_cmd_urdf_.assign(ARM_DOF, 0.0);
  fallback_slew_ = std::stod(get_param("fallback_slew", "0.25"));
  inj_fresh_s_ = std::stod(get_param("inj_fresh_s", "0.2"));
  if (to_bool(get_param("mpc_enable", "true"))) {
    mpc_node_ = rclcpp::Node::make_shared("zero_offset_hw_inj");
    const std::string inj_topic =
        get_param("mpc_topic", "/right_mpc_position_commands");
    inj_name_ = (inj_topic.find("_rl_") != std::string::npos) ? "RL"
                                                             : "MPC";
    inj_sub_ = mpc_node_->create_subscription<
        std_msgs::msg::Float64MultiArray>(
        inj_topic, 1,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
          const size_t n = msg->data.size();
          if (n != ARM_DOF && n != 2 * ARM_DOF && n != 3 * ARM_DOF) return;
          const bool has_vel = n == 3 * ARM_DOF;
          const bool has_ff = n >= 2 * ARM_DOF;
          for (size_t i = 0; i < ARM_DOF; ++i) {
            double v = std::max(-6.5, std::min(6.5, msg->data[i]));
            // 速率钳位: 相邻消息最大 0.2 rad(防御异常发布器)
            v = std::max(inj_.cmd[i] - 0.2,
                         std::min(inj_.cmd[i] + 0.2, v));
            inj_.cmd[i] = v;
            if (has_vel) {
              double u = std::max(-vel_clamp_[i], std::min(
                  vel_clamp_[i], msg->data[ARM_DOF + i]));
              u = std::max(inj_.vel[i] - 0.1,
                           std::min(inj_.vel[i] + 0.1, u));
              inj_.vel[i] = u;
            } else {
              inj_.vel[i] = 0.0;
            }
            double f = has_ff ? msg->data[2 * ARM_DOF + i] : 0.0;
            f = std::max(-ff_clamp_[i], std::min(ff_clamp_[i], f));
            f = std::max(inj_.ff[i] - 0.5, std::min(inj_.ff[i] + 0.5, f));
            inj_.ff[i] = f;
          }
          inj_.stamp = mpc_node_->now();
          inj_.got.store(true);
        });
    mpc_exec_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    mpc_exec_->add_node(mpc_node_);
    mpc_spin_ = std::thread([this]() { mpc_exec_->spin(); });
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                "注入通道: %s [%s] (7/14/21 项兼容, 单源无残留)",
                inj_topic.c_str(), inj_name_.c_str());
  } else {
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                "注入通道: 关闭(mpc_enable=false, 纯 JTC 模式)");
  }

  RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"), "初始化成功");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ZeroOffsetHW::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // 只收帧刷新状态, 不使能电机
  openarm_->refresh_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
ZeroOffsetHW::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> out;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    out.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION,
                     &pos_states_[i]);
    out.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY,
                     &vel_states_[i]);
    out.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT,
                     &tau_states_[i]);
  }
  return out;
}

std::vector<hardware_interface::CommandInterface>
ZeroOffsetHW::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> out;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    out.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION,
                     &pos_commands_[i]);
  }
  return out;
}

hardware_interface::CallbackReturn ZeroOffsetHW::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
              "使能电机并锁定当前位置...");
  openarm_->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  openarm_->enable_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  // 读当前位形, 换算成 URDF 系后作为命令 → write() 循环刚性保持原地,
  // 不做任何大动作(官方版本此处会回电机零位, 已移除)
  openarm_->refresh_all();
  openarm_->recv_all();
  const auto& motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < ARM_DOF && i < motors.size(); ++i) {
    const auto& m = motors[channel_perm_[i]];
    pos_states_[i] = channel_sign_[i] * m.get_position() - d_[i];
    vel_states_[i] = channel_sign_[i] * m.get_velocity();
    tau_states_[i] = channel_sign_[i] * m.get_torque();
    pos_commands_[i] = pos_states_[i];
    vel_commands_[i] = 0.0;
    tau_commands_[i] = 0.0;
  }
  RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"), "已锁定当前位置(URDF 系)");
  if (hand_hold_) {
    hand_q0_ = motors.back().get_position();
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"),
                "夹爪 ESC8 已使能锁定当前位置 q0=%.3f rad (kp=%.1f kd=%.2f)",
                hand_q0_, hand_kp_, hand_kd_);
  }
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ZeroOffsetHW::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"), "失能全部电机...");
  std::fill(integ_.begin(), integ_.end(), 0.0);   // 清积分, 防再使能跳变
  std::fill(err_f_.begin(), err_f_.end(), 0.0);
  for (int i = 0; i < 3; ++i) {
    openarm_->disable_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    openarm_->recv_all();
  }
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type ZeroOffsetHW::read(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  openarm_->refresh_all();
  openarm_->recv_all();
  const auto& motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < ARM_DOF && i < motors.size(); ++i) {
    const auto& m = motors[channel_perm_[i]];
    pos_states_[i] = channel_sign_[i] * m.get_position() - d_[i];
    vel_states_[i] = channel_sign_[i] * m.get_velocity();
    tau_states_[i] = channel_sign_[i] * m.get_torque();
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type ZeroOffsetHW::write(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  const double dt = period.seconds();
  const auto& motors = openarm_->get_arm().get_motors();
  // 注入源新鲜度(单话题: RL 或 MPC 由配置决定, 运行时不同时存在)
  const rclcpp::Time now = mpc_node_ ? mpc_node_->now()
                                     : rclcpp::Time(0, 0, RCL_ROS_TIME);
  const bool use_inj = mpc_node_ && inj_.got.load() &&
      (now - inj_.stamp).seconds() < inj_fresh_s_;
  if (use_inj && !inj_.active.load()) {
    // 上升沿: 基线重置为当前实际位置(上会话残留结构性清除)
    for (size_t i = 0; i < ARM_DOF; ++i) {
      inj_.cmd[i] = pos_states_[i];
      inj_.vel[i] = 0.0;
      inj_.ff[i] = 0.0;
    }
  }
  inj_.active.store(use_inj);
  // 流断 → 限速滑回 JTC(闭锁至追上; 看门狗回位轨迹经 JTC 无缝衔接)
  if (!use_inj && prev_use_inj_) fallback_active_ = true;
  if (use_inj) fallback_active_ = false;
  prev_use_inj_ = use_inj;
  if (fallback_active_) {
    double gap = 0.0;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      gap = std::max(gap, std::abs(pos_commands_[i] - last_cmd_urdf_[i]));
    }
    if (gap < 0.02) {
      fallback_active_ = false;    // 已追上 JTC 目标(<1.1°), 恢复直通
    }
  }
  const std::string src_name = use_inj ? inj_name_ : "JTC";
  if (src_name != last_src_) {
    RCLCPP_INFO(rclcpp::get_logger("ZeroOffsetHW"), "指令源: %s",
                src_name.c_str());
    last_src_ = src_name;
  }
  // urdf = s*motor - d  ⇒  电机 MIT 目标 = s*(urdf + d)
  // hand=hold 时第 8 项为夹爪: 以激活时位置 hand_q0_ 定点 MIT 锁定
  const size_t n_motors = ARM_DOF + (hand_hold_ ? 1 : 0);
  std::vector<openarm::damiao_motor::MITParam> params(n_motors);
  for (size_t i = 0; i < ARM_DOF; ++i) {
    const double s = channel_sign_[i];
    double cmd_i, vel_i, ff_i;
    if (use_inj) {
      cmd_i = inj_.cmd[i];
      vel_i = inj_.vel[i];
      ff_i = inj_.ff[i];
    } else {
      cmd_i = pos_commands_[i];
      vel_i = vel_commands_[i];
      ff_i = 0.0;
    }
    if (!use_inj && fallback_active_) {
      // 滑向 JTC 目标, 每周期最多 fallback_slew_×dt
      const double lo_s = last_cmd_urdf_[i] - fallback_slew_ * dt;
      const double hi_s = last_cmd_urdf_[i] + fallback_slew_ * dt;
      cmd_i = std::max(lo_s, std::min(hi_s, cmd_i));
    }
    last_cmd_urdf_[i] = cmd_i;
    const double target_m = s * (cmd_i + d_[i]);
    double tau_ff = 0.0;
    if (ki_[i] > 0.0) {
      const double err_m =
          target_m - motors[channel_perm_[i]].get_position();
      // EMA 平滑误差(α=0.3): 抑制积分器通路的高频抖动
      err_f_[i] = 0.3 * err_m + 0.7 * err_f_[i];
      if (std::abs(err_f_[i]) > 0.5) {
        integ_[i] = 0.0;   // 大偏差(换段/中止瞬间)清零防积分饱和
        err_f_[i] = 0.0;
      } else {
        integ_[i] += ki_[i] * err_f_[i] * dt;
        integ_[i] = std::max(-integ_clamp_[i],
                             std::min(integ_clamp_[i], integ_[i]));
      }
      tau_ff = integ_[i];
    }
    params[channel_perm_[i]] = {kp_[i], kd_[i], target_m,
                                s * vel_i,
                                s * tau_commands_[i] + tau_ff + s * ff_i};
  }
  if (hand_hold_) {
    params[ARM_DOF] = {hand_kp_, hand_kd_, hand_q0_, 0.0, 0.0};
  }
  openarm_->get_arm().mit_control_all(params);
  openarm_->recv_all(100);
  return hardware_interface::return_type::OK;
}

hardware_interface::CallbackReturn ZeroOffsetHW::on_shutdown(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (mpc_exec_) {
    mpc_exec_->cancel();
    if (mpc_spin_.joinable()) mpc_spin_.join();
    if (mpc_node_) mpc_exec_->remove_node(mpc_node_);
    inj_sub_.reset();
    mpc_node_.reset();
    mpc_exec_.reset();
  }
  return CallbackReturn::SUCCESS;
}

}  // namespace openarm_zero_hw

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(openarm_zero_hw::ZeroOffsetHW,
                       hardware_interface::SystemInterface)
