// Copyright 2026 OpenArm 实验 / Apache-2.0
// MPC 控制器插件实现。行为规格: docs/MPC控制器插件_线路一改造设计.md §四
// 与旧 mpc_exec.py 的行为对应(重要差异 = hold 语义 + 偏差保护):
//   - 模型状态只在 goal 起始从测量初始化, 之后按模型推进(MIT 内环管实际跟踪)
//   - watchdog TRIPPED/RETURNING 或 estop → abort + hold(旧版: 停流回落 JTC)
//   - 结束后持续 hold 终点指令(goal_time=0 语义)
#include "openarm_mpc_controller/mpc_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include "pluginlib/class_list_macros.hpp"
#include "yaml-cpp/yaml.h"

namespace openarm_mpc_controller
{

namespace
{
constexpr double RAD2DEG = 180.0 / M_PI;
}

controller_interface::CallbackReturn OpenArmMpcController::on_init()
{
  RCLCPP_INFO(get_node()->get_logger(),
              "OpenArmMpcController (MPC 接管 JTC 槽位, 线路一插件化)");

  auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
  auto_declare<int>("N", 25);
  auto_declare<double>("dt", 0.01);
  auto_declare<double>("w_p", 1000.0);
  auto_declare<double>("w_v", 200.0);
  auto_declare<double>("w_a", 1.0);
  auto_declare<double>("a_max", 1.2);
  auto_declare<double>("speed_scale", 1.0);
  auto_declare<double>("clearance_deg", 3.0);
  auto_declare<double>("start_tolerance_rad", 0.035);
  auto_declare<double>("goal_tolerance_rad", 0.20);
  auto_declare<double>("abort_deviation_rad", 0.40);
  auto_declare<std::string>("safety_yaml", "");

  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "参数 joints 为空, 拒绝初始化");
    return controller_interface::CallbackReturn::ERROR;
  }
  n_joints_ = static_cast<int>(joint_names_.size());

  N_ = static_cast<int>(get_node()->get_parameter("N").as_int());
  dt_ = get_node()->get_parameter("dt").as_double();
  w_p_ = get_node()->get_parameter("w_p").as_double();
  w_v_ = get_node()->get_parameter("w_v").as_double();
  w_a_ = get_node()->get_parameter("w_a").as_double();
  a_max_ = get_node()->get_parameter("a_max").as_double();
  speed_scale_ = get_node()->get_parameter("speed_scale").as_double();
  clearance_deg_ = get_node()->get_parameter("clearance_deg").as_double();
  start_tol_ = get_node()->get_parameter("start_tolerance_rad").as_double();
  goal_tol_ = get_node()->get_parameter("goal_tolerance_rad").as_double();
  abort_dev_ = get_node()->get_parameter("abort_deviation_rad").as_double();
  safety_yaml_ = get_node()->get_parameter("safety_yaml").as_string();

  if (N_ < 2 || dt_ <= 0.0 || speed_scale_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "参数非法: N=%d dt=%g speed_scale=%g", N_, dt_, speed_scale_);
    return controller_interface::CallbackReturn::ERROR;
  }
  // 模型推进周期必须等于控制周期, 否则时间轴错位(部署固定 100Hz)
  const double dt_cm = 1.0 / static_cast<double>(get_update_rate());
  if (std::abs(dt_cm - dt_) > 1e-9) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "dt(%g) 与 controller_manager 周期(%g, %u Hz)不一致",
                 dt_, dt_cm, get_update_rate());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
OpenArmMpcController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & jn : joint_names_) {
    conf.names.push_back(jn + "/position");
  }
  return conf;
}

controller_interface::InterfaceConfiguration
OpenArmMpcController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & jn : joint_names_) {
    conf.names.push_back(jn + "/position");
    conf.names.push_back(jn + "/velocity");
  }
  return conf;
}

controller_interface::CallbackReturn OpenArmMpcController::on_configure(
  const rclcpp_lifecycle::State &)
{
  // 增益预计算(一次性, 非 RT 线程)
  mpc_ = std::make_unique<PreviewMPC>(n_joints_, N_, dt_, w_p_, w_v_, w_a_);
  RCLCPP_INFO(get_node()->get_logger(),
              "预览 MPC 增益就绪: N=%d dt=%.1fms w=(%.0f,%.0f,%.1f) a_max=%.2f",
              N_, dt_ * 1000.0, w_p_, w_v_, w_a_, a_max_);

  // 净空盒 = real_safety.yaml 软限位 ± clearance_deg(单一权威来源, 直读)
  lo_box_.assign(static_cast<size_t>(n_joints_), 0.0);
  hi_box_.assign(static_cast<size_t>(n_joints_), 0.0);
  ref_win_.assign(static_cast<size_t>(N_) + 1, 0.0);
  if (safety_yaml_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "参数 safety_yaml 为空");
    return controller_interface::CallbackReturn::ERROR;
  }
  try {
    YAML::Node cfg = YAML::LoadFile(safety_yaml_);
    const YAML::Node lim = cfg["soft_limits_rad"];
    for (int i = 0; i < n_joints_; ++i) {
      const std::string & jn = joint_names_[static_cast<size_t>(i)];
      lo_box_[static_cast<size_t>(i)] =
        lim[jn][0].as<double>() + clearance_deg_ / RAD2DEG;
      hi_box_[static_cast<size_t>(i)] =
        lim[jn][1].as<double>() - clearance_deg_ / RAD2DEG;
      RCLCPP_INFO(get_node()->get_logger(), "净空盒 %s: [%.4f, %.4f] rad",
                  jn.c_str(), lo_box_[static_cast<size_t>(i)],
                  hi_box_[static_cast<size_t>(i)]);
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "读 real_safety.yaml(%s)失败: %s", safety_yaml_.c_str(),
                 e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  q_model_.assign(static_cast<size_t>(n_joints_), 0.0);
  v_model_.assign(static_cast<size_t>(n_joints_), 0.0);
  q_cmd_.assign(static_cast<size_t>(n_joints_), 0.0);
  meas_pos_.assign(static_cast<size_t>(n_joints_), 0.0);
  meas_vel_.assign(static_cast<size_t>(n_joints_), 0.0);
  traj_box_.initRT(TrajSlot{});
  gh_box_.initRT(GhSlot{});

  // watchdog / estop(订阅在生命周期节点上, 回调跑 executor 线程)
  sub_safety_ = get_node()->create_subscription<std_msgs::msg::String>(
    "/safety/status", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      // 与 mpc_exec 相同口径: 仅 TRIPPED/RETURNING 触发中止
      const bool bad = (msg->data == "TRIPPED") || (msg->data == "RETURNING");
      watchdog_state_.store(bad ? 1 : 0);
    });
  sub_estop_ = get_node()->create_subscription<std_msgs::msg::Bool>(
    "/safety/estop", 10,
    [this](const std_msgs::msg::Bool::SharedPtr msg) {
      if (msg->data) {
        estop_latch_.store(true);
        RCLCPP_ERROR(get_node()->get_logger(),
                     "收到 estop, 挂起(拒绝新 goal; 重启控制器清除)");
      }
    });

  // action: ~ → /<槽位名>/follow_joint_trajectory(MoveIt/脚本零修改)
  using namespace std::placeholders;
  action_server_ = rclcpp_action::create_server<FollowJTrajAction>(
    get_node()->get_node_base_interface(),
    get_node()->get_node_clock_interface(),
    get_node()->get_node_logging_interface(),
    get_node()->get_node_waitables_interface(), "~/follow_joint_trajectory",
    std::bind(&OpenArmMpcController::on_goal, this, _1, _2),
    std::bind(&OpenArmMpcController::on_cancel, this, _1),
    std::bind(&OpenArmMpcController::on_accepted, this, _1));

  RCLCPP_INFO(get_node()->get_logger(),
              "已配置: action=~/%s safety_yaml=%s", "follow_joint_trajectory",
              safety_yaml_.c_str());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn OpenArmMpcController::on_activate(
  const rclcpp_lifecycle::State &)
{
  // 校验 loaned 接口顺序与关节序一致(防御性, 不匹配拒绝激活)
  for (int i = 0; i < n_joints_; ++i) {
    const size_t k = static_cast<size_t>(i);
    const auto & ci = command_interfaces_[k];
    const auto & sp = state_interfaces_[2 * k];
    const auto & sv = state_interfaces_[2 * k + 1];
    if (ci.get_prefix_name() != joint_names_[k] ||
      ci.get_interface_name() != "position" ||
      sp.get_prefix_name() != joint_names_[k] ||
      sp.get_interface_name() != "position" ||
      sv.get_prefix_name() != joint_names_[k] ||
      sv.get_interface_name() != "velocity") {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "第 %d 个接口与 %s 不匹配(指令/状态序错乱)", i,
                   joint_names_[k].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  // 激活即从测量初始化 hold(与 ZeroOffsetHW 锁位语义衔接, 无跳变)
  traj_.reset();
  traj_running_ = false;
  traj_duration_ = 0.0;
  traj_gen_ = 0;
  next_gen_ = 1;
  // 清空交接盒, 防止重新激活后采纳上次会话的旧轨迹
  traj_box_.writeFromNonRT(TrajSlot{});
  gh_box_.writeFromNonRT(GhSlot{});
  estop_latch_.store(false);
  for (int i = 0; i < n_joints_; ++i) {
    const size_t k = static_cast<size_t>(i);
    meas_pos_[k] = state_interfaces_[2 * k].get_value();
    meas_vel_[k] = state_interfaces_[2 * k + 1].get_value();
    q_model_[k] = meas_pos_[k];
    v_model_[k] = meas_vel_[k];
    q_cmd_[k] = meas_pos_[k];
    (void)command_interfaces_[k].set_value(q_cmd_[k]);
  }
  {
    std::lock_guard<std::mutex> lock(meas_mutex_);
    meas_cache_ = Meas{meas_pos_, meas_vel_, true};
  }
  RCLCPP_INFO(get_node()->get_logger(), "已激活: 从当前测量 hold");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn OpenArmMpcController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  finish_goal_(false, "控制器失活");
  traj_.reset();
  traj_running_ = false;
  return controller_interface::CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------- action

rclcpp_action::GoalResponse OpenArmMpcController::on_goal(
  const rclcpp_action::GoalUUID &,
  std::shared_ptr<const FollowJTrajAction::Goal> goal)
{
  if (estop_latch_.load()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "拒绝 goal: estop 挂起(重启控制器清除)");
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (goal_busy_.load()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "拒绝 goal: 已有活跃 goal(并发不支持)");
    return rclcpp_action::GoalResponse::REJECT;
  }

  TrajPtr traj;
  std::string why;
  if (!prepare_traj(*goal, traj, why)) {
    RCLCPP_ERROR(get_node()->get_logger(), "拒绝 goal: %s", why.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }
  pending_traj_ = std::move(traj);
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse OpenArmMpcController::on_cancel(
  const std::shared_ptr<GoalHandle> gh)
{
  // rclcpp_action 无 setCancelRequested: 接受后中间件把 goal 置 canceling,
  // RT 侧经 cancel_req_ 走 setCanceled 终结。
  if (goal_busy_.load() && gh) {
    cancel_req_.store(true);
    return rclcpp_action::CancelResponse::ACCEPT;
  }
  RCLCPP_INFO(get_node()->get_logger(),
              "拒绝取消请求: 无活跃 goal");
  return rclcpp_action::CancelResponse::REJECT;
}

void OpenArmMpcController::on_accepted(const std::shared_ptr<GoalHandle> gh)
{
  auto handle = gh;  // 构造函数签名要非 const 左值引用
  auto rt_gh = std::make_shared<RtGoalHandle>(handle);
  rt_gh->execute();  // 排队"已执行"; 真正的响应由 update 侧 runNonRealtime 冲发
  goal_busy_.store(true);
  const uint64_t gen = next_gen_++;  // 纯非 RT 计数, 不读 RT 侧 traj_gen_
  gh_box_.writeFromNonRT(GhSlot{std::move(rt_gh), gen});
  traj_box_.writeFromNonRT(TrajSlot{std::move(pending_traj_), gen});
  RCLCPP_INFO(get_node()->get_logger(),
              "goal 已接受(gen=%lu), MPC 跟踪即将开始",
              static_cast<unsigned long>(gen));
}

bool OpenArmMpcController::prepare_traj(
  const FollowJTrajAction::Goal & goal, TrajPtr & out, std::string & why) const
{
  const auto & names = goal.trajectory.joint_names;
  if (names.size() != joint_names_.size()) {
    why = "关节数 " + std::to_string(names.size()) + " != " +
      std::to_string(joint_names_.size());
    return false;
  }
  // goal 关节序 → 内部关节序映射
  std::vector<size_t> order(names.size(), 0);
  for (size_t g = 0; g < names.size(); ++g) {
    auto it = std::find(joint_names_.begin(), joint_names_.end(), names[g]);
    if (it == joint_names_.end()) {
      why = "未知关节 " + names[g];
      return false;
    }
    order[g] = static_cast<size_t>(it - joint_names_.begin());
  }

  const auto & pts = goal.trajectory.points;
  if (pts.empty()) {
    why = "轨迹为空";
    return false;
  }
  std::vector<double> times(pts.size(), 0.0);
  for (size_t p = 0; p < pts.size(); ++p) {
    const double t = static_cast<double>(pts[p].time_from_start.sec) +
      1e-9 * static_cast<double>(pts[p].time_from_start.nanosec);
    if (p > 0 && t < times[p - 1]) {
      why = "时间轴在第 " + std::to_string(p) + " 点回退";
      return false;
    }
    times[p] = t;
    if (pts[p].positions.size() != names.size()) {
      why = "第 " + std::to_string(p) + " 点位置维数不符";
      return false;
    }
  }
  // 首点校验: 与当前测量偏差超限则拒绝(防跳变; mpc_exec 同口径 2°)
  Meas m;
  {
    std::lock_guard<std::mutex> lock(meas_mutex_);
    m = meas_cache_;
  }
  if (!m.valid) {
    why = "尚无状态测量(激活后等一个控制周期再发 goal)";
    return false;
  }
  for (size_t g = 0; g < names.size(); ++g) {
    const size_t k = order[g];
    const double dev = std::abs(pts[0].positions[g] - m.pos[k]);
    if (dev > start_tol_) {
      char buf[256];
      snprintf(buf, sizeof(buf), "起点偏差 %s %.2f° > %.2f°, 先回 home",
               names[g].c_str(), RAD2DEG * dev, RAD2DEG * start_tol_);
      why = buf;
      return false;
    }
  }

  auto traj = std::make_shared<Traj>();
  traj->times = std::move(times);
  traj->points.reserve(pts.size());
  for (const auto & p : pts) {
    std::vector<double> row(static_cast<size_t>(n_joints_), 0.0);
    for (size_t g = 0; g < order.size(); ++g) {
      row[order[g]] = p.positions[g];
    }
    traj->points.push_back(std::move(row));
  }
  traj->duration = traj->times.back();
  if (!std::isfinite(traj->duration) || traj->duration <= 0.0) {
    why = "轨迹时长非法";
    return false;
  }
  out = std::move(traj);
  return true;
}

// ---------------------------------------------------------------- goal 终结

void OpenArmMpcController::finish_goal_(bool success, const std::string & reason)
{
  const GhSlot slot = *gh_box_.readFromRT();
  if (!slot.gh || !slot.gh->valid()) {
    goal_busy_.store(false);
    return;
  }
  auto result = std::make_shared<FollowJTrajAction::Result>();
  if (success) {
    result->error_code = FollowJTrajAction::Result::SUCCESSFUL;
    slot.gh->setSucceeded(result);
  } else {
    // FollowJointTrajectory 无 ABORTED 码; 终态 ABORTED 由 gh->abort 决定,
    // error_code 取容差类(保守对齐旧 JTC 容差中止语义)
    result->error_code = FollowJTrajAction::Result::PATH_TOLERANCE_VIOLATED;
    result->error_string = reason;
    slot.gh->setAborted(result);
  }
  slot.gh->runNonRealtime();  // 立即冲发(此处在 update/on_deactivate, 可容忍)
  goal_busy_.store(false);
}

void OpenArmMpcController::flush_goal_()
{
  const GhSlot slot = *gh_box_.readFromRT();
  if (slot.gh && slot.gh->valid()) {
    slot.gh->runNonRealtime();
  }
}

// ---------------------------------------------------------------- update

double OpenArmMpcController::sample_at_(
  const Traj & t, size_t j, double tt) const
{
  if (tt <= 0.0) {
    return t.points.front()[j];
  }
  if (tt >= t.duration) {
    return t.points.back()[j];
  }
  auto it = std::upper_bound(t.times.begin(), t.times.end(), tt);
  const size_t i = static_cast<size_t>(it - t.times.begin()) - 1;
  const double t0 = t.times[i];
  const double t1 = t.times[i + 1];
  const double f = (t1 > t0) ? (tt - t0) / (t1 - t0) : 0.0;
  return t.points[i][j] + f * (t.points[i + 1][j] - t.points[i][j]);
}

controller_interface::return_type OpenArmMpcController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  const size_t n = static_cast<size_t>(n_joints_);

  // ① 读硬件状态(博客线路一 update() 第一步) → 缓存供起点校验
  for (size_t k = 0; k < n; ++k) {
    meas_pos_[k] = state_interfaces_[2 * k].get_value();
    meas_vel_[k] = state_interfaces_[2 * k + 1].get_value();
  }
  {
    std::lock_guard<std::mutex> lock(meas_mutex_);
    meas_cache_.pos = meas_pos_;
    meas_cache_.vel = meas_vel_;
    meas_cache_.valid = true;
  }
  flush_goal_();  // goal 响应/结果回发(RealtimeServerGoalHandle 要求周期调用)

  // 采纳新 goal(gen 比较, 防完结轨迹被再采纳)
  const GhSlot slot = *gh_box_.readFromRT();
  const TrajSlot tslot = *traj_box_.readFromRT();
  if (tslot.traj && tslot.gen != traj_gen_) {
    traj_ = tslot.traj;
    traj_duration_ = traj_->duration;
    traj_start_ = time;
    traj_running_ = true;
    traj_gen_ = tslot.gen;
    for (size_t k = 0; k < n; ++k) {  // 模型状态从测量重新初始化(mpc_exec 同口径)
      q_model_[k] = meas_pos_[k];
      v_model_[k] = meas_vel_[k];
    }
    RCLCPP_INFO(get_node()->get_logger(),
                "MPC 跟踪: %zu 点, 时长 %.1fs, speed_scale=%.2f",
                traj_->points.size(), traj_duration_, speed_scale_);
  }
  const auto gh = slot.gh;
  const bool gh_current = traj_running_ && gh && slot.gen == traj_gen_;

  // ② 安全链: watchdog / estop / cancel → abort + hold
  if (traj_running_ &&
    (watchdog_state_.load() != 0 || estop_latch_.load() ||
      cancel_req_.exchange(false)))
  {
    const char * why = watchdog_state_.load() != 0 ? "watchdog 触发"
                     : (estop_latch_.load() ? "estop" : "被取消");
    if (gh_current) {
      finish_goal_(false, why);
    }
    traj_.reset();
    traj_running_ = false;
    RCLCPP_ERROR(get_node()->get_logger(), "中止 MPC 跟踪: %s(指令 hold)", why);
  }

  // 无活跃轨迹: hold 最后指令(含 goal 结束后)
  if (!traj_running_ || !traj_) {
    for (size_t k = 0; k < n; ++k) {
      (void)command_interfaces_[k].set_value(q_cmd_[k]);
    }
    return controller_interface::return_type::OK;
  }

  // ③ 参考窗插值 + 闭式 MPC + 模型推进(博客线路一第二/三/四步)
  const double t_ref =
    (time - traj_start_).seconds() / (speed_scale_ > 0.0 ? speed_scale_ : 1.0);
  const double h = period.seconds();
  for (size_t k = 0; k < n; ++k) {
    for (int j = 0; j <= N_; ++j) {
      ref_win_[static_cast<size_t>(j)] =
        sample_at_(*traj_, k, t_ref + static_cast<double>(j) * dt_);
    }
    const double a = mpc_->step(
      static_cast<int>(k), q_model_[k], v_model_[k], ref_win_.data(), a_max_);
    v_model_[k] += h * a;
    q_model_[k] += h * v_model_[k] + 0.5 * h * h * a;
    q_cmd_[k] = std::max(lo_box_[k], std::min(hi_box_[k], q_model_[k]));
    (void)command_interfaces_[k].set_value(q_cmd_[k]);
  }

  // 保护: 测量与指令偏差超限(替代 JTC trajectory 容差监视)
  double dev = 0.0;
  for (size_t k = 0; k < n; ++k) {
    dev = std::max(dev, std::abs(meas_pos_[k] - q_cmd_[k]));
  }
  if (dev > abort_dev_) {
    if (gh_current) {
      finish_goal_(false, "跟踪偏差超限");
    }
    traj_.reset();
    traj_running_ = false;
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "中止 MPC 跟踪: 测量-指令偏差 %.2f° > %.2f°(hold)", RAD2DEG * dev,
      RAD2DEG * abort_dev_);
    return controller_interface::return_type::OK;
  }

  // ④ 结束判定: 参考走完后向终点收敛, 测量全关节数入容差 → SUCCEED
  if (t_ref >= traj_duration_ && gh_current) {
    bool ok = true;
    for (size_t k = 0; k < n; ++k) {
      if (std::abs(meas_pos_[k] - traj_->points.back()[k]) > goal_tol_) {
        ok = false;
        break;
      }
    }
    if (ok) {
      finish_goal_(true, "");
      traj_.reset();
      traj_running_ = false;
      RCLCPP_INFO(get_node()->get_logger(), "MPC 轨迹完成, 指令 hold 终点");
    }
  }
  return controller_interface::return_type::OK;
}

}  // namespace openarm_mpc_controller

PLUGINLIB_EXPORT_CLASS(openarm_mpc_controller::OpenArmMpcController,
                       controller_interface::ControllerInterface)
