cmake_minimum_required(VERSION 3.22)
project(openarm_mpc_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# 数学核(独立库, 供增益一致性验收工具复用)
add_library(preview_mpc SHARED src/preview_mpc.cpp)
target_include_directories(preview_mpc PUBLIC include)
target_link_libraries(preview_mpc Eigen3::Eigen)
target_compile_features(preview_mpc PUBLIC cxx_std_17)

# 控制器插件
add_library(${PROJECT_NAME} SHARED src/mpc_controller.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} preview_mpc yaml-cpp::yaml-cpp)
ament_target_dependencies(${PROJECT_NAME}
  control_msgs
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  trajectory_msgs
)
pluginlib_export_plugin_description_file(controller_interface openarm_mpc_controller.xml)

# 增益一致性验收: 导出 C++ 增益, 与 mpc_core.py 的 PreviewMPC.K 比对
add_executable(mpc_gain_dump tools/mpc_gain_dump.cpp)
target_link_libraries(mpc_gain_dump preview_mpc)

install(TARGETS preview_mpc ${PROJECT_NAME}
  LIBRARY DESTINATION lib
)
install(TARGETS mpc_gain_dump
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(preview_mpc ${PROJECT_NAME})
ament_export_dependencies(controller_interface hardware_interface rclcpp)

ament_package()
