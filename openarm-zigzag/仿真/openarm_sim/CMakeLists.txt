cmake_minimum_required(VERSION 3.8)
project(openarm_sim)

# 编译器默认设置（无 C++ 代码，纯 Python 节点包）
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# 将 Python 包安装到 site-packages，保证节点脚本可以 import 同包模块
ament_python_install_package(${PROJECT_NAME})

# 节点启动脚本
install(PROGRAMS
  scripts/scene_node
  scripts/joint_console
  DESTINATION lib/${PROJECT_NAME}
)

# launch 与参数配置
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
