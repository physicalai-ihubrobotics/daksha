cmake_minimum_required(VERSION 3.8)
project(daksha_hardware)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── ROS2 dependencies ──────────────────────────────────────────────────────────
find_package(ament_cmake    REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib       REQUIRED)
find_package(rclcpp          REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(sensor_msgs     REQUIRED)   # for IMU subscription
find_package(std_msgs        REQUIRED)   # for Bool / Float32 subscriptions
find_package(std_srvs        REQUIRED)

# ── Library ────────────────────────────────────────────────────────────────────
add_library(
  ${PROJECT_NAME}
  SHARED
  src/drax_hardware.cpp
  src/socket_can.cpp
  src/canopen_node.cpp
  src/motor.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  ${PROJECT_NAME}
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  diagnostic_updater
  sensor_msgs
  std_msgs
  std_srvs
)

# ── Plugin export ──────────────────────────────────────────────────────────────
pluginlib_export_plugin_description_file(
  hardware_interface
  plugin.xml
)

# ── Install ────────────────────────────────────────────────────────────────────
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

# hardware.launch.py (hardware only; daksha_bringup/bringup.launch.py = this, then Nav2)
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.bak" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

# Python nodes (ros2 run daksha_hardware brake_push_mode_node.py)
install(
  PROGRAMS script/brake_push_mode_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Ament export ───────────────────────────────────────────────────────────────
ament_export_dependencies(
    hardware_interface
    pluginlib
    rclcpp
    rclcpp_lifecycle
    diagnostic_updater
    sensor_msgs
    std_msgs
    std_srvs
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

ament_package()
