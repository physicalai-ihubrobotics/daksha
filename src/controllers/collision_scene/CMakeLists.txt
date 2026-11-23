cmake_minimum_required(VERSION 3.16)
project(collision_scene)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Discover cmeel prefix from Python pinocchio install
execute_process(
  COMMAND python3 -c
    "import pinocchio, os; p = os.path.dirname(pinocchio.__file__); \
     # Walk up from .../cmeel.prefix/lib/python3.x/site-packages/pinocchio \
     # to the cmeel.prefix directory (5 levels up). \
     for _ in range(5): p = os.path.dirname(p); \
     print(p)"
  OUTPUT_VARIABLE CMEEL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "cmeel prefix (auto): ${CMEEL_PREFIX}")

# Verify the config exists; fall back to known path if not
if(NOT EXISTS "${CMEEL_PREFIX}/lib/cmake/pinocchio/pinocchioConfig.cmake")
  set(CMEEL_PREFIX "/home/ihub/.local/lib/python3.12/site-packages/cmeel.prefix")
  message(STATUS "cmeel prefix (fallback): ${CMEEL_PREFIX}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${CMEEL_PREFIX}")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pinocchio REQUIRED)
find_package(coal REQUIRED)
find_package(urdf REQUIRED)

find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
endif()

# Keep launch files and config
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

add_executable(collision_scene_node
  src/acm.cpp
  src/collision_robot.cpp
  src/collision_world.cpp
  src/planning_scene.cpp
  src/collision_scene_node.cpp
)

target_include_directories(collision_scene_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(collision_scene_node
  rclcpp std_msgs sensor_msgs geometry_msgs visualization_msgs
  ament_index_cpp urdf)

if(yaml-cpp_FOUND)
  target_link_libraries(collision_scene_node
    pinocchio::pinocchio
    coal::coal
    yaml-cpp
    tinyxml2)
else()
  target_include_directories(collision_scene_node PRIVATE ${YAML_CPP_INCLUDE_DIRS})
  target_link_libraries(collision_scene_node
    pinocchio::pinocchio
    coal::coal
    ${YAML_CPP_LIBRARIES}
    tinyxml2)
endif()

# RPATH so the binary finds libs in the cmeel prefix at runtime
set_target_properties(collision_scene_node PROPERTIES
  BUILD_RPATH  "${CMEEL_PREFIX}/lib"
  INSTALL_RPATH "${CMEEL_PREFIX}/lib")

add_executable(speed_governor_node src/speed_governor_node.cpp)
ament_target_dependencies(speed_governor_node rclcpp std_msgs sensor_msgs)

add_executable(limiter_tuner_node src/limiter_tuner_node.cpp)
ament_target_dependencies(limiter_tuner_node rclcpp std_msgs)

install(TARGETS collision_scene_node speed_governor_node limiter_tuner_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
