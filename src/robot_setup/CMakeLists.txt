cmake_minimum_required(VERSION 3.8)
project(robot_setup)

find_package(ament_cmake REQUIRED)

# The scripts also run straight from src/ (needed on a fresh machine, before
# the first build). Installing them makes `ros2 run robot_setup ...` work too.
install(PROGRAMS
  scripts/install.sh
  scripts/install_apt.sh
  scripts/install_pcan.sh
  scripts/install_llm_agent.sh
  scripts/start_llm_server.sh
  scripts/llm_env.sh
  scripts/uninstall.sh
  scripts/verify_env.sh
  scripts/check_requirements.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  requirements.txt
  apt_requirements.txt
  templates/activate_daksha.sh.in
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
