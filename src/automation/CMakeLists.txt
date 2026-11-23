cmake_minimum_required(VERSION 3.8)
project(automation)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  automation/bring_up.py
  DESTINATION lib/${PROJECT_NAME}
)
# Flask looks for templates/ and static/ next to bring_up.py, so
# they must be installed alongside it in lib/${PROJECT_NAME}, not share/.
# config/ (ROS 2 param-style yaml, e.g. viewer_joint_overrides.yaml) is
# resolved by bring_up.py the same way, so it goes here too.
install(DIRECTORY
  automation/templates
  automation/static
  automation/config
  DESTINATION lib/${PROJECT_NAME}
)

# Shared browser-access gate, imported by every Daksha web UI. They locate it
# with get_package_prefix("automation") + lib/automation; it reads
# config/web_access.yaml (installed above) relative to itself.
install(FILES
  automation/daksha_web_access.py
  automation/daksha_flask_gate.py
  DESTINATION lib/${PROJECT_NAME}
)
# Gate env files, read by daksha_web_access from share/automation/webpage.
install(FILES
  webpage/daksha.conf
  webpage/.env
  DESTINATION share/${PROJECT_NAME}/webpage
  OPTIONAL
)

ament_package()
