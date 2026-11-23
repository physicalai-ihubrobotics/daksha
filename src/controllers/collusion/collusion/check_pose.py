"""Offline check of one configuration, no ROS graph needed.

    ros2 run collusion check_pose left_joint_2=0.5 right_joint_4=-1.2
    ros2 run collusion check_pose --obstacles my_cell.yaml --top 20 left_joint_1=1.0

Unlisted joints stay at 0. Prints the closest pairs, closest first.
"""

import argparse
import os

from ament_index_python.packages import get_package_share_directory

from collusion.collision_checker import CollisionChecker, load_obstacles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joints", nargs="*", help="name=radians")
    ap.add_argument("--urdf", default=os.path.join(
        get_package_share_directory("daksha_description_full_body"),
        "urdf", "robot_collision.urdf"))
    ap.add_argument("--pairs", default=os.path.join(
        get_package_share_directory("kinematics"), "config", "collision_pairs.json"))
    ap.add_argument("--obstacles", default=os.path.join(
        get_package_share_directory("collusion"), "config", "obstacles.yaml"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    obstacles = load_obstacles(args.obstacles) if os.path.exists(args.obstacles) else []
    checker = CollisionChecker(args.urdf, args.pairs, obstacles)

    positions = {}
    for item in args.joints:
        name, _, value = item.partition("=")
        positions[name] = float(value)
    unknown = checker.set_joints(positions)
    if unknown:
        print(f"ignored, not in URDF: {unknown}")

    pairs = checker.check()
    for c in pairs[:args.top]:
        flag = "COLLISION" if c.distance <= 0 else ""
        print(f"{c.kind:4} {c.link_a:>24} <-> {c.link_b:<24} "
              f"{c.distance * 1000:8.1f} mm  {flag}")
    if not pairs:
        print("no pairs checked")


if __name__ == "__main__":
    main()
