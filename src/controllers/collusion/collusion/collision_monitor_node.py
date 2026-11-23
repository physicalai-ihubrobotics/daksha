"""ROS2 node: live self/environment collision monitor.

Reads joint positions from /joint_states, checks every pair at a fixed rate
and publishes:

    /collision/min_distance  std_msgs/Float64  closest pair separation (m)
    /collision/markers       visualization_msgs/MarkerArray

    /collision/predictive_guard  sensor_msgs/JointState  per joint, position
    /collision/reactive_guard    sensor_msgs/JointState  1.0 = flagged, 0.0 = ok
    /collision/joint_distance    sensor_msgs/JointState  per joint, position =
                                 closest separation (m) of the link it carries;
                                 inf = link in no checked pair, nan = stale

Per-joint guards. A flagged pair flags only the joint carrying each link in
contact (see CollisionChecker.link_joint) - left_link_5 hitting base_link
flags left_joint_4 alone, since joint N moves link N+1:
  predictive  the pose extrapolated prediction_horizon seconds ahead at the
              current joint velocity puts the pair within
              predictive_distance - i.e. it is GOING to collide.
  reactive    the pair is within contact_distance now - it HAS collided.

Missing or stale joint states set every joint in the predictive guard: no
data is not the same as no collision.

This is a software monitor. It is not a substitute for a hardware e-stop.

    ros2 launch collusion collision_monitor.launch.py
"""

from __future__ import annotations

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Float64
from visualization_msgs.msg import Marker, MarkerArray

from collusion.collision_checker import CollisionChecker, load_obstacles


def _default_path(package, *parts):
    try:
        return os.path.join(get_package_share_directory(package), *parts)
    except Exception:
        return ""


class CollisionMonitor(Node):

    def __init__(self):
        super().__init__("collision_monitor")

        # Must be the SAME geometry the pair allowlist was generated from.
        # Empty -> package default.
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('collision_pairs_file', '')
        self.declare_parameter('obstacles_file', '')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('check_self', True)
        self.declare_parameter('check_environment', True)
        self.declare_parameter('joint_state_timeout', 0.5)
        self.declare_parameter('marker_frame', 'world')
        self.declare_parameter('contact_distance', 0.0)
        self.declare_parameter('predictive_distance', 0.03)
        self.declare_parameter('prediction_horizon', 0.3)
        # Low-pass on the finite-difference velocity, 1.0 = unfiltered.
        self.declare_parameter('velocity_filter', 0.5)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        urdf = p('urdf_file') or _default_path(
            "daksha_description_full_body", "urdf", "robot_collision.urdf")
        pairs = p('collision_pairs_file') or _default_path(
            "kinematics", "config", "collision_pairs.json")
        obstacles_file = p('obstacles_file') or _default_path(
            "collusion", "config", "obstacles.yaml")

        if not urdf or not os.path.exists(urdf):
            raise RuntimeError(f"[collision] urdf_file not found: '{urdf}'")
        if not pairs or not os.path.exists(pairs):
            # Without the allowlist placo checks every pair (slow) and
            # parent/child-adjacent geometry the filter can't catch reads
            # as colliding. Run anyway, but loudly.
            self.get_logger().warn(
                f"[collision] collision_pairs_file not found: '{pairs}'. "
                f"Checking ALL pairs - slow and may false-trigger. Generate with "
                f"kinematics/scripts/gen_collision_pairs.py --urdf {urdf}")
            pairs = None

        obstacles = []
        if p('check_environment'):
            if obstacles_file and os.path.exists(obstacles_file):
                obstacles = load_obstacles(obstacles_file)
            else:
                self.get_logger().warn(
                    f"[collision] obstacles_file not found: '{obstacles_file}'")

        self.checker = CollisionChecker(urdf, pairs, obstacles)
        self.get_logger().info(
            f"[collision] urdf={urdf} pairs={pairs} "
            f"obstacles={[o.name for o in obstacles]}")

        # /joint_states from kinematics/ik_node is multiplied by its
        # joint_direction map; divide it back out to get URDF-convention
        # angles. Load the same joint_direction.yaml into this node.
        self._joint_dir = {
            name: self.declare_parameter(f'joint_direction.{name}', 1.0).value
            for name in sorted(self.checker.joint_names)
        }

        self._positions = {}
        self._velocities = {}
        self._last_js = None
        self._last_stamp = None
        self._warned_unknown = False
        self._reactive = set()

        self.create_subscription(
            JointState, p('joint_states_topic'), self._joint_cb, 10)
        self.pub_min = self.create_publisher(Float64, '/collision/min_distance', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/collision/markers', 10)
        self.pub_predictive = self.create_publisher(
            JointState, '/collision/predictive_guard', 10)
        self.pub_reactive = self.create_publisher(
            JointState, '/collision/reactive_guard', 10)
        self.pub_joint_distance = self.create_publisher(
            JointState, '/collision/joint_distance', 10)

        self.create_timer(1.0 / p('rate_hz'), self._tick)

    def _joint_cb(self, msg: JointState) -> None:
        now = self.get_clock().now()
        # Prefer the message's own stamp: receive time jitters with the
        # executor. joint_state_publisher_gui stamps its messages.
        stamp = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                 if msg.header.stamp.sec else now.nanoseconds * 1e-9)
        dt = stamp - self._last_stamp if self._last_stamp is not None else 0.0
        alpha = self.get_parameter('velocity_filter').value
        have_vel = len(msg.velocity) == len(msg.name)

        for i, (name, pos) in enumerate(zip(msg.name, msg.position)):
            pos /= self._joint_dir.get(name, 1.0)
            if have_vel:
                vel = msg.velocity[i] / self._joint_dir.get(name, 1.0)
            elif dt > 1e-4 and name in self._positions:
                vel = (pos - self._positions[name]) / dt
            else:
                vel = 0.0
            old = self._velocities.get(name, vel)
            self._velocities[name] = alpha * vel + (1.0 - alpha) * old
            self._positions[name] = pos
        self._last_stamp = stamp
        self._last_js = now

    def _tick(self) -> None:
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        if self._last_js is None or (
                (self.get_clock().now() - self._last_js).nanoseconds * 1e-9
                > p('joint_state_timeout')):
            self._publish(float('nan'), [], [])
            # Unknown state: flag every joint predictively, keep the last
            # reactive result - nothing new is known to have collided.
            self._publish_guards(self.checker.joint_names, self._reactive)
            self._publish_joint_distance(None)
            return

        checks = (p('check_self'), p('check_environment'))
        unknown = self.checker.set_joints(self._positions)
        if unknown and not self._warned_unknown:
            self.get_logger().warn(f"[collision] joints not in URDF, ignored: {unknown}")
            self._warned_unknown = True

        pairs = self.checker.check(*checks)
        min_d = pairs[0].distance if pairs else float('inf')
        reactive = self.checker.flagged_joints(pairs, p('contact_distance'))
        for c in pairs:
            if c.distance > p('contact_distance'):
                break
            new = self.checker.pair_joints(c) - self._reactive
            if new:
                self.get_logger().error(
                    f"[collision] {c.kind}: {c.link_a} <-> {c.link_b} "
                    f"{c.distance * 1000:.1f} mm -> {sorted(new)}")
        self._reactive = reactive
        self._publish_joint_distance(self.checker.joint_distances(pairs))

        # Predictive: same check on the pose prediction_horizon ahead. Only
        # costs a second check while something is actually moving.
        horizon = p('prediction_horizon')
        moving = {n: v for n, v in self._velocities.items() if abs(v) > 1e-3}
        if moving and horizon > 0.0:
            self.checker.set_joints({
                n: q + moving.get(n, 0.0) * horizon for n, q in self._positions.items()})
            predicted = self.checker.check(*checks)
        else:
            predicted = pairs
        predictive = self.checker.flagged_joints(predicted, p('predictive_distance'))
        self._publish_guards(predictive, self._reactive)
        self._publish(min_d, pairs, predicted)

    def _publish_guards(self, predictive, reactive):
        stamp = self.get_clock().now().to_msg()
        names = self.checker.joint_order
        for pub, flagged in ((self.pub_predictive, predictive),
                             (self.pub_reactive, reactive)):
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = list(names)
            msg.position = [1.0 if n in flagged else 0.0 for n in names]
            pub.publish(msg)

    def _publish_joint_distance(self, distances):
        """distances None -> stale, every joint nan."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.checker.joint_order)
        msg.position = [float('nan') if distances is None
                        else distances.get(n, float('inf')) for n in msg.name]
        self.pub_joint_distance.publish(msg)

    def _publish(self, min_d, pairs, predicted):
        self.pub_min.publish(Float64(data=min_d))
        self._publish_markers(pairs, predicted)

    def _publish_markers(self, pairs, predicted):
        frame = self.get_parameter('marker_frame').value
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        for i, obs in enumerate(self.checker.obstacles):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns, m.id, m.action = 'obstacles', i, Marker.ADD
            m.type = {'box': Marker.CUBE, 'sphere': Marker.SPHERE,
                      'cylinder': Marker.CYLINDER}[obs.kind]
            T = obs.transform
            m.pose.position.x, m.pose.position.y, m.pose.position.z = T[:3, 3]
            qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
            m.pose.orientation.x, m.pose.orientation.y = qx, qy
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            if obs.kind == 'box':
                m.scale.x, m.scale.y, m.scale.z = obs.size
            elif obs.kind == 'sphere':
                m.scale.x = m.scale.y = m.scale.z = 2 * obs.size[0]
            else:
                m.scale.x = m.scale.y = 2 * obs.size[0]
                m.scale.z = obs.size[1]
            m.color.r, m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.6, 0.4
            markers.markers.append(m)

        # Red = in contact now (reactive), yellow = going to collide
        # (predictive). Spheres, because at contact the two closest points
        # coincide and a line would have zero length. Yellow is drawn at the
        # pair's closest points on the CURRENT pose, so it sits on the robot
        # RViz shows, plus a line across the remaining gap.
        contact = self.get_parameter('contact_distance').value
        red_c = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        yellow_c = ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0)

        def pt(v):
            return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))

        def key(c):
            return (c.kind, c.link_a, c.link_b)

        hit = [c for c in pairs if c.distance <= contact]
        hit_keys = {key(c) for c in hit}
        now = {}
        for c in pairs:                         # sorted, keep closest per link pair
            now.setdefault(key(c), c)
        near = []
        for c in predicted:
            if c.distance > self.get_parameter('predictive_distance').value:
                break
            if key(c) not in hit_keys:
                near.append(now.get(key(c), c))
                hit_keys.add(key(c))            # once per link pair

        spheres = self._marker(frame, stamp, 'collision_points', Marker.SPHERE_LIST)
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.03
        lines = self._marker(frame, stamp, 'predictive_gap', Marker.LINE_LIST)
        lines.scale.x = 0.004
        for c, color in [(c, red_c) for c in hit] + [(c, yellow_c) for c in near]:
            spheres.points += [pt(c.point_a), pt(c.point_b)]
            spheres.colors += [color, color]
        for c in near:
            lines.points += [pt(c.point_a), pt(c.point_b)]
            lines.colors += [yellow_c, yellow_c]
        for m in (spheres, lines):
            if not m.points:
                m.action = Marker.DELETE
            markers.markers.append(m)

        self.pub_markers.publish(markers)

    @staticmethod
    def _marker(frame, stamp, ns, kind):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = ns, 0, kind, Marker.ADD
        m.pose.orientation.w = 1.0
        return m


def main(args=None):
    rclpy.init(args=args)
    node = CollisionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
