// speed_governor_node
//
// Estimates per-joint velocity/acceleration, predicts the state `horizon`
// seconds ahead, and computes how much each joint must slow down based on the
// collision distance published by collision_scene_node. It only PUBLISHES;
// nothing here feeds the command path.
//
// Subscribes
//   /joint_states                     sensor_msgs/JointState
//   /collision_scene/joint_distance   sensor_msgs/JointState (position = min distance [m])
//
// Publishes (JointState has no acceleration field, so `effort` carries it)
//   ~/state      position=current  velocity=estimated     effort=estimated accel
//   ~/predicted  position=at t+T   velocity=at t+T        effort=accel
//   ~/limits     position=scale    velocity=allowed v_max effort=allowed a_max
//   ~/scale      std_msgs/Float64  global scale (min over joints)

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double clamp01(double x) { return std::min(1.0, std::max(0.0, x)); }

struct JointEstimate {
    double pos = 0.0, vel = 0.0, acc = 0.0;
    bool   seeded = false;

    double dist = kInf;        // closest collision distance [m]
    double dist_rate = 0.0;    // d(dist)/dt [m/s], negative = approaching
    double dist_stamp = -1.0;

    double scale = 1.0;        // smoothed output scale
};

}  // namespace

class SpeedGovernor : public rclcpp::Node {
public:
    SpeedGovernor() : Node("speed_governor") {
        rate_hz_        = declare_parameter("rate_hz", 50.0);
        horizon_        = declare_parameter("prediction_horizon", 0.3);
        vel_alpha_      = declare_parameter("velocity_filter", 0.5);
        acc_alpha_      = declare_parameter("acceleration_filter", 0.3);
        dist_alpha_     = declare_parameter("distance_rate_filter", 0.4);
        max_vel_        = declare_parameter("max_velocity", 3.0);
        max_acc_        = declare_parameter("max_acceleration", 6.0);
        grip_max_vel_   = declare_parameter("gripper_max_velocity", 0.25);
        grip_max_acc_   = declare_parameter("gripper_max_acceleration", 2.5);
        slow_dist_      = declare_parameter("slow_distance", 0.10);
        stop_dist_      = declare_parameter("stop_distance", 0.01);
        ttc_slow_       = declare_parameter("ttc_slow", 1.0);
        ttc_stop_       = declare_parameter("ttc_stop", 0.2);
        release_scale_  = declare_parameter("release_scale", 0.3);
        closing_deadband_ = declare_parameter("closing_deadband", 0.01);
        min_scale_      = declare_parameter("min_scale", 0.0);
        rise_rate_      = declare_parameter("scale_rise_rate", 1.0);
        dist_timeout_   = declare_parameter("distance_timeout", 0.5);
        stale_scale_    = declare_parameter("stale_scale", 0.0);

        auto js_topic   = declare_parameter("joint_states_topic", std::string("/joint_states"));
        auto jd_topic   = declare_parameter("joint_distance_topic",
                                            std::string("/collision_scene/joint_distance"));

        sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
            js_topic, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { onJointStates(*m); });
        sub_jd_ = create_subscription<sensor_msgs::msg::JointState>(
            jd_topic, 10,
            [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { onJointDistance(*m); });

        pub_state_ = create_publisher<sensor_msgs::msg::JointState>("~/state", 10);
        pub_pred_  = create_publisher<sensor_msgs::msg::JointState>("~/predicted", 10);
        pub_lim_   = create_publisher<sensor_msgs::msg::JointState>("~/limits", 10);
        pub_scale_ = create_publisher<std_msgs::msg::Float64>("~/scale", 10);

        timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz_),
                                   [this] { tick(); });

        RCLCPP_INFO(get_logger(),
                    "[speed_governor] slow<%.3fm stop<%.3fm ttc %.2f..%.2fs horizon %.2fs",
                    slow_dist_, stop_dist_, ttc_stop_, ttc_slow_, horizon_);
    }

private:
    static double stampSec(const builtin_interfaces::msg::Time& t) {
        return t.sec + t.nanosec * 1e-9;
    }

    double msgTime(const std_msgs::msg::Header& h) {
        double s = stampSec(h.stamp);
        return s > 0.0 ? s : now().seconds();
    }

    void onJointStates(const sensor_msgs::msg::JointState& m) {
        double t = msgTime(m.header);
        double dt = last_js_time_ > 0.0 ? t - last_js_time_ : 0.0;
        if (last_js_time_ > 0.0 && dt <= 1e-4) return;   // duplicate stamp
        last_js_time_ = t;

        bool have_vel = m.velocity.size() == m.name.size();
        for (std::size_t i = 0; i < m.name.size() && i < m.position.size(); i++) {
            auto [it, inserted] = joints_.try_emplace(m.name[i]);
            if (inserted) order_.push_back(m.name[i]);
            auto& j = it->second;
            double p = m.position[i];
            if (!j.seeded || dt <= 0.0) {
                j.pos = p; j.vel = have_vel ? m.velocity[i] : 0.0; j.acc = 0.0; j.seeded = true;
                continue;
            }
            double v_raw = have_vel ? m.velocity[i] : (p - j.pos) / dt;
            double v_new = vel_alpha_ * v_raw + (1.0 - vel_alpha_) * j.vel;
            double a_raw = (v_new - j.vel) / dt;
            j.acc = acc_alpha_ * a_raw + (1.0 - acc_alpha_) * j.acc;
            j.vel = v_new;
            j.pos = p;
        }
    }

    void onJointDistance(const sensor_msgs::msg::JointState& m) {
        double t = msgTime(m.header);
        last_jd_time_ = now().seconds();
        for (std::size_t i = 0; i < m.name.size() && i < m.position.size(); i++) {
            auto [it, inserted] = joints_.try_emplace(m.name[i]);
            if (inserted) order_.push_back(m.name[i]);
            auto& j = it->second;
            double d = m.position[i];
            double dt = t - j.dist_stamp;
            if (std::isfinite(d) && std::isfinite(j.dist) && j.dist_stamp > 0.0 && dt > 1e-4) {
                double r = (d - j.dist) / dt;
                j.dist_rate = dist_alpha_ * r + (1.0 - dist_alpha_) * j.dist_rate;
            } else if (!std::isfinite(d)) {
                j.dist_rate = 0.0;
            }
            j.dist = d;
            j.dist_stamp = t;
        }
    }

    double targetScale(const JointEstimate& j) const {
        if (!std::isfinite(j.dist)) return 1.0;
        double s_dist = clamp01((j.dist - stop_dist_) / (slow_dist_ - stop_dist_));
        double closing = -j.dist_rate;
        if (closing > closing_deadband_) {
            double ttc = std::max(0.0, j.dist) / closing;
            double s_ttc = clamp01((ttc - ttc_stop_) / (ttc_slow_ - ttc_stop_));
            return std::min(s_dist, s_ttc);
        }
        // Not approaching: let the joint back away instead of freezing it.
        return std::max(s_dist, release_scale_);
    }

    void tick() {
        if (order_.empty()) return;
        double t_now = now().seconds();
        double dt = last_tick_ > 0.0 ? t_now - last_tick_ : 1.0 / rate_hz_;
        last_tick_ = t_now;

        bool stale = last_jd_time_ < 0.0 || (t_now - last_jd_time_) > dist_timeout_;
        if (stale && !warned_stale_) {
            RCLCPP_WARN(get_logger(),
                        "[speed_governor] no fresh joint_distance — holding scale at %.2f",
                        stale_scale_);
            warned_stale_ = true;
        } else if (!stale) {
            warned_stale_ = false;
        }

        sensor_msgs::msg::JointState st, pr, lim;
        auto stamp = now();
        st.header.stamp = pr.header.stamp = lim.header.stamp = stamp;
        double global = 1.0;

        for (auto& name : order_) {
            auto& j = joints_[name];
            if (!j.seeded) continue;

            double target = stale ? stale_scale_ : targetScale(j);
            target = std::max(min_scale_, target);
            // Drop immediately, recover at a bounded rate (no jerk on release).
            j.scale = target < j.scale ? target
                                       : std::min(target, j.scale + rise_rate_ * dt);
            global = std::min(global, j.scale);

            bool grip = name.find("gripper") != std::string::npos;
            double vmax = (grip ? grip_max_vel_ : max_vel_) * j.scale;
            double amax = (grip ? grip_max_acc_ : max_acc_) * j.scale;

            st.name.push_back(name);
            st.position.push_back(j.pos);
            st.velocity.push_back(j.vel);
            st.effort.push_back(j.acc);

            double T = horizon_;
            pr.name.push_back(name);
            pr.position.push_back(j.pos + j.vel * T + 0.5 * j.acc * T * T);
            pr.velocity.push_back(j.vel + j.acc * T);
            pr.effort.push_back(j.acc);

            lim.name.push_back(name);
            lim.position.push_back(j.scale);
            lim.velocity.push_back(vmax);
            lim.effort.push_back(amax);
        }

        pub_state_->publish(st);
        pub_pred_->publish(pr);
        pub_lim_->publish(lim);
        std_msgs::msg::Float64 g; g.data = global;
        pub_scale_->publish(g);
    }

    double rate_hz_, horizon_, vel_alpha_, acc_alpha_, dist_alpha_;
    double max_vel_, max_acc_, grip_max_vel_, grip_max_acc_;
    double slow_dist_, stop_dist_, ttc_slow_, ttc_stop_;
    double release_scale_, closing_deadband_, min_scale_, rise_rate_, dist_timeout_, stale_scale_;

    std::unordered_map<std::string, JointEstimate> joints_;
    std::vector<std::string> order_;
    double last_js_time_ = -1.0, last_jd_time_ = -1.0, last_tick_ = -1.0;
    bool warned_stale_ = false;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_, sub_jd_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_state_, pub_pred_, pub_lim_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_scale_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SpeedGovernor>());
    rclcpp::shutdown();
    return 0;
}
