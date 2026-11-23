// limiter_tuner_node
//
// Subscribes to /speed_governor/scale and sets joint_command_limiter's
// max_velocity / max_acceleration / gripper_max_velocity parameters at runtime
// as base * scale. Restores the base values when the governor goes silent and
// on shutdown.
//
// joint_command_limiter clamps acceleration in both directions, so lowering
// max_acceleration also weakens braking. While the scale is falling (arm must
// slow down) acceleration is held at full base value for `brake_hold` seconds;
// it is only scaled when the scale is steady or rising, and never below
// `min_acceleration`.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

using namespace std::chrono_literals;

namespace {
std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop = true; }
}  // namespace

class LimiterTuner : public rclcpp::Node {
public:
    LimiterTuner() : Node("limiter_tuner") {
        target_       = declare_parameter("target_node", std::string("/joint_command_limiter"));
        auto topic    = declare_parameter("scale_topic", std::string("/speed_governor/scale"));
        base_vel_     = declare_parameter("base_max_velocity", -1.0);       // <0 = read from target
        base_acc_     = declare_parameter("base_max_acceleration", -1.0);
        base_grip_    = declare_parameter("base_gripper_max_velocity", -1.0);
        scale_grip_   = declare_parameter("scale_gripper", true);
        min_acc_      = declare_parameter("min_acceleration", 2.0);
        brake_hold_   = declare_parameter("brake_hold", 0.5);
        rate_hz_      = declare_parameter("update_rate", 10.0);
        threshold_    = declare_parameter("change_threshold", 0.05);
        stale_timeout_= declare_parameter("stale_timeout", 0.5);
        stale_scale_  = declare_parameter("stale_scale", 1.0);

        client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, target_);

        sub_ = create_subscription<std_msgs::msg::Float64>(
            topic, 10, [this](std_msgs::msg::Float64::ConstSharedPtr m) {
                scale_ = std::clamp(m->data, 0.0, 1.0);
                last_msg_ = now();
                have_msg_ = true;
            });

        timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz_),
                                   [this] { tick(); });
    }

    // Blocking restore used at shutdown, after the executor has stopped.
    void restore(rclcpp::Executor& exec) {
        if (!have_base_ || !client_->service_is_ready()) return;
        auto fut = client_->set_parameters(makeParams(base_vel_, base_acc_, base_grip_));
        exec.spin_until_future_complete(fut, 1s);
        RCLCPP_INFO(get_logger(), "[limiter_tuner] restored base limits on %s", target_.c_str());
    }

private:
    std::vector<rclcpp::Parameter> makeParams(double v, double a, double g) const {
        std::vector<rclcpp::Parameter> p{
            rclcpp::Parameter("max_velocity", v),
            rclcpp::Parameter("max_acceleration", a)};
        if (scale_grip_) p.emplace_back("gripper_max_velocity", g);
        return p;
    }

    void fetchBase() {
        if (fetch_pending_ || !client_->service_is_ready()) return;
        fetch_pending_ = true;
        client_->get_parameters(
            {"max_velocity", "max_acceleration", "gripper_max_velocity"},
            [this](std::shared_future<std::vector<rclcpp::Parameter>> f) {
                fetch_pending_ = false;
                auto p = f.get();
                if (p.size() != 3) return;
                if (base_vel_  < 0) base_vel_  = p[0].as_double();
                if (base_acc_  < 0) base_acc_  = p[1].as_double();
                if (base_grip_ < 0) base_grip_ = p[2].as_double();
                have_base_ = true;
                RCLCPP_INFO(get_logger(),
                            "[limiter_tuner] base limits: v=%.2f rad/s a=%.2f rad/s^2 grip=%.3f m/s",
                            base_vel_, base_acc_, base_grip_);
            });
    }

    void tick() {
        if (!have_base_) {
            if (!client_->service_is_ready() && !warned_wait_) {
                RCLCPP_WARN(get_logger(), "[limiter_tuner] waiting for %s parameter service",
                            target_.c_str());
                warned_wait_ = true;
            }
            fetchBase();
            return;
        }

        auto t = now();
        bool stale = !have_msg_ || (t - last_msg_).seconds() > stale_timeout_;
        if (stale != was_stale_) {
            if (stale)
                RCLCPP_WARN(get_logger(), "[limiter_tuner] no %s — using scale %.2f",
                            "speed_governor scale", stale_scale_);
            was_stale_ = stale;
        }
        double s = stale ? stale_scale_ : scale_;

        if (s < prev_scale_ - 1e-3) brake_until_ = t + rclcpp::Duration::from_seconds(brake_hold_);
        prev_scale_ = s;
        bool braking = t < brake_until_;

        double v = base_vel_ * s;
        double a = braking ? base_acc_ : std::max(std::min(min_acc_, base_acc_), base_acc_ * s);
        double g = base_grip_ * s;

        auto changed = [this](double now_v, double last_v, double base) {
            return std::abs(now_v - last_v) > threshold_ * std::max(base, 1e-6);
        };
        bool need = !sent_once_ || changed(v, sent_v_, base_vel_) ||
                    changed(a, sent_a_, base_acc_) ||
                    (scale_grip_ && changed(g, sent_g_, base_grip_)) ||
                    (s == 0.0 && sent_v_ != 0.0) || (s == 1.0 && sent_v_ != base_vel_);
        if (!need || set_pending_) return;

        set_pending_ = true;
        client_->set_parameters(makeParams(v, a, g),
            [this, v, a, g](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> f) {
                set_pending_ = false;
                bool ok = true;
                for (auto& r : f.get()) ok = ok && r.successful;
                if (!ok) {
                    RCLCPP_WARN(get_logger(), "[limiter_tuner] set_parameters rejected");
                    return;
                }
                sent_v_ = v; sent_a_ = a; sent_g_ = g; sent_once_ = true;
            });
    }

    std::string target_;
    double base_vel_, base_acc_, base_grip_, min_acc_, brake_hold_, rate_hz_;
    double threshold_, stale_timeout_, stale_scale_;
    bool scale_grip_;

    rclcpp::AsyncParametersClient::SharedPtr client_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    double scale_ = 1.0, prev_scale_ = 1.0;
    rclcpp::Time last_msg_{0, 0, RCL_ROS_TIME};
    rclcpp::Time brake_until_{0, 0, RCL_ROS_TIME};
    bool have_msg_ = false, have_base_ = false, was_stale_ = false, warned_wait_ = false;
    bool fetch_pending_ = false, set_pending_ = false, sent_once_ = false;
    double sent_v_ = 0.0, sent_a_ = 0.0, sent_g_ = 0.0;
};

int main(int argc, char** argv) {
    // Own SIGINT handling so the base limits can be restored before shutdown.
    rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    auto node = std::make_shared<LimiterTuner>();
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    while (rclcpp::ok() && !g_stop) exec.spin_some(50ms);

    node->restore(exec);
    rclcpp::shutdown();
    return 0;
}
