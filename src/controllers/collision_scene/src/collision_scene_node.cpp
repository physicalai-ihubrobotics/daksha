#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <yaml-cpp/yaml.h>

#include "collision_scene/planning_scene.hpp"

using namespace std::chrono_literals;

namespace collision_scene {

static std::string defaultPath(const std::string& pkg, const std::string& rel) {
    try {
        return ament_index_cpp::get_package_share_directory(pkg) + "/" + rel;
    } catch (...) {
        return "";
    }
}

class CollisionSceneNode : public rclcpp::Node {
public:
    CollisionSceneNode()
        : Node("collision_scene")
    {
        declare_parameter("urdf_source", "topic");
        declare_parameter("urdf_file", "");
        declare_parameter("srdf_file", "");
        declare_parameter("robot_description_topic", "/robot_description");
        declare_parameter("collision_pairs_file", "");
        declare_parameter("obstacles_file", "");
        declare_parameter("joint_states_topic", "/joint_states");
        declare_parameter("rate_hz", 30.0);
        declare_parameter("check_self", true);
        declare_parameter("check_world", true);
        declare_parameter("joint_state_timeout", 1.0);
        declare_parameter("marker_frame", "world");
        declare_parameter("contact_distance", 0.0);
        declare_parameter("predictive_distance", 0.03);
        declare_parameter("prediction_horizon", 0.3);
        declare_parameter("velocity_filter", 0.5);
        declare_parameter("marker_closest_n", 10);
        declare_parameter("always_check_links",
                          std::vector<std::string>{"gripper", "finger"});

        // Publishers
        pub_self_      = create_publisher<std_msgs::msg::Bool>("/collision_scene/self_collision", 10);
        pub_world_     = create_publisher<std_msgs::msg::Bool>("/collision_scene/world_collision", 10);
        pub_col_       = create_publisher<std_msgs::msg::Bool>("/collision_scene/collision", 10);
        pub_min_       = create_publisher<std_msgs::msg::Float64>("/collision_scene/min_distance", 10);
        pub_pred_      = create_publisher<sensor_msgs::msg::JointState>("/collision_scene/predictive_guard", 10);
        pub_react_     = create_publisher<sensor_msgs::msg::JointState>("/collision_scene/reactive_guard", 10);
        pub_jdist_     = create_publisher<sensor_msgs::msg::JointState>("/collision_scene/joint_distance", 10);
        pub_markers_   = create_publisher<visualization_msgs::msg::MarkerArray>("/collision_scene/markers", 10);

        // Joint states subscriber
        auto js_topic = get_parameter("joint_states_topic").as_string();
        sub_joints_ = create_subscription<sensor_msgs::msg::JointState>(
            js_topic, 10,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) { jointCb(msg); });

        // Obstacle subscriber
        sub_obstacle_ = create_subscription<std_msgs::msg::String>(
            "/collision_scene/add_obstacle", 10,
            [this](std_msgs::msg::String::SharedPtr msg) { obstacleCb(msg); });

        // URDF source
        auto urdf_source = get_parameter("urdf_source").as_string();
        if (urdf_source == "topic") {
            auto topic = get_parameter("robot_description_topic").as_string();
            RCLCPP_INFO(get_logger(), "[collision_scene] waiting for URDF on '%s'", topic.c_str());
            rclcpp::QoS qos(1);
            qos.reliable().transient_local();
            sub_urdf_ = create_subscription<std_msgs::msg::String>(
                topic, qos,
                [this](std_msgs::msg::String::SharedPtr msg) { urdfCb(msg); });
        } else {
            auto urdf = get_parameter("urdf_file").as_string();
            if (urdf.empty())
                urdf = defaultPath("daksha_description_full_body", "urdf/robot_collision.urdf");
            initScene(urdf);
        }

        double rate_hz = get_parameter("rate_hz").as_double();
        auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / rate_hz));
        timer_ = create_wall_timer(period_ns, [this]() { tick(); });
    }

private:
    // ── scene init ─────────────────────────────────────────────────────────────

    void urdfCb(std_msgs::msg::String::SharedPtr msg) {
        if (scene_) return;
        // Write to temp file
        char tmpl[] = "/tmp/collision_scene_XXXXXX.urdf";
        int fd = mkstemps(tmpl, 5);
        if (fd < 0) {
            RCLCPP_ERROR(get_logger(), "[collision_scene] mkstemp failed");
            return;
        }
        write(fd, msg->data.c_str(), msg->data.size());
        close(fd);
        tmp_urdf_ = tmpl;
        RCLCPP_INFO(get_logger(), "[collision_scene] URDF received (%zu bytes) → %s",
                    msg->data.size(), tmpl);
        initScene(tmpl);
    }

    void initScene(const std::string& urdf_path) {
        auto srdf = get_parameter("srdf_file").as_string();

        auto pairs = get_parameter("collision_pairs_file").as_string();
        if (pairs.empty()) pairs = defaultPath("kinematics", "config/collision_pairs.json");

        auto obs = get_parameter("obstacles_file").as_string();
        if (obs.empty()) obs = defaultPath("collision_scene", "config/obstacles.yaml");

        try {
            scene_ = std::make_unique<PlanningScene>(urdf_path, srdf, pairs, obs);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "[collision_scene] PlanningScene init failed: %s", e.what());
            return;
        }

        {
            auto pats = get_parameter("always_check_links").as_string_array();
            auto matched = scene_->robot().setAlwaysCheckPatterns(pats);
            std::string s;
            for (auto& l : matched) s += l + " ";
            RCLCPP_INFO(get_logger(), "[collision_scene] always-checked links: %s", s.c_str());
        }

        // Declare joint_direction params
        for (auto& jname : scene_->jointOrder()) {
            try { declare_parameter("joint_direction." + jname, 1.0); } catch (...) {}
            joint_direction_[jname] = get_parameter("joint_direction." + jname).as_double();
        }

        RCLCPP_INFO(get_logger(), "[collision_scene] ready — %zu joints, obstacles=%zu",
                    scene_->jointNames().size(),
                    scene_->world().objects().size());
    }

    // ── joint-state callback ───────────────────────────────────────────────────

    void jointCb(sensor_msgs::msg::JointState::SharedPtr msg) {
        double stamp = (msg->header.stamp.sec > 0)
            ? msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9
            : now().nanoseconds() * 1e-9;

        double dt = (last_stamp_ > 0.0) ? (stamp - last_stamp_) : 0.0;
        double alpha = get_parameter("velocity_filter").as_double();
        bool have_vel = (msg->velocity.size() == msg->name.size());

        for (std::size_t i = 0; i < msg->name.size(); i++) {
            const std::string& name = msg->name[i];
            double dir = 1.0;
            auto dit = joint_direction_.find(name);
            if (dit != joint_direction_.end()) dir = dit->second;

            double pos = msg->position[i] / dir;
            double vel = 0.0;
            if (have_vel) {
                vel = msg->velocity[i] / dir;
            } else if (dt > 1e-4 && positions_.count(name)) {
                vel = (pos - positions_[name]) / dt;
            }

            double prev_vel = velocities_.count(name) ? velocities_[name] : vel;
            velocities_[name] = alpha * vel + (1.0 - alpha) * prev_vel;
            positions_[name]  = pos;
        }
        last_stamp_   = stamp;
        last_js_time_ = now().nanoseconds() * 1e-9;
    }

    // ── obstacle callback ──────────────────────────────────────────────────────

    void obstacleCb(std_msgs::msg::String::SharedPtr msg) {
        if (!scene_) {
            RCLCPP_WARN(get_logger(), "[collision_scene] URDF not received yet — ignoring obstacle");
            return;
        }
        try {
            YAML::Node n = YAML::Load(msg->data);
            std::string name = n["name"].as<std::string>();
            std::string type = n["type"].as<std::string>();
            auto sv = n["size"].as<std::vector<double>>();
            std::vector<double> xyz = {0, 0, 0}, rpy = {0, 0, 0};
            if (n["xyz"]) xyz = n["xyz"].as<std::vector<double>>();
            if (n["rpy"]) rpy = n["rpy"].as<std::vector<double>>();
            std::set<std::string> ignore;
            if (n["ignore_links"])
                for (auto il : n["ignore_links"]) ignore.insert(il.as<std::string>());
            auto obj = worldObjectFromParams(name, type, sv, xyz, rpy, ignore);
            scene_->addWorldObject(obj);
            RCLCPP_INFO(get_logger(), "[collision_scene] added obstacle '%s'", name.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "[collision_scene] add_obstacle error: %s", e.what());
        }
    }

    // ── timer tick ─────────────────────────────────────────────────────────────

    void tick() {
        if (!scene_) return;

        double now_s     = now().nanoseconds() * 1e-9;
        double timeout   = get_parameter("joint_state_timeout").as_double();
        bool stale = (last_js_time_ < 0.0) || (now_s - last_js_time_ > timeout);
        if (stale) {
            publishStale();
            return;
        }

        bool check_self  = get_parameter("check_self").as_bool();
        bool check_world = get_parameter("check_world").as_bool();
        double contact_d = get_parameter("contact_distance").as_double();
        double pred_d    = get_parameter("predictive_distance").as_double();
        double horizon   = get_parameter("prediction_horizon").as_double();

        // Apply current joint state
        {
            auto unk = scene_->setJoints(positions_);
            if (!unk.empty() && !warned_unknown_) {
                RCLCPP_WARN(get_logger(), "[collision_scene] unknown joints (ignored): count=%zu", unk.size());
                warned_unknown_ = true;
            }
        }

        // Current collision check
        auto result = scene_->checkCollision(check_self, check_world,
                                              std::numeric_limits<double>::infinity());

        bool self_col  = false, world_col = false;
        for (auto& c : result.self_contacts)  if (c.distance <= contact_d) { self_col  = true; break; }
        for (auto& c : result.world_contacts) if (c.distance <= contact_d) { world_col = true; break; }

        auto reactive = scene_->flaggedJointsSelf(result.self_contacts, contact_d);
        auto rw = scene_->flaggedJointsWorld(result.world_contacts, contact_d);
        reactive.insert(rw.begin(), rw.end());

        // Log new collisions
        for (auto& c : result.self_contacts) {
            if (c.distance <= contact_d) {
                auto joints = scene_->robot().flaggedJoints({c});
                bool is_new = false;
                for (auto& j : joints) if (!prev_reactive_.count(j)) { is_new = true; break; }
                if (is_new) {
                    std::string jlist;
                    for (auto& j : joints) jlist += j + " ";
                    RCLCPP_ERROR(get_logger(), "[collision_scene] SELF: %s <-> %s %.1f mm → [%s]",
                                 c.link_a.c_str(), c.link_b.c_str(),
                                 c.distance * 1000.0, jlist.c_str());
                }
            }
        }
        prev_reactive_ = reactive;

        // Predictive check
        CollisionResult pred_result = result;
        std::set<std::string> predictive;

        bool any_moving = false;
        for (auto& [n, v] : velocities_) if (std::abs(v) > 1e-3) { any_moving = true; break; }

        if (any_moving && horizon > 0.0) {
            std::unordered_map<std::string, double> pred_pos;
            for (auto& [n, p] : positions_)
                pred_pos[n] = p + (velocities_.count(n) ? velocities_.at(n) * horizon : 0.0);
            scene_->setJoints(pred_pos);
            pred_result = scene_->checkCollision(check_self, check_world, pred_d);
            auto pj = scene_->flaggedJointsSelf(pred_result.self_contacts, pred_d);
            auto pw = scene_->flaggedJointsWorld(pred_result.world_contacts, pred_d);
            predictive.insert(pj.begin(), pj.end());
            predictive.insert(pw.begin(), pw.end());
            scene_->setJoints(positions_);
        }

        // Publish
        {
            std_msgs::msg::Bool b;
            b.data = self_col; pub_self_->publish(b);
            b.data = world_col; pub_world_->publish(b);
            b.data = self_col || world_col; pub_col_->publish(b);
        }
        {
            std_msgs::msg::Float64 f;
            f.data = result.minDistance();
            pub_min_->publish(f);
        }

        auto all_pred = predictive;
        all_pred.insert(reactive.begin(), reactive.end());
        publishGuards(all_pred, reactive);
        publishJointDistance(result);
        publishMarkers(result, pred_result);
    }

    // ── publish helpers ────────────────────────────────────────────────────────

    void publishStale() {
        std_msgs::msg::Bool b; b.data = false;
        pub_self_->publish(b); pub_world_->publish(b); pub_col_->publish(b);
        std_msgs::msg::Float64 f; f.data = std::numeric_limits<double>::quiet_NaN();
        pub_min_->publish(f);
    }

    void publishGuards(const std::set<std::string>& predictive,
                       const std::set<std::string>& reactive)
    {
        if (!scene_) return;
        auto stamp = now();
        const auto& names = scene_->jointOrder();

        for (auto* pub : {&pub_pred_, &pub_react_}) {
            const auto& flagged = (pub == &pub_pred_) ? predictive : reactive;
            sensor_msgs::msg::JointState msg;
            msg.header.stamp = stamp;
            msg.name = names;
            msg.position.reserve(names.size());
            for (auto& n : names)
                msg.position.push_back(flagged.count(n) ? 1.0 : 0.0);
            (*pub)->publish(msg);
        }
    }

    void publishJointDistance(const CollisionResult& result) {
        if (!scene_) return;
        const auto& names = scene_->jointOrder();
        std::unordered_map<std::string, double> jdist;
        for (auto& c : result.self_contacts) {
            for (const auto& link : {c.link_a, c.link_b}) {
                auto j = scene_->robot().linkJoint(link);
                if (j) {
                    auto it = jdist.find(*j);
                    if (it == jdist.end() || c.distance < it->second)
                        jdist[*j] = c.distance;
                }
            }
        }
        for (auto& c : result.world_contacts) {
            auto j = scene_->robot().linkJoint(c.robot_link);
            if (j) {
                auto it = jdist.find(*j);
                if (it == jdist.end() || c.distance < it->second)
                    jdist[*j] = c.distance;
            }
        }
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = now();
        msg.name = names;
        for (auto& n : names) {
            auto it = jdist.find(n);
            msg.position.push_back(it != jdist.end() ? it->second
                                                      : std::numeric_limits<double>::infinity());
        }
        pub_jdist_->publish(msg);
    }

    void publishMarkers(const CollisionResult& result,
                        const CollisionResult& pred_result)
    {
        if (!scene_) return;
        std::string frame = get_parameter("marker_frame").as_string();
        double contact_d = get_parameter("contact_distance").as_double();
        double pred_d    = get_parameter("predictive_distance").as_double();

        visualization_msgs::msg::MarkerArray ma;
        auto stamp = now();

        // Helper: create a base marker
        auto newMarker = [&](const std::string& ns, int type) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame;
            m.header.stamp    = stamp;
            m.ns  = ns;
            m.id  = 0;
            m.type   = type;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.orientation.w = 1.0;
            return m;
        };

        // Red/yellow colours
        std_msgs::msg::ColorRGBA red, yellow, blue;
        red.r = 1.0f; red.g = 0.0f; red.b = 0.0f; red.a = 1.0f;
        yellow.r = 1.0f; yellow.g = 0.85f; yellow.b = 0.0f; yellow.a = 1.0f;
        blue.r = 0.5f; blue.g = 0.5f; blue.b = 0.6f; blue.a = 0.4f;

        auto toPoint = [](const Eigen::Vector3d& v) {
            geometry_msgs::msg::Point p;
            p.x = v.x(); p.y = v.y(); p.z = v.z();
            return p;
        };

        // World obstacle markers
        {
            auto objs = scene_->world().objects();
            for (std::size_t i = 0; i < objs.size(); i++) {
                auto& obj = objs[i];
                visualization_msgs::msg::Marker m;
                m.header.frame_id = frame;
                m.header.stamp    = stamp;
                m.ns  = "obstacles";
                m.id  = (int)i;
                m.action = visualization_msgs::msg::Marker::ADD;
                if      (obj.kind == "box")      m.type = visualization_msgs::msg::Marker::CUBE;
                else if (obj.kind == "sphere")   m.type = visualization_msgs::msg::Marker::SPHERE;
                else if (obj.kind == "cylinder") m.type = visualization_msgs::msg::Marker::CYLINDER;
                else                             m.type = visualization_msgs::msg::Marker::CUBE;

                const auto& T = obj.transform;
                m.pose.position.x = T(0, 3);
                m.pose.position.y = T(1, 3);
                m.pose.position.z = T(2, 3);
                Eigen::Quaterniond q(T.topLeftCorner<3,3>());
                q.normalize();
                m.pose.orientation.x = q.x();
                m.pose.orientation.y = q.y();
                m.pose.orientation.z = q.z();
                m.pose.orientation.w = q.w();
                if (obj.kind == "box") {
                    m.scale.x = obj.size[0]; m.scale.y = obj.size[1]; m.scale.z = obj.size[2];
                } else if (obj.kind == "sphere") {
                    m.scale.x = m.scale.y = m.scale.z = 2.0 * obj.size[0];
                } else {
                    m.scale.x = m.scale.y = 2.0 * obj.size[0]; m.scale.z = obj.size[1];
                }
                m.color = blue;
                ma.markers.push_back(m);
            }
        }

        // Build hit / near sets
        using SelfKey = std::pair<std::string, std::string>;
        std::set<SelfKey> hit_keys;

        // reactive pairs
        std::vector<const ContactResult*>      hit_self;
        std::vector<const WorldContactResult*> hit_world;
        for (auto& c : result.self_contacts)
            if (c.distance <= contact_d) {
                hit_self.push_back(&c);
                hit_keys.insert({c.link_a, c.link_b});
            }
        for (auto& c : result.world_contacts)
            if (c.distance <= contact_d)
                hit_world.push_back(&c);

        // lookup maps for current-pose contacts
        std::unordered_map<std::string, const ContactResult*>      now_self;
        std::unordered_map<std::string, const WorldContactResult*> now_world;
        for (auto& c : result.self_contacts)
            now_self[c.link_a + "|" + c.link_b] = &c;
        for (auto& c : result.world_contacts)
            now_world[c.robot_link + "|" + c.object_name] = &c;

        // predictive pairs
        std::vector<const ContactResult*>      near_self;
        std::vector<const WorldContactResult*> near_world;
        for (auto& c : pred_result.self_contacts) {
            if (c.distance > pred_d) break;
            SelfKey k{c.link_a, c.link_b};
            if (!hit_keys.count(k)) {
                auto it = now_self.find(c.link_a + "|" + c.link_b);
                near_self.push_back(it != now_self.end() ? it->second : &c);
                hit_keys.insert(k);
            }
        }
        for (auto& c : pred_result.world_contacts) {
            if (c.distance > pred_d) break;
            std::string wk = c.robot_link + "|" + c.object_name;
            auto it = now_world.find(wk);
            near_world.push_back(it != now_world.end() ? it->second : &c);
        }

        // closest_N pairs — always visible, green→yellow→red by distance
        int closest_n = get_parameter("marker_closest_n").as_int();
        if (closest_n > 0 && !result.self_contacts.empty()) {
            auto proximity = newMarker("closest_pairs",
                                       visualization_msgs::msg::Marker::SPHERE_LIST);
            proximity.scale.x = proximity.scale.y = proximity.scale.z = 0.025;
            int shown = 0;
            for (auto& c : result.self_contacts) {
                if (shown >= closest_n) break;
                // green (far) → yellow (medium) → red (close/contact)
                std_msgs::msg::ColorRGBA col;
                col.a = 0.9f;
                if (c.distance <= contact_d) {
                    col.r = 1.0f; col.g = 0.0f; col.b = 0.0f;  // red
                } else if (c.distance <= pred_d) {
                    col.r = 1.0f; col.g = 0.85f; col.b = 0.0f; // yellow
                } else {
                    float t = std::min(1.0f, (float)(pred_d / std::max(c.distance, 1e-6)));
                    col.r = t; col.g = 1.0f - 0.15f * t; col.b = 0.0f; // green→yellow
                }
                proximity.points.push_back(toPoint(c.point_a));
                proximity.points.push_back(toPoint(c.point_b));
                proximity.colors.push_back(col);
                proximity.colors.push_back(col);
                shown++;
            }
            proximity.action = visualization_msgs::msg::Marker::ADD;
            ma.markers.push_back(proximity);
        }

        // collision_points SPHERE_LIST
        auto spheres = newMarker("collision_points",
                                 visualization_msgs::msg::Marker::SPHERE_LIST);
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.03;

        for (auto* c : hit_self) {
            spheres.points.push_back(toPoint(c->point_a));
            spheres.points.push_back(toPoint(c->point_b));
            spheres.colors.push_back(red); spheres.colors.push_back(red);
        }
        for (auto* c : hit_world) {
            spheres.points.push_back(toPoint(c->point_on_robot));
            spheres.points.push_back(toPoint(c->point_on_object));
            spheres.colors.push_back(red); spheres.colors.push_back(red);
        }
        for (auto* c : near_self) {
            spheres.points.push_back(toPoint(c->point_a));
            spheres.points.push_back(toPoint(c->point_b));
            spheres.colors.push_back(yellow); spheres.colors.push_back(yellow);
        }
        for (auto* c : near_world) {
            spheres.points.push_back(toPoint(c->point_on_robot));
            spheres.points.push_back(toPoint(c->point_on_object));
            spheres.colors.push_back(yellow); spheres.colors.push_back(yellow);
        }
        spheres.action = spheres.points.empty()
            ? visualization_msgs::msg::Marker::DELETE
            : visualization_msgs::msg::Marker::ADD;
        ma.markers.push_back(spheres);

        // predictive_gap LINE_LIST
        auto lines = newMarker("predictive_gap",
                               visualization_msgs::msg::Marker::LINE_LIST);
        lines.scale.x = 0.004;
        for (auto* c : near_self) {
            lines.points.push_back(toPoint(c->point_a));
            lines.points.push_back(toPoint(c->point_b));
            lines.colors.push_back(yellow); lines.colors.push_back(yellow);
        }
        for (auto* c : near_world) {
            lines.points.push_back(toPoint(c->point_on_robot));
            lines.points.push_back(toPoint(c->point_on_object));
            lines.colors.push_back(yellow); lines.colors.push_back(yellow);
        }
        lines.action = lines.points.empty()
            ? visualization_msgs::msg::Marker::DELETE
            : visualization_msgs::msg::Marker::ADD;
        ma.markers.push_back(lines);

        pub_markers_->publish(ma);
    }

    // ── members ────────────────────────────────────────────────────────────────

    std::unique_ptr<PlanningScene> scene_;
    std::string tmp_urdf_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr        sub_urdf_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joints_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr        sub_obstacle_;
    rclcpp::TimerBase::SharedPtr                                   timer_;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr             pub_self_, pub_world_, pub_col_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr          pub_min_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr    pub_pred_, pub_react_, pub_jdist_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;

    std::unordered_map<std::string, double> positions_;
    std::unordered_map<std::string, double> velocities_;
    std::unordered_map<std::string, double> joint_direction_;
    std::set<std::string>                   prev_reactive_;

    double last_stamp_   = -1.0;
    double last_js_time_ = -1.0;
    bool   warned_unknown_ = false;
};

} // namespace collision_scene

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<collision_scene::CollisionSceneNode>());
    rclcpp::shutdown();
    return 0;
}
