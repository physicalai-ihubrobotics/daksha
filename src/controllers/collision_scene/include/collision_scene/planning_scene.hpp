#pragma once

#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "collision_scene/collision_robot.hpp"
#include "collision_scene/collision_world.hpp"

namespace collision_scene {

struct CollisionResult {
    bool collision;
    std::vector<ContactResult>      self_contacts;
    std::vector<WorldContactResult> world_contacts;

    double minDistance() const {
        double d = std::numeric_limits<double>::infinity();
        for (auto& c : self_contacts)  d = std::min(d, c.distance);
        for (auto& c : world_contacts) d = std::min(d, c.distance);
        return d;
    }
};

class PlanningScene {
public:
    PlanningScene(const std::string& urdf_path,
                  const std::string& srdf_path = "",
                  const std::string& allowed_pairs_path = "",
                  const std::string& obstacles_yaml = "");

    std::vector<std::string> setJoints(
        const std::unordered_map<std::string, double>& positions);

    CollisionResult checkCollision(
        bool self_collision = true,
        bool world_collision = true,
        double distance_threshold = std::numeric_limits<double>::infinity());

    void addWorldObject(const WorldObject& obj);
    void removeWorldObject(const std::string& name);

    std::set<std::string> flaggedJointsSelf(
        const std::vector<ContactResult>& contacts, double threshold = 0.0);
    std::set<std::string> flaggedJointsWorld(
        const std::vector<WorldContactResult>& contacts, double threshold = 0.0);

    CollisionRobotFCL& robot() { return robot_; }
    CollisionWorldFCL& world() { return world_; }
    const std::set<std::string>&    jointNames() const { return robot_.jointNames(); }
    const std::vector<std::string>& jointOrder() const { return robot_.jointOrder(); }

private:
    CollisionRobotFCL robot_;
    CollisionWorldFCL world_;
};

} // namespace collision_scene
