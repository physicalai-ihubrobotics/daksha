#include "collision_scene/planning_scene.hpp"

namespace collision_scene {

PlanningScene::PlanningScene(const std::string& urdf_path,
                              const std::string& srdf_path,
                              const std::string& allowed_pairs_path,
                              const std::string& obstacles_yaml)
    : robot_(urdf_path, srdf_path, allowed_pairs_path)
{
    if (!obstacles_yaml.empty()) {
        for (auto& obj : loadWorldObjects(obstacles_yaml))
            world_.addObject(obj);
    }
}

std::vector<std::string> PlanningScene::setJoints(
    const std::unordered_map<std::string, double>& positions)
{
    return robot_.setJoints(positions);
}

CollisionResult PlanningScene::checkCollision(
    bool self_collision, bool world_collision, double distance_threshold)
{
    CollisionResult result{false, {}, {}};
    if (self_collision) {
        result.self_contacts = robot_.checkSelfCollisionDistance(distance_threshold);
        for (auto& c : result.self_contacts)
            if (c.distance <= 0.0) { result.collision = true; break; }
    }
    if (world_collision) {
        result.world_contacts = world_.checkRobotWorld(
            robot_.fclObjects(), robot_.linkNames(), distance_threshold);
        for (auto& c : result.world_contacts)
            if (c.distance <= 0.0) { result.collision = true; break; }
    }
    return result;
}

void PlanningScene::addWorldObject(const WorldObject& obj) { world_.addObject(obj); }
void PlanningScene::removeWorldObject(const std::string& name) { world_.removeObject(name); }

std::set<std::string> PlanningScene::flaggedJointsSelf(
    const std::vector<ContactResult>& contacts, double threshold)
{
    std::vector<ContactResult> filtered;
    for (auto& c : contacts) if (c.distance <= threshold) filtered.push_back(c);
    return robot_.flaggedJoints(filtered);
}

std::set<std::string> PlanningScene::flaggedJointsWorld(
    const std::vector<WorldContactResult>& contacts, double threshold)
{
    std::set<std::string> joints;
    for (auto& c : contacts) {
        if (c.distance <= threshold) {
            auto j = robot_.linkJoint(c.robot_link);
            if (j) joints.insert(*j);
        }
    }
    return joints;
}

} // namespace collision_scene
