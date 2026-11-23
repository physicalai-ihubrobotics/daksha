#include "collision_scene/collision_world.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
#include <coal/distance.h>
#include <yaml-cpp/yaml.h>

namespace collision_scene {

static coal::Transform3s toCoalTf(const Eigen::Matrix4d& T) {
    coal::Transform3s tf;
    tf.setRotation(T.topLeftCorner<3, 3>().cast<coal::FCL_REAL>());
    tf.setTranslation(T.topRightCorner<3, 1>().cast<coal::FCL_REAL>());
    return tf;
}

CollisionWorldFCL::CollisionWorldFCL() {
    world_manager_ = std::make_unique<coal::DynamicAABBTreeCollisionManager>();
    world_manager_->setup();
}

void CollisionWorldFCL::notifyObjectChange(WorldObject& obj) {
    if (!obj.fcl_obj) return;
    world_manager_->update(obj.fcl_obj.get());
}

void CollisionWorldFCL::addObject(const WorldObject& obj_in) {
    std::lock_guard<std::mutex> lk(mutex_);
    WorldObject obj = obj_in;

    // Create FCL CollisionObject for this world obstacle
    obj.fcl_obj = std::make_shared<coal::CollisionObject>(
        obj.shape, toCoalTf(obj.transform));

    // If a previous object with the same name exists, unregister it
    auto it = objects_.find(obj.name);
    if (it != objects_.end() && it->second.fcl_obj)
        world_manager_->unregisterObject(it->second.fcl_obj.get());

    objects_[obj.name] = std::move(obj);
    world_manager_->registerObject(objects_[obj_in.name].fcl_obj.get());
    world_manager_->update();
}

void CollisionWorldFCL::removeObject(const std::string& name) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = objects_.find(name);
    if (it != objects_.end()) {
        if (it->second.fcl_obj)
            world_manager_->unregisterObject(it->second.fcl_obj.get());
        objects_.erase(it);
        world_manager_->update();
    }
}

std::vector<WorldObject> CollisionWorldFCL::objects() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<WorldObject> v;
    v.reserve(objects_.size());
    for (auto& [k, o] : objects_) v.push_back(o);
    return v;
}

// Callback for robot-vs-world distance (MoveIt pattern)
struct WorldDistanceData : coal::DistanceCallBackBase {
    std::string current_robot_link;
    double threshold;
    std::vector<WorldContactResult> results;
    const std::unordered_map<std::string, WorldObject>* objects_map;
    std::unordered_map<coal::CollisionObject*, std::string> fcl_to_name;
    coal::DistanceRequest req;

    bool distance(coal::CollisionObject* o1, coal::CollisionObject* o2,
                  coal::CoalScalar& min_dist) override {
        // Identify which is the world object
        auto it1 = fcl_to_name.find(o1);
        auto it2 = fcl_to_name.find(o2);
        if (it1 == fcl_to_name.end() && it2 == fcl_to_name.end()) return false;
        const std::string& obj_name = (it1 != fcl_to_name.end())
            ? it1->second : it2->second;

        auto mit = objects_map->find(obj_name);
        if (mit == objects_map->end()) return false;
        const WorldObject& wo = mit->second;
        if (!current_robot_link.empty() && wo.ignore_links.count(current_robot_link))
            return false;

        coal::DistanceResult res;
        coal::distance(o1, o2, req, res);

        double d = static_cast<double>(res.min_distance);
        // Prune by threshold only; see SelfDistanceData in collision_robot.cpp.
        if (std::isfinite(threshold))
            min_dist = std::min(min_dist, static_cast<coal::CoalScalar>(threshold + 1e-6));

        if (d <= threshold) {
            results.push_back({
                current_robot_link, obj_name, d,
                res.nearest_points[0].cast<double>(),
                res.nearest_points[1].cast<double>(),
            });
        }
        return false;
    }
};

std::vector<WorldContactResult> CollisionWorldFCL::checkRobotWorld(
    const std::vector<std::shared_ptr<coal::CollisionObject>>& robot_fcl_objs,
    const std::vector<std::string>& link_names,
    double threshold)
{
    std::lock_guard<std::mutex> lk(mutex_);
    if (objects_.empty()) return {};

    WorldDistanceData dd;
    dd.threshold = threshold;
    dd.objects_map = &objects_;
    dd.req.enable_nearest_points = true;

    for (auto& [name, obj] : objects_)
        if (obj.fcl_obj) dd.fcl_to_name[obj.fcl_obj.get()] = name;

    // MoveIt: for each robot link, check against world manager
    for (std::size_t k = 0; k < robot_fcl_objs.size(); k++) {
        if (!robot_fcl_objs[k]) continue;
        dd.current_robot_link = (k < link_names.size()) ? link_names[k] : "";
        world_manager_->distance(robot_fcl_objs[k].get(), &dd);
    }

    std::sort(dd.results.begin(), dd.results.end(),
              [](const WorldContactResult& a, const WorldContactResult& b) {
                  return a.distance < b.distance;
              });
    return dd.results;
}

WorldObject worldObjectFromParams(
    const std::string& name, const std::string& type,
    const std::vector<double>& size,
    const std::vector<double>& xyz, const std::vector<double>& rpy,
    const std::set<std::string>& ignore_links)
{
    std::shared_ptr<coal::CollisionGeometry> shape;
    if (type == "box") {
        shape = std::make_shared<coal::Box>(size[0], size[1], size[2]);
    } else if (type == "sphere") {
        shape = std::make_shared<coal::Sphere>(size[0]);
    } else if (type == "cylinder") {
        shape = std::make_shared<coal::Cylinder>(size[0], size[1]);
    } else {
        throw std::runtime_error("Unknown obstacle type: " + type);
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topRightCorner<3, 1>() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
    T.topLeftCorner<3, 3>() =
        (Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX())).toRotationMatrix();

    return {name, shape, T, ignore_links, type, size, nullptr};
}

std::vector<WorldObject> loadWorldObjects(const std::string& yaml_path) {
    std::vector<WorldObject> objs;
    if (yaml_path.empty()) return objs;
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        auto list = root["obstacles"];
        if (!list || !list.IsSequence()) return objs;
        for (auto node : list) {
            std::string name = node["name"].as<std::string>();
            std::string type = node["type"].as<std::string>();
            auto sv = node["size"].as<std::vector<double>>();
            std::vector<double> xyz = {0, 0, 0}, rpy = {0, 0, 0};
            if (node["xyz"]) xyz = node["xyz"].as<std::vector<double>>();
            if (node["rpy"]) rpy = node["rpy"].as<std::vector<double>>();
            std::set<std::string> ignore;
            if (node["ignore_links"])
                for (auto il : node["ignore_links"]) ignore.insert(il.as<std::string>());
            objs.push_back(worldObjectFromParams(name, type, sv, xyz, rpy, ignore));
        }
    } catch (const std::exception&) {}
    return objs;
}

} // namespace collision_scene
