#pragma once

#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <coal/shape/geometric_shapes.h>
#include <coal/broadphase/broadphase_dynamic_AABB_tree.h>
#include <coal/broadphase/broadphase_callbacks.h>
#include <coal/collision_object.h>

namespace collision_scene {

struct WorldObject {
    std::string name;
    std::shared_ptr<coal::CollisionGeometry> shape;
    Eigen::Matrix4d transform;
    std::set<std::string> ignore_links;
    std::string kind;
    std::vector<double> size;
    mutable std::shared_ptr<coal::CollisionObject> fcl_obj;
};

struct WorldContactResult {
    std::string robot_link, object_name;
    double distance;
    Eigen::Vector3d point_on_robot, point_on_object;
};

class CollisionWorldFCL {
public:
    CollisionWorldFCL();

    void addObject(const WorldObject& obj);
    void removeObject(const std::string& name);
    std::vector<WorldObject> objects() const;

    std::vector<WorldContactResult> checkRobotWorld(
        const std::vector<std::shared_ptr<coal::CollisionObject>>& robot_fcl_objs,
        const std::vector<std::string>& link_names,
        double threshold = std::numeric_limits<double>::infinity());

private:
    void notifyObjectChange(WorldObject& obj);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, WorldObject> objects_;
    std::unique_ptr<coal::DynamicAABBTreeCollisionManager> world_manager_;
};

WorldObject worldObjectFromParams(
    const std::string& name, const std::string& type,
    const std::vector<double>& size,
    const std::vector<double>& xyz, const std::vector<double>& rpy,
    const std::set<std::string>& ignore_links = {});

std::vector<WorldObject> loadWorldObjects(const std::string& yaml_path);

} // namespace collision_scene
