#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/model.hpp>

#include <coal/broadphase/broadphase_dynamic_AABB_tree.h>
#include <coal/broadphase/broadphase_callbacks.h>
#include <coal/collision_object.h>

#include "collision_scene/acm.hpp"

namespace collision_scene {

struct ContactResult {
    std::string link_a, link_b;
    double distance;
    Eigen::Vector3d point_a;
    Eigen::Vector3d point_b;
};

struct SelfCollisionResult {
    bool collision;
    std::vector<ContactResult> contacts;
};

class CollisionRobotFCL {
public:
    CollisionRobotFCL(const std::string& urdf_path,
                      const std::string& srdf_path = "",
                      const std::string& allowed_pairs_path = "");

    std::vector<std::string> setJoints(
        const std::unordered_map<std::string, double>& positions);

    std::vector<ContactResult> checkSelfCollisionDistance(
        double threshold = std::numeric_limits<double>::infinity());

    SelfCollisionResult checkSelfCollision();

    std::optional<std::string> linkJoint(const std::string& link);
    std::set<std::string> flaggedJoints(const std::vector<ContactResult>& contacts);

    const pinocchio::GeometryModel& geomModel() const { return geom_model_; }
    pinocchio::GeometryData&        geomData()        { return geom_data_; }
    const std::vector<std::string>& linkNames() const { return link_names_; }
    const std::set<std::string>&    jointNames() const { return joint_names_; }
    const std::vector<std::string>& jointOrder() const { return joint_order_; }
    const AllowedCollisionMatrix&   acm() const { return acm_; }

    const std::vector<std::shared_ptr<coal::CollisionObject>>& fclObjects() const {
        return fcl_objs_;
    }

    // Links whose name contains any pattern are checked against every other
    // link, bypassing the allowlist (ACM/adjacency still apply).
    // Returns the matched link names.
    std::set<std::string> setAlwaysCheckPatterns(const std::vector<std::string>& patterns);

    struct GeomUserData {
        std::string link_name;
        std::size_t geom_idx;
        bool always_check = false;
    };

private:
    static std::string stripIndex(const std::string& name);

    void updateFCLTransforms();

    pinocchio::Model         model_;
    pinocchio::Data          data_;
    pinocchio::GeometryModel geom_model_;
    pinocchio::GeometryData  geom_data_;
    Eigen::VectorXd          q_;

    AllowedCollisionMatrix acm_;
    // Pairs to check (placo-style allowlist). Empty = check every non-ACM pair.
    AllowedCollisionMatrix check_list_;
    bool has_check_list_ = false;
    std::vector<GeomUserData> geom_user_data_;
    std::vector<std::shared_ptr<coal::CollisionObject>> fcl_objs_;
    std::unique_ptr<coal::DynamicAABBTreeCollisionManager> self_manager_;

    std::unordered_map<std::string, int> joint_q_idx_;
    std::set<std::string>                joint_names_;
    std::vector<std::string>             joint_order_;
    std::vector<std::string>             link_names_;

    std::unordered_map<std::string, std::string> link_to_joint_name_;
    std::unordered_map<std::string, std::string> link_to_parent_link_;
    struct Mimic { std::string joint, master; double multiplier, offset; };
    std::vector<Mimic> mimics_;
    std::unordered_map<std::string, std::string> link_joint_cache_;
};

} // namespace collision_scene
