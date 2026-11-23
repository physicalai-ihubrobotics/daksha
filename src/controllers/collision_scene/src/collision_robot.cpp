#include "collision_scene/collision_robot.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <urdf_parser/urdf_parser.h>

#include <coal/distance.h>

namespace collision_scene {

static std::string stripIndexImpl(const std::string& name) {
    auto pos = name.rfind('_');
    if (pos != std::string::npos && pos + 1 < name.size()) {
        bool all_digits = true;
        for (auto c : name.substr(pos + 1))
            if (!std::isdigit(static_cast<unsigned char>(c))) { all_digits = false; break; }
        if (all_digits) return name.substr(0, pos);
    }
    return name;
}

static std::vector<std::string> getPackageDirs() {
    std::vector<std::string> dirs;
    const char* ament = std::getenv("AMENT_PREFIX_PATH");
    if (!ament) return dirs;
    std::string s(ament);
    std::string::size_type pos = 0, next;
    while ((next = s.find(':', pos)) != std::string::npos) {
        dirs.push_back(s.substr(pos, next - pos) + "/share");
        pos = next + 1;
    }
    dirs.push_back(s.substr(pos) + "/share");
    return dirs;
}

static std::string findSRDF() {
    const char* ament = std::getenv("AMENT_PREFIX_PATH");
    if (!ament) return "";
    std::istringstream ss(ament);
    std::string prefix;
    while (std::getline(ss, prefix, ':')) {
        std::string candidate =
            prefix + "/share/daksha_description_full_body/srdf/daksha.srdf";
        if (std::ifstream(candidate).good()) return candidate;
    }
    return "";
}

std::string CollisionRobotFCL::stripIndex(const std::string& name) {
    return stripIndexImpl(name);
}

CollisionRobotFCL::CollisionRobotFCL(const std::string& urdf_path,
                                      const std::string& srdf_path,
                                      const std::string& allowed_pairs_path)
{
    pinocchio::urdf::buildModel(urdf_path,
                               pinocchio::JointModelFreeFlyer(), model_);
    data_ = pinocchio::Data(model_);

    auto pkg_dirs = getPackageDirs();
    pinocchio::urdf::buildGeom(model_, urdf_path,
                               pinocchio::GeometryType::COLLISION,
                               geom_model_, pkg_dirs);

    link_names_.reserve(geom_model_.ngeoms);
    for (std::size_t k = 0; k < geom_model_.ngeoms; k++)
        link_names_.push_back(stripIndexImpl(geom_model_.geometryObjects[k].name));

    // Joint name → q-index (skip universe=0 and free-flyer=1)
    for (std::size_t i = 2; i < (std::size_t)model_.njoints; i++) {
        const std::string& jname = model_.names[i];
        int nqj = (int)model_.joints[i].nq();
        if (nqj == 1) {
            joint_names_.insert(jname);
            joint_order_.push_back(jname);
            joint_q_idx_[jname] = (int)model_.joints[i].idx_q();
        }
    }

    // Build parent-child tree for linkJoint()
    auto urdf_model = urdf::parseURDFFile(urdf_path);
    if (urdf_model) {
        for (auto& [jname, joint] : urdf_model->joints_) {
            if (joint->child_link_name.empty()) continue;
            link_to_joint_name_[joint->child_link_name]  = jname;
            link_to_parent_link_[joint->child_link_name] = joint->parent_link_name;
            if (joint->mimic && joint_q_idx_.count(jname) &&
                joint_q_idx_.count(joint->mimic->joint_name))
                mimics_.push_back({jname, joint->mimic->joint_name,
                                   joint->mimic->multiplier, joint->mimic->offset});
        }
    }

    // --- ACM: load from SRDF, then merge JSON, then add adjacent/same-link pairs ---
    {
        std::string srdf = srdf_path.empty() ? findSRDF() : srdf_path;
        if (!srdf.empty())
            acm_ = AllowedCollisionMatrix::fromSRDF(srdf);
        // collision_pairs.json is the placo allowlist of pairs TO CHECK
        // (see kinematics/scripts/gen_collision_pairs.py), not pairs to skip.
        if (!allowed_pairs_path.empty()) {
            check_list_ = AllowedCollisionMatrix::fromJSON(allowed_pairs_path);
            has_check_list_ = check_list_.size() > 0;
        }
    }

    // Always allow same-link and adjacent pairs
    for (std::size_t i = 0; i < geom_model_.ngeoms; i++) {
        for (std::size_t j = i + 1; j < geom_model_.ngeoms; j++) {
            if (link_names_[i] == link_names_[j]) {
                acm_.setAllowed(link_names_[i], link_names_[j]);
                continue;
            }
            std::size_t ja = geom_model_.geometryObjects[i].parentJoint;
            std::size_t jb = geom_model_.geometryObjects[j].parentJoint;
            bool adjacent = (ja == jb) ||
                            (model_.parents[ja] == jb) ||
                            (model_.parents[jb] == ja);
            if (adjacent)
                acm_.setAllowed(link_names_[i], link_names_[j]);
        }
    }

    // Pre-compute FCL objects (computeLocalAABB called once inside CollisionObject ctor)
    geom_user_data_.resize(geom_model_.ngeoms);
    fcl_objs_.resize(geom_model_.ngeoms);
    for (std::size_t k = 0; k < geom_model_.ngeoms; k++) {
        geom_user_data_[k] = {link_names_[k], k};
        // Store user data on the CollisionObject itself
        fcl_objs_[k] = std::make_shared<coal::CollisionObject>(
            geom_model_.geometryObjects[k].geometry,
            coal::Transform3s());
        fcl_objs_[k]->setUserData(static_cast<void*>(&geom_user_data_[k]));
    }

    // Setup broad-phase self-collision manager
    self_manager_ = std::make_unique<coal::DynamicAABBTreeCollisionManager>();
    for (auto& obj : fcl_objs_)
        self_manager_->registerObject(obj.get());
    self_manager_->setup();

    geom_data_ = pinocchio::GeometryData(geom_model_);

    q_ = pinocchio::neutral(model_);
    updateFCLTransforms();
}

void CollisionRobotFCL::updateFCLTransforms() {
    pinocchio::forwardKinematics(model_, data_, q_);
    pinocchio::updateGeometryPlacements(model_, data_, geom_model_, geom_data_);

    for (std::size_t k = 0; k < geom_model_.ngeoms; k++) {
        const auto& se3 = geom_data_.oMg[k];
        coal::Transform3s tf;
        tf.setRotation(se3.rotation().cast<coal::FCL_REAL>());
        tf.setTranslation(se3.translation().cast<coal::FCL_REAL>());
        fcl_objs_[k]->setTransform(tf);
        fcl_objs_[k]->computeAABB();
    }
    self_manager_->update();
}

std::vector<std::string> CollisionRobotFCL::setJoints(
    const std::unordered_map<std::string, double>& positions)
{
    std::vector<std::string> unknown;
    for (auto& [name, val] : positions) {
        auto it = joint_q_idx_.find(name);
        if (it != joint_q_idx_.end())
            q_[it->second] = val;
        else
            unknown.push_back(name);
    }
    // Mimic joints (e.g. the second gripper finger) are usually absent from
    // /joint_states; derive them from their master joint.
    for (auto& mj : mimics_)
        if (!positions.count(mj.joint))
            q_[joint_q_idx_.at(mj.joint)] =
                mj.multiplier * q_[joint_q_idx_.at(mj.master)] + mj.offset;
    updateFCLTransforms();
    return unknown;
}

// Local struct for self-collision distance callback (MoveIt pattern)
struct SelfDistanceData : coal::DistanceCallBackBase {
    const AllowedCollisionMatrix& acm;
    const AllowedCollisionMatrix* check_list = nullptr;
    double threshold;
    std::vector<ContactResult> results;
    coal::DistanceRequest req;

    SelfDistanceData(const AllowedCollisionMatrix& a, double t)
        : acm(a), threshold(t) {
        req.enable_nearest_points = true;
    }

    bool distance(coal::CollisionObject* o1, coal::CollisionObject* o2,
                  coal::CoalScalar& min_dist) override {
        const auto* ud1 = static_cast<const CollisionRobotFCL::GeomUserData*>(
            o1->getUserData());
        const auto* ud2 = static_cast<const CollisionRobotFCL::GeomUserData*>(
            o2->getUserData());
        if (!ud1 || !ud2) return false;
        if (ud1->link_name == ud2->link_name) return false;
        if (acm.isAllowed(ud1->link_name, ud2->link_name)) return false;
        if (check_list && !ud1->always_check && !ud2->always_check &&
            !check_list->isAllowed(ud1->link_name, ud2->link_name))
            return false;

        coal::DistanceResult res;
        coal::distance(o1, o2, req, res);

        double d = static_cast<double>(res.min_distance);

        // coal descends only while aabb_dist < min_dist (strict). Shrinking
        // min_dist to found distances would prune every other contact (overlapping
        // AABBs have dist 0), so prune by the requested threshold only.
        if (std::isfinite(threshold))
            min_dist = std::min(min_dist, static_cast<coal::CoalScalar>(threshold + 1e-6));

        if (d <= threshold) {
            // For penetrating geometries, nearest_points may be (0,0,0).
            // Fall back to object origins so markers appear at the right link.
            Eigen::Vector3d pa = res.nearest_points[0].cast<double>();
            Eigen::Vector3d pb = res.nearest_points[1].cast<double>();
            if (d < 0.0 && pa.isZero(1e-6) && pb.isZero(1e-6)) {
                pa = o1->getTranslation().cast<double>();
                pb = o2->getTranslation().cast<double>();
            }
            results.push_back({ud1->link_name, ud2->link_name, d, pa, pb});
        }
        return false;
    }
};

std::set<std::string> CollisionRobotFCL::setAlwaysCheckPatterns(
    const std::vector<std::string>& patterns)
{
    std::set<std::string> matched;
    for (auto& ud : geom_user_data_) {
        ud.always_check = false;
        for (auto& p : patterns)
            if (!p.empty() && ud.link_name.find(p) != std::string::npos) {
                ud.always_check = true;
                matched.insert(ud.link_name);
                break;
            }
    }
    return matched;
}

std::vector<ContactResult> CollisionRobotFCL::checkSelfCollisionDistance(double threshold) {
    SelfDistanceData dd(acm_, threshold);
    if (has_check_list_) dd.check_list = &check_list_;
    self_manager_->distance(&dd);
    std::sort(dd.results.begin(), dd.results.end(),
              [](const ContactResult& a, const ContactResult& b) {
                  return a.distance < b.distance;
              });
    return dd.results;
}

SelfCollisionResult CollisionRobotFCL::checkSelfCollision() {
    auto contacts = checkSelfCollisionDistance(0.0);
    return {!contacts.empty(), std::move(contacts)};
}

std::optional<std::string> CollisionRobotFCL::linkJoint(const std::string& link) {
    auto cached = link_joint_cache_.find(link);
    if (cached != link_joint_cache_.end())
        return cached->second.empty() ? std::nullopt
                                      : std::optional<std::string>(cached->second);

    std::string current = link;
    while (link_to_joint_name_.count(current)) {
        const std::string& jname = link_to_joint_name_.at(current);
        if (joint_names_.count(jname)) {
            link_joint_cache_[link] = jname;
            return jname;
        }
        auto pit = link_to_parent_link_.find(current);
        if (pit == link_to_parent_link_.end()) break;
        current = pit->second;
    }
    link_joint_cache_[link] = "";
    return std::nullopt;
}

std::set<std::string> CollisionRobotFCL::flaggedJoints(
    const std::vector<ContactResult>& contacts)
{
    std::set<std::string> joints;
    for (auto& c : contacts) {
        for (const auto& link : {c.link_a, c.link_b}) {
            auto j = linkJoint(link);
            if (j) joints.insert(*j);
        }
    }
    return joints;
}

} // namespace collision_scene
