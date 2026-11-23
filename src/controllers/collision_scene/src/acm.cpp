#include "collision_scene/acm.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

namespace collision_scene {

std::string AllowedCollisionMatrix::key(const std::string& a, const std::string& b) {
    return (a < b) ? (a + "|" + b) : (b + "|" + a);
}

void AllowedCollisionMatrix::setAllowed(const std::string& l1, const std::string& l2, bool allowed) {
    entries_[key(l1, l2)] = allowed;
}

bool AllowedCollisionMatrix::isAllowed(const std::string& l1, const std::string& l2) const {
    auto it = entries_.find(key(l1, l2));
    return it != entries_.end() && it->second;
}

AllowedCollisionMatrix AllowedCollisionMatrix::fromSRDF(const std::string& srdf_path) {
    AllowedCollisionMatrix acm;
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(srdf_path.c_str()) != tinyxml2::XML_SUCCESS) return acm;
    auto* robot = doc.RootElement();
    if (!robot) return acm;
    for (auto* el = robot->FirstChildElement("disable_collisions"); el;
         el = el->NextSiblingElement("disable_collisions")) {
        const char* l1 = el->Attribute("link1");
        const char* l2 = el->Attribute("link2");
        const char* reason = el->Attribute("reason");
        // "Never" only means random sampling found no contact; for a dual-arm
        // robot the arms can still touch, so those pairs must stay checked.
        if (reason && std::string(reason) == "Never") continue;
        if (l1 && l2) acm.setAllowed(l1, l2);
    }
    return acm;
}

AllowedCollisionMatrix AllowedCollisionMatrix::fromJSON(const std::string& json_path) {
    AllowedCollisionMatrix acm;
    std::ifstream f(json_path);
    if (!f.is_open()) return acm;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    try {
        YAML::Node root = YAML::Load(content);
        if (!root.IsSequence()) return acm;
        for (auto pair : root) {
            if (pair.IsSequence() && pair.size() == 2)
                acm.setAllowed(pair[0].as<std::string>(), pair[1].as<std::string>());
        }
    } catch (...) {}
    return acm;
}

} // namespace collision_scene
