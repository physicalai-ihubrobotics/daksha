#pragma once

#include <string>
#include <unordered_map>

namespace collision_scene {

class AllowedCollisionMatrix {
public:
    void setAllowed(const std::string& l1, const std::string& l2, bool allowed = true);
    bool isAllowed(const std::string& l1, const std::string& l2) const;

    std::size_t size() const { return entries_.size(); }

    static AllowedCollisionMatrix fromSRDF(const std::string& srdf_path);
    static AllowedCollisionMatrix fromJSON(const std::string& json_path);

private:
    static std::string key(const std::string& a, const std::string& b);
    std::unordered_map<std::string, bool> entries_;
};

} // namespace collision_scene
