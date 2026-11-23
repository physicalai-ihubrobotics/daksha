import os

from setuptools import setup

package_name = "bionic_hand"
share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version="0.1.0",
    # The SDK modules and the ROS node all live in the bionic_hand package.
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (share_dir, ["package.xml"]),
        (os.path.join(share_dir, "launch"), ["launch/bionic_hand.launch.py"]),
        (os.path.join(share_dir, "config"), ["config/bionic_hand.yaml"]),
        # Web UI assets, loaded at runtime by the hand_web_ui node.
        (os.path.join(share_dir, "web"), ["web/app.py", "web/index.html"]),
    ],
    install_requires=["setuptools", "pyserial>=3.5,<4.0"],
    zip_safe=True,
    maintainer="Baranidharan",
    maintainer_email="ihrphysicalai@gmail.com",
    description="ROS 2 driver and SDK for the SCS0009 left/right bionic hands.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "bionic_hand_node = bionic_hand.ros_node:main",
            "hand_web_ui = bionic_hand.web_node:main",
        ],
    },
)
