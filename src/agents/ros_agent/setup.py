from glob import glob

from setuptools import setup

package_name = "ros_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/web", glob("web/*.html")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ihub",
    description="Daksha agent: read-only robot Q&A (URDF/SRDF + live state + cameras + LLM) and a real-robot "
                "VLM pick-and-place pipeline (local_daksha_rax/local_pipeline), one web UI.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "agent_node = ros_agent.agent_node:main",
            "ask = ros_agent.ask:main",
            "pipeline_node = ros_agent.pipeline_node:main",
        ],
    },
)
