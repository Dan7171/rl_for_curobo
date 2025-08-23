from setuptools import setup

package_name = "curobo_ros_workspace"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Helper ROS 2 nodes for CuRobo multi-robot demos.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "plan_listener = curobo_ros_workspace.plan_listener:main",
        ],
    },
) 