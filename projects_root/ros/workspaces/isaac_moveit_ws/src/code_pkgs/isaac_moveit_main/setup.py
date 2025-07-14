from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'isaac_moveit_main'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('isaac_moveit_main', 'launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Isaac Sim MoveIt integration package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'launch_isaac = isaac_moveit_main.launch_isaac_wrapper:main',
            'goal_sender = isaac_moveit_main.goal_sender:main',
            'launch_rviz_moveit = isaac_moveit_main.launch_rviz_moveit:main',
            'moveit_monitor = isaac_moveit_main.moveit_monitor:main',
        ],
    },
)
