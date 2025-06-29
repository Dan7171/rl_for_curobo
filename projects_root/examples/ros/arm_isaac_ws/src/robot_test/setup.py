from setuptools import setup

package_name = 'robot_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='your_email@example.com',
    description='Isaac Sim Franka Panda Robot Control Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = robot_test.robot_controller:main',
            'robot_subscriber = robot_test.robot_subscriber:main',
        ],
    },
) 