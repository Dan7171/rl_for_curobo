from setuptools import setup

package_name = 'goal_sender'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name + '/launch', ['launch/send_goal.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Sends pose goals to MoveIt2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_goal = goal_sender.send_goal:main',
        ],
    },
)
