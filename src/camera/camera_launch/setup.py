from setuptools import setup, find_packages
from glob import glob

package_name = 'camera_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/templates', glob(package_name + '/templates/*.html')),
        ('share/' + package_name + '/static/css', glob(package_name + '/static/css/*.css')),
        ('share/' + package_name + '/static/js', glob(package_name + '/static/js/*.js')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phy_ai',
    maintainer_email='selvambharani100@gmial.com',
    description="Launch files bringing up the robot's RealSense cameras (world D455 + left/right D405 wrist cameras)",
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_ui = camera_launch.camera_ui:main',
        ],
    },
)
