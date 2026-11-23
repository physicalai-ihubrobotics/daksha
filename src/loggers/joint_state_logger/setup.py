from setuptools import setup
from glob import glob

package_name = 'joint_state_logger'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),

        ('share/' + package_name,
         ['package.xml']),

        ('share/' + package_name + '/launch',
         glob('launch/*.launch.py')),

        ('share/' + package_name + '/templates',
         glob(package_name + '/templates/*.html')),
        ('share/' + package_name + '/static/css',
         glob(package_name + '/static/css/*.css')),
        ('share/' + package_name + '/static/js',
         glob(package_name + '/static/js/*.js')),
        ('share/' + package_name + '/config',
         glob('config/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='baranidharan',
    maintainer_email='selvambharani100@gmail.com',
    description='Logs /joint_states (position/velocity/effort), motor_status (mos_temp/rotor_temp) and gesture_status into per-session Parquet files, with CSV fallback and a web control panel (start/stop/new session, rename/delete, record toggles, replay trigger, timeline graphs).',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'joint_state_logger = joint_state_logger.joint_state_logger_node:main',
        ],
    },
)
