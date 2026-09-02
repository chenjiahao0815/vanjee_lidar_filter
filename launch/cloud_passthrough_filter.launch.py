from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vanjee_lidar_filter')
    default_params = os.path.join(pkg_share, 'config', 'cloud_passthrough_filter.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to cloud_passthrough_filter params yaml',
    )

    node = Node(
        package='vanjee_lidar_filter',
        executable='cloud_passthrough_filter_node',
        name='cloud_passthrough_filter_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        node,
    ])
