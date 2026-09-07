#!/usr/bin/env bash
# 编译 vanjee_lidar_filter，并在当前 shell 里 source。
# 用法（必须 source，否则环境不会留在当前终端）：
#   source /home/linux/ros_cpp/vanjee_lidar_filter/build.sh
#   或source ./build.sh

_VLF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_VLF_ROOT" || return 1 2>/dev/null || exit 1

if [[ -z "${ROS_DISTRO:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

echo "[build.sh] 开始编译: $_VLF_ROOT"
colcon build --packages-select vanjee_lidar_filter --symlink-install
_VLF_RC=$?

if [[ $_VLF_RC -ne 0 ]]; then
  echo "[build.sh] 编译失败 (exit=$_VLF_RC)"
  unset _VLF_ROOT _VLF_RC
  return $_VLF_RC 2>/dev/null || exit $_VLF_RC
fi

# install(DIRECTORY) 仍会复制 yaml；改成软链，避免一编译就把 install 里手改的参数盖掉
_VLF_SHARE="$_VLF_ROOT/install/vanjee_lidar_filter/share/vanjee_lidar_filter/config"
_VLF_YAML="$_VLF_SHARE/cloud_passthrough_filter.yaml"
if [[ -d "$_VLF_SHARE" && -f "$_VLF_ROOT/config/cloud_passthrough_filter.yaml" ]]; then
  ln -sfn "$_VLF_ROOT/config/cloud_passthrough_filter.yaml" "$_VLF_YAML"
fi

# shellcheck disable=SC1091
source "$_VLF_ROOT/install/setup.bash"
echo "[build.sh] 编译完成，已 source: $_VLF_ROOT/install/setup.bash"

unset _VLF_ROOT _VLF_RC _VLF_SHARE _VLF_YAML
return 0 2>/dev/null || exit 0
