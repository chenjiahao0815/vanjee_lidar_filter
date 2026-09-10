#!/usr/bin/env bash
# 编译 vanjee_lidar_filter，并在当前 shell 里 source。
# ros2 bag play /home/linux/vehicle_total/vajee_lidar/wanjee_bag/vanjee_noise_motion_20260902_143028 --clock --rate 0.5 --loop
#   source /home/linux/ros_cpp/vanjee_lidar_filter/build.sh
#   或source ./build.sh

_VLF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_VLF_ROOT" || return 1 2>/dev/null || exit 1

if [[ -z "${ROS_DISTRO:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

# 清掉"指向已不存在的旧工作区"的残留路径。
# 由来：以前在顶层 vajee_lidar/ 下跑过 colcon 并 source 过它的 install/setup.bash，
# 之后顶层 build/install/log 被删掉了，但当前 shell 的环境变量还留着那些路径，
# colcon 就会刷一堆 "The path '.../vajee_lidar/install' ... doesn't exist" 警告。
# 这里把不存在的条目摘掉再编译，效果等同于开一个新终端。
_vlf_prune_prefix() {
  local old="${1:-}" out=""
  [[ -z "$old" ]] && return 0
  local entry
  local IFS=':'
  for entry in $old; do
    [[ -z "$entry" ]] && continue
    [[ -e "$entry" ]] && out="${out:+$out:}$entry"
  done
  shift
  export "$1=$out"
}
for _VLF_V in COLCON_PREFIX_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH; do
  _vlf_prune_prefix "${!_VLF_V:-}" "$_VLF_V"
done
unset _VLF_V

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
