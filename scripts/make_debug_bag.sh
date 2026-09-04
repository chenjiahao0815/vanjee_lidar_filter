#!/usr/bin/env bash
# 编译滤波节点，抽帧跑一遍，把判定结果烧成一个可循环回放的可视化包。
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "${ROOT}/.." && pwd)"
BAG="${1:-${WS}/wanjee_bag/vanjee_noise_motion_20260902_143028}"
N_FRAMES="${2:-30}"
SEED="${3:-20260903}"
OUT="${4:-${WS}/wanjee_bag/vanjee_debug_view}"
DOMAIN="${ROS_DOMAIN_ID_EVAL:-92}"

if [[ -d /tmp/vanjee_bag_read && -f /tmp/vanjee_bag_read/metadata.yaml ]]; then
  BAG_READ="/tmp/vanjee_bag_read"
else
  BAG_READ="${BAG}"
fi

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "未找到 ROS 2 Humble" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [[ -f "${WS}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS}/install/setup.bash"
fi

echo "==> 编译 vanjee_lidar_filter"
cd "${WS}"
colcon build --packages-select vanjee_lidar_filter --cmake-args -DCMAKE_BUILD_TYPE=Release
# shellcheck disable=SC1091
source "${WS}/install/setup.bash"

PARAMS="${WS}/install/vanjee_lidar_filter/share/vanjee_lidar_filter/config/cloud_passthrough_filter.yaml"

echo "==> 抽 ${N_FRAMES} 帧生成可视化包 (ROS_DOMAIN_ID=${DOMAIN}，不影响当前正在播的包)"
python3 "${ROOT}/scripts/make_debug_bag.py" \
  --bag "${BAG_READ}" \
  --params "${PARAMS}" \
  --out "${OUT}" \
  --n "${N_FRAMES}" \
  --seed "${SEED}" \
  --domain "${DOMAIN}"

cat <<EOF

==> 看结果（两个终端）
  rviz2 -d ${ROOT}/rviz/vanjee_debug_view.rviz
  ros2 bag play ${OUT} -l -p

  空格 = 播放/暂停，s = 单步（一次一条消息，5 条 = 一帧）
EOF
