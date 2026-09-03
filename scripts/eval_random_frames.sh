#!/usr/bin/env bash
# 编译滤波节点，从 bag 随机抽约 30 帧（蓄水池，不连号），用独立 ROS_DOMAIN 跑节点并对照删点。
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "${ROOT}/.." && pwd)"
BAG="${1:-${WS}/wanjee_bag/vanjee_noise_motion_20260902_143028}"
N_FRAMES="${2:-30}"
SEED="${3:-20260903}"
DOMAIN="${ROS_DOMAIN_ID_EVAL:-87}"

if [[ ! -d "${BAG}" ]]; then
  echo "找不到 bag 目录: ${BAG}" >&2
  exit 1
fi

# 压缩 bag 优先用已解压副本，避免抽样时再解压
if [[ -d /tmp/vanjee_bag_read && -f /tmp/vanjee_bag_read/metadata.yaml ]]; then
  BAG_READ="/tmp/vanjee_bag_read"
else
  BAG_READ="${BAG}"
fi

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "未找到 ROS 2 Humble，无法用本包节点离线跑帧。" >&2
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
PY="${ROOT}/scripts/eval_random_frames.py"
if [[ ! -f "${PY}" ]]; then
  PY="$(ros2 pkg prefix vanjee_lidar_filter)/share/vanjee_lidar_filter/scripts/eval_random_frames.py"
fi

echo "==> 随机抽 ${N_FRAMES} 帧并投喂节点 (ROS_DOMAIN_ID=${DOMAIN}，不影响当前正在播的包)"
exec python3 "${PY}" \
  --bag "${BAG_READ}" \
  --params "${PARAMS}" \
  --n "${N_FRAMES}" \
  --seed "${SEED}" \
  --domain "${DOMAIN}"
