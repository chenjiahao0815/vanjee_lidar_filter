#!/usr/bin/env python3
"""扫描 bag，找直通内但立方体外的稀疏单线团（这些点体素滤波根本不处理）。"""
from __future__ import annotations

import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/linux/vehicle_total/vajee_lidar/vanjee_lidar_filter/scripts")
from eval_random_frames import (  # noqa: E402
    CUBE,
    LIM,
    clusters,
    cloud_xyz_ring,
    in_cube,
    in_pass,
    open_bag,
    ring_geom,
)
from rosidl_runtime_py.utilities import get_message  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosbag2_py import StorageFilter  # noqa: E402

BAG = "/tmp/vanjee_bag_read"


def main() -> None:
    Lidar = get_message("sensor_msgs/msg/PointCloud2")
    reader = open_bag(BAG)
    reader.set_filter(StorageFilter(topics=["/vanjee/lidar"]))

    n = 0
    stat_out = defaultdict(int)   # 直通内 + 立方体外
    stat_in = defaultdict(int)    # 直通内 + 立方体内
    samples_out = []
    frames_with_out = 0

    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic != "/vanjee/lidar":
            continue
        n += 1
        if n % 4 != 1:           # 每 4 帧看 1 帧，够统计
            continue
        msg = deserialize_message(data, Lidar)
        A = cloud_xyz_ring(msg)
        if len(A) == 0:
            continue

        keep = np.array([in_pass(p) for p in A], dtype=bool)
        P = A[keep]
        if len(P) == 0:
            continue
        RG = P[:, 3].astype(int)   # 用 bag 里的 ring 字段，和评估脚本一致

        outside = np.array([not in_cube(p) for p in P], dtype=bool)
        Pout, RGout = P[outside], RG[outside]
        got_out = False
        for idx, rings in clusters(Pout, RGout):
            if len(rings) != 1 or len(idx) > 60:
                continue
            pts = Pout[idx]
            r = float(np.mean(np.hypot(pts[:, 0], pts[:, 1])))
            stat_out[1] += len(idx)
            got_out = True
            if len(samples_out) < 12:
                samples_out.append(
                    (n, len(idx), rings[0], r,
                     float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])),
                     float(np.mean(pts[:, 2])), float(np.mean(pts[:, 4]))))
        if got_out:
            frames_with_out += 1

        Pin, RGin = P[~outside], RG[~outside]
        for idx, rings in clusters(Pin, RGin):
            if len(rings) == 1:
                stat_in[1] += len(idx)

    print(f"扫过 {n} 帧，采样 {n // 4 + 1} 帧")
    print(f"直通内·立方体外 单线稀疏团点数合计: {sum(stat_out.values())}"
          f"  出现在 {frames_with_out} 个采样帧")
    print(f"直通内·立方体内 单线团点数合计: {sum(stat_in.values())}")
    print("\n样例 (帧, 点数, ring, r, x, y, z, intensity):")
    for s in samples_out:
        print(f"  #{s[0]:<5} n={s[1]:<3} ring={s[2]:<3} r={s[3]:.2f} "
              f"x={s[4]:+.2f} y={s[5]:+.2f} z={s[6]:+.2f} I={s[7]:.0f}")


if __name__ == "__main__":
    main()
