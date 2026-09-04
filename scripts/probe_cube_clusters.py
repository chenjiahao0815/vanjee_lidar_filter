#!/usr/bin/env python3
"""列出每帧 pass∩cube 里的点团，给出 r / z跨度 / 线数 / XY尺寸，用来定团保护门槛。"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, "/home/linux/vehicle_total/vajee_lidar/vanjee_lidar_filter/scripts")
from eval_random_frames import (  # noqa: E402
    clusters,
    cloud_xyz_ring,
    in_cube,
    in_pass,
    open_bag,
)
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402
from rosbag2_py import StorageFilter  # noqa: E402


def describe(P, RG):
    r = np.hypot(P[:, 0], P[:, 1])
    rings = sorted(set(int(v) for v in RG.tolist() if v >= 0))
    return dict(
        n=len(P),
        r_min=float(r.min()), r_max=float(r.max()),
        x=float(P[:, 0].mean()), y=float(P[:, 1].mean()), z=float(P[:, 2].mean()),
        span_z=float(P[:, 2].max() - P[:, 2].min()),
        ext_x=float(P[:, 0].max() - P[:, 0].min()),
        ext_y=float(P[:, 1].max() - P[:, 1].min()),
        span_r=float(r.max() - r.min()),
        nring=len(rings),
        rings=rings,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/tmp/vanjee_bag_read")
    ap.add_argument("--target", type=int, default=0, help="只打印 pass∩cube 点数接近该值的帧")
    ap.add_argument("--tol", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("--gap", type=float, default=0.12)
    args = ap.parse_args()

    Lidar = get_message("sensor_msgs/msg/PointCloud2")
    reader = open_bag(args.bag)
    reader.set_filter(StorageFilter(topics=["/vanjee/lidar"]))

    n = 0
    shown = 0
    while reader.has_next() and n < args.max_frames:
        topic, data, _t = reader.read_next()
        if topic != "/vanjee/lidar":
            continue
        n += 1
        msg = deserialize_message(data, Lidar)
        A = cloud_xyz_ring(msg)
        if len(A) == 0:
            continue
        m = np.array([in_pass(p) and in_cube(p) for p in A], dtype=bool)
        if not m.any():
            continue
        P = A[m][:, :3]
        RG = A[m][:, 3].astype(int)
        if args.target and abs(len(P) - args.target) > args.tol:
            continue

        cl = clusters(P, RG, g=args.gap)
        print(f"\n=== 帧#{n}  stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}  "
              f"pass∩cube={len(P)}  团数={len(cl)}")
        for ci, (idx, _rings) in enumerate(cl[:8], 1):
            d = describe(P[idx], RG[idx])
            print(f"  团{ci}: n={d['n']:4d} 中心=({d['x']:+.2f},{d['y']:+.2f},{d['z']:+.2f}) "
                  f"r={d['r_min']:.2f}~{d['r_max']:.2f} spanZ={d['span_z']:.3f} "
                  f"extXY=({d['ext_x']:.2f},{d['ext_y']:.2f}) spanR={d['span_r']:.2f} "
                  f"线数={d['nring']} rings={d['rings'][:12]}")
        shown += 1
        if shown >= 12:
            break
    print(f"\n扫过 {n} 帧，打印 {shown} 帧")


if __name__ == "__main__":
    main()
