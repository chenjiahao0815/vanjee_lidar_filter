#!/usr/bin/env python3
"""指定帧号投喂滤波节点，逐团报告删了多少、留了多少，用来对着截图核对。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import StorageFilter
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, "/home/linux/vehicle_total/vajee_lidar/vanjee_lidar_filter/scripts")
from eval_random_frames import (  # noqa: E402
    clusters,
    cloud_xyz_ring,
    in_cube,
    in_pass,
    open_bag,
    stamp_key,
    xyz_set,
)


def rewrite_params_wildcard(src: str, dst: str) -> str:
    """--params-file 按节点名匹配；这里改名跑，所以把键换成 /** 通配，否则 yaml 整份失效。"""
    import yaml
    with open(src) as f:
        doc = yaml.safe_load(f)
    params = None
    for _k, v in (doc or {}).items():
        if isinstance(v, dict) and "ros__parameters" in v:
            params = v["ros__parameters"]
            break
    if params is None:
        raise RuntimeError(f"{src} 里找不到 ros__parameters")
    with open(dst, "w") as f:
        yaml.safe_dump({"/**": {"ros__parameters": params}}, f,
                       allow_unicode=True, default_flow_style=False)
    return dst


def start_node(params_file: str, domain: int, log_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain)
    cmd = [
        "ros2", "run", "vanjee_lidar_filter", "cloud_passthrough_filter_node",
        "--ros-args", "--params-file", params_file,
        "-p", "use_sim_time:=false",
        "-p", "input_topic:=/verify/lidar",
        "-p", "output_topic:=/verify/passthrough",
        "-p", "removed_topic:=/verify/removed",
        "-p", "debug_topic_x:=/verify/px",
        "-p", "debug_topic_y:=/verify/py",
        "-p", "debug_topic_z:=/verify/pz",
        "-p", "verdict_topic:=/verify/verdict",
        "-p", "boxes_topic:=/verify/boxes",
        "-p", "voxels_topic:=/verify/voxels",
        "-r", "__node:=cloud_passthrough_verify",
    ]
    # debug_mode 下节点刷屏很凶，走文件；用 PIPE 会把管道塞满把节点卡死
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/tmp/vanjee_bag_read")
    ap.add_argument("--params", required=True)
    ap.add_argument("--frames", default="", help="逗号分隔的帧号")
    ap.add_argument("--random", type=int, default=0, help="随机抽这么多帧做回归")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--min-h", type=float, default=0.08, help="判定“该留”的最小物理高度")
    ap.add_argument("--domain", type=int, default=93)
    ap.add_argument("--gap", type=float, default=0.12)
    args = ap.parse_args()

    Lidar = get_message("sensor_msgs/msg/PointCloud2")
    if args.random > 0:
        import random
        rng = random.Random(args.seed)
        reader = open_bag(args.bag)
        reader.set_filter(StorageFilter(topics=["/vanjee/lidar"]))
        total = 0
        while reader.has_next():
            topic, _d, _t = reader.read_next()
            if topic == "/vanjee/lidar":
                total += 1
        want = sorted(rng.sample(range(1, total + 1), min(args.random, total)))
    else:
        want = sorted(int(s) for s in args.frames.split(","))

    reader = open_bag(args.bag)
    reader.set_filter(StorageFilter(topics=["/vanjee/lidar"]))
    picked = []
    n = 0
    while reader.has_next() and len(picked) < len(want):
        topic, data, _t = reader.read_next()
        if topic != "/vanjee/lidar":
            continue
        n += 1
        if n in want:
            picked.append((n, deserialize_message(data, Lidar)))
    print(f"取到 {len(picked)} 帧: {[i for i, _ in picked]}")

    log_path = f"/tmp/verify_node_{args.domain}.log"
    params_file = rewrite_params_wildcard(
        args.params, f"/tmp/verify_params_{args.domain}.yaml")
    proc = start_node(params_file, args.domain, log_path)
    time.sleep(3.0)
    print(f"节点日志: {log_path}")
    if proc.poll() is not None:
        with open(log_path) as f:
            print("节点启动失败:\n" + f.read())
        sys.exit(1)

    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    import rclpy
    from rclpy.node import Node as RclNode

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    rclpy.init()
    node = RclNode("verify_frames_client")
    pub = node.create_publisher(PointCloud2, "/verify/lidar", qos)
    box = []
    node.create_subscription(PointCloud2, "/verify/passthrough", box.append, qos)

    t0 = time.time()
    while pub.get_subscription_count() < 1 and time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() < 1:
        print("等待订阅超时")
        proc.terminate()
        sys.exit(1)
    time.sleep(0.4)

    ok = True
    # 汇总：该留=真实线数≥2 且 spanZ≥min_h；其余算该删
    agg = dict(keep_all=0, keep_del=0, drop_all=0, drop_kept=0)
    try:
        for idx, msg in picked:
            A = cloud_xyz_ring(msg)
            m = np.array([in_pass(p) and in_cube(p) for p in A], dtype=bool)
            P, RG = A[m][:, :3], A[m][:, 3].astype(int)

            box.clear()
            key = stamp_key(msg)
            pub.publish(msg)
            t1 = time.time()
            out = None
            while time.time() - t1 < 8.0:
                rclpy.spin_once(node, timeout_sec=0.05)
                for k in box:
                    if stamp_key(k) == key:
                        out = k
                if out is not None:
                    break
                if time.time() - t1 > 1.5:
                    pub.publish(msg)
                    t1 = time.time()
            if out is None:
                print(f"帧#{idx} 超时")
                ok = False
                continue

            kept = cloud_xyz_ring(out)
            ks = xyz_set(kept[:, :3] if len(kept) else np.zeros((0, 3)))
            n_keep_cube = sum(1 for p in (kept[:, :3] if len(kept) else [])
                              if in_cube(p) and in_pass(p))
            print(f"\n=== 帧#{idx}  pass∩cube={len(P)}  输出总数={len(kept)}  "
                  f"输出中 pass∩cube={n_keep_cube}")
            for ci, (cidx, _r) in enumerate(clusters(P, RG, g=args.gap)[:8], 1):
                pts = P[cidx]
                rr = np.hypot(pts[:, 0], pts[:, 1])
                rings = sorted(set(int(v) for v in RG[cidx].tolist() if v >= 0))
                nk = sum(1 for i in cidx
                         if tuple(np.round(P[i] * 1e4).astype(np.int64).tolist()) in ks)
                nd = len(cidx) - nk
                verdict = "全删" if nk == 0 else ("全留" if nd == 0 else "部分")
                span = float(pts[:, 2].max() - pts[:, 2].min())
                should_keep = len(rings) >= 2 and span >= args.min_h
                if should_keep:
                    agg["keep_all"] += len(cidx)
                    agg["keep_del"] += nd
                else:
                    agg["drop_all"] += len(cidx)
                    agg["drop_kept"] += nk
                print(f"  团{ci}: n={len(cidx):4d} 中心=({pts[:,0].mean():+.2f},"
                      f"{pts[:,1].mean():+.2f},{pts[:,2].mean():+.2f}) "
                      f"r={rr.min():.2f}~{rr.max():.2f} spanZ={span:.3f} "
                      f"真实线数={len(rings)} {'该留' if should_keep else '该删'} "
                      f"→ 留{nk} 删{nd} [{verdict}]")
        print("\n======== 汇总 ========")
        print(f"该留(真实线数≥2 且 spanZ≥{args.min_h:.2f}): 误删 "
              f"{agg['keep_del']} / {agg['keep_all']} 点")
        print(f"该删(单线 或 spanZ<{args.min_h:.2f}):        漏删 "
              f"{agg['drop_kept']} / {agg['drop_all']} 点")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
