#!/usr/bin/env python3
"""从 bag 蓄水池抽样约 30 帧，拉起本包滤波节点（独立 ROS_DOMAIN），逐帧投喂并对照扫描线团。"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import List, Tuple

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import (
    ConverterOptions,
    SequentialReader,
    StorageFilter,
    StorageOptions,
)
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

ANG_V = 0.0174532925
LIM = dict(x=(0.0, 3.5), y=(-2.0, 2.0), z=(-0.2, 0.4))
CUBE = (2.0, 2.0, 1.0)  # half L/W/H of 4x4x2
CLUSTER_G = 0.12


def open_bag(uri: str) -> SequentialReader:
    last = None
    for opener in ("seq", "tmp"):
        try:
            path = uri
            if opener == "tmp":
                path = "/tmp/vanjee_bag_read"
            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=path, storage_id="sqlite3"),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
            )
            return reader
        except Exception as ex:
            last = ex
    raise RuntimeError(f"无法打开 bag: {uri} ({last})")


def reservoir_lidar_frames(bag_uri: str, k: int, seed: int):
    rng = random.Random(seed)
    Lidar = get_message("sensor_msgs/msg/PointCloud2")
    reader = open_bag(bag_uri)
    reader.set_filter(StorageFilter(topics=["/vanjee/lidar"]))
    reservoir = []
    n = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != "/vanjee/lidar":
            continue
        n += 1
        item = (n, t, data)
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(1, n)
            if j <= k:
                reservoir[j - 1] = item
    frames = []
    for idx, t, data in sorted(reservoir, key=lambda x: x[0]):
        msg = deserialize_message(data, Lidar)
        frames.append((idx, t, msg))
    return n, frames


def cloud_xyz_ring(msg: PointCloud2):
    names = [f.name for f in msg.fields]
    has_ring = "ring" in names
    has_i = "intensity" in names
    fields = ("x", "y", "z") + (("ring",) if has_ring else ()) + (("intensity",) if has_i else ())
    rows = []
    for p in pc2.read_points(msg, field_names=fields, skip_nans=False):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        ring = int(p[3]) if has_ring else -1
        inten = float(p[4]) if has_i else float("nan")
        rows.append((x, y, z, ring, inten))
    if not rows:
        return np.zeros((0, 5))
    return np.array(rows, dtype=np.float64)


def ring_geom(x, y, z):
    rxy = max(math.hypot(x, y), 1e-6)
    return int(round(math.atan2(z, rxy) / ANG_V))


def in_pass(p):
    return (LIM["x"][0] <= p[0] <= LIM["x"][1]
            and LIM["y"][0] <= p[1] <= LIM["y"][1]
            and LIM["z"][0] <= p[2] <= LIM["z"][1])


def in_cube(p):
    return abs(p[0]) <= CUBE[0] and abs(p[1]) <= CUBE[1] and abs(p[2]) <= CUBE[2]


def clusters(P: np.ndarray, RG: np.ndarray, g: float = CLUSTER_G):
    if len(P) == 0:
        return []
    cell = defaultdict(list)
    for i, p in enumerate(P):
        cell[(int(math.floor(p[0] / g)),
              int(math.floor(p[1] / g)),
              int(math.floor(p[2] / g)))].append(i)
    seen = set()
    out = []
    for k0 in cell:
        if k0 in seen:
            continue
        q = deque([k0])
        seen.add(k0)
        comp = []
        while q:
            c0 = q.popleft()
            comp.extend(cell[c0])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (c0[0] + dx, c0[1] + dy, c0[2] + dz)
                        if nk in cell and nk not in seen:
                            seen.add(nk)
                            q.append(nk)
        idx = np.array(comp, dtype=int)
        rings = sorted(set(int(r) for r in RG[idx].tolist() if r >= 0))
        if not rings:
            rings = sorted(set(ring_geom(*P[i]) for i in idx))
        out.append((idx, rings))
    out.sort(key=lambda x: -len(x[0]))
    return out


def stamp_key(msg: PointCloud2) -> Tuple[int, int]:
    return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))


def xyz_set(arr: np.ndarray):
    if len(arr) == 0:
        return set()
    q = np.round(arr[:, :3] * 1e4).astype(np.int64)
    return set(map(tuple, q.tolist()))


def start_node(params_file: str, domain: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain)
    cmd = [
        "ros2", "run", "vanjee_lidar_filter", "cloud_passthrough_filter_node",
        "--ros-args",
        "--params-file", params_file,
        "-p", "use_sim_time:=false",
        "-p", "input_topic:=/eval/lidar",
        "-p", "output_topic:=/eval/passthrough",
        "-p", "removed_topic:=/eval/removed",
        "-p", "debug_topic_x:=/eval/px",
        "-p", "debug_topic_y:=/eval/py",
        "-p", "debug_topic_z:=/eval/pz",
        "-p", "debug_mode:=true",
        "-r", "__node:=cloud_passthrough_eval",
    ]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--domain", type=int, default=87)
    args = ap.parse_args()

    print(f"抽样 bag={args.bag}  n={args.n} seed={args.seed} domain={args.domain}")
    total, frames = reservoir_lidar_frames(args.bag, args.n, args.seed)
    print(f"bag 中 /vanjee/lidar 共 {total} 帧，抽到 {len(frames)} 帧（按原序号）:")
    print("  " + ", ".join(str(i) for i, _, _ in frames))

    proc = start_node(args.params, args.domain)
    time.sleep(3.0)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        print("节点启动失败:\n" + out)
        sys.exit(1)

    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    import rclpy
    from rclpy.node import Node as RclNode

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )

    rclpy.init()
    node = RclNode("eval_random_frames_client")
    pub = node.create_publisher(PointCloud2, "/eval/lidar", qos)
    kept_box: List[PointCloud2] = []
    rem_box: List[PointCloud2] = []

    def on_kept(m):
        kept_box.append(m)

    def on_rem(m):
        rem_box.append(m)

    node.create_subscription(PointCloud2, "/eval/passthrough", on_kept, qos)
    node.create_subscription(PointCloud2, "/eval/removed", on_rem, qos)
    t_wait = time.time()
    while pub.get_subscription_count() < 1 and time.time() - t_wait < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() < 1:
        print("等待 /eval/lidar 订阅超时，节点可能没起来")
        proc.terminate()
        rclpy.shutdown()
        sys.exit(1)
    print(f"节点已订阅 /eval/lidar (count={pub.get_subscription_count()})")
    time.sleep(0.3)

    rows = []
    try:
        for fi, (idx, t_ns, msg) in enumerate(frames, 1):
            src = cloud_xyz_ring(msg)
            P = src[:, :3]
            RG = src[:, 3].astype(int)
            pass_m = np.array([in_pass(p) for p in P], dtype=bool) if len(P) else np.array([], dtype=bool)
            cube_m = np.array([in_cube(p) for p in P], dtype=bool) if len(P) else np.array([], dtype=bool)
            work = pass_m & cube_m
            Pw, RGw = (P[work], RG[work]) if work.any() else (np.zeros((0, 3)), np.zeros((0,), dtype=int))
            clus = clusters(Pw, RGw)

            kept_box.clear()
            rem_box.clear()
            key = stamp_key(msg)
            pub.publish(msg)
            t0 = time.time()
            got_k = got_r = False
            kept_msg = rem_msg = None
            published = 1
            while time.time() - t0 < 6.0:
                rclpy.spin_once(node, timeout_sec=0.05)
                for m in kept_box:
                    if stamp_key(m) == key:
                        kept_msg = m
                        got_k = True
                for m in rem_box:
                    if stamp_key(m) == key:
                        rem_msg = m
                        got_r = True
                if got_k and (got_r or time.time() - t0 > 0.8):
                    break
                if (not got_k) and time.time() - t0 > 1.2 * published:
                    pub.publish(msg)
                    published += 1
            if not got_k:
                print(f"  [{fi:02d}] 帧#{idx} 超时未收到输出 stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
                rows.append(None)
                continue

            kept_all = cloud_xyz_ring(kept_msg)
            kept_xyz = kept_all[:, :3] if len(kept_all) else np.zeros((0, 3))
            rem_all = cloud_xyz_ring(rem_msg) if rem_msg is not None else np.zeros((0, 5))
            rem_xyz = rem_all[:, :3] if len(rem_all) else np.zeros((0, 3))
            # 被删点以输出为准反推：removed 话题是 best-effort，可能丢帧
            ks = xyz_set(kept_xyz)
            rs = xyz_set(rem_xyz)

            n_single = n_multi = 0
            del_single = keep_single = 0
            del_multi = keep_multi = 0
            leak_single = []
            hurt_multi = []
            for cidx, rings in clus:
                n = len(cidx)
                n_del = 0
                n_keep = 0
                for i in cidx:
                    keyp = tuple(np.round(Pw[i] * 1e4).astype(np.int64).tolist())
                    if keyp in ks:
                        n_keep += 1
                    else:
                        n_del += 1
                if len(rings) <= 1:
                    n_single += n
                    del_single += n_del
                    keep_single += n_keep
                    if n_keep > 0:
                        leak_single.append((n, n_keep, n_del, rings))
                else:
                    n_multi += n
                    del_multi += n_del
                    keep_multi += n_keep
                    if n_del > 0:
                        hurt_multi.append((n, n_keep, n_del, rings))

            stamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            print(
                f"  [{fi:02d}] 原序号#{idx:<4d} t={stamp}  "
                f"finite={len(P):5d} pass∩cube={len(Pw):4d} 输出={len(kept_xyz):4d} "
                f"(removed话题={len(rem_xyz):4d})  "
                f"单线团删 {del_single}/{n_single}  "
                f"多线团误删 {del_multi}/{n_multi}"
            )
            if leak_single:
                top = sorted(leak_single, key=lambda x: -x[1])[:2]
                print("        漏删单线团: " + "; ".join(
                    f"n={a} 留={b} 删={c} 线={d}" for a, b, c, d in top))
            if hurt_multi:
                top = sorted(hurt_multi, key=lambda x: -x[2])[:2]
                print("        误删多线团: " + "; ".join(
                    f"n={a} 留={b} 删={c} 线数={len(d)}" for a, b, c, d in top))
            rows.append(dict(
                idx=idx, stamp=stamp, n_in=len(P), n_work=len(Pw),
                n_kept=len(kept_xyz), n_rem=len(rem_xyz),
                n_single=n_single, del_single=del_single,
                n_multi=n_multi, del_multi=del_multi,
            ))
    finally:
        node.destroy_node()
        rclpy.shutdown()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    ok = [r for r in rows if r]
    print("\n======== 汇总 ========")
    print(f"成功处理 {len(ok)}/{len(frames)} 帧")
    if not ok:
        sys.exit(2)
    s_del = sum(r["del_single"] for r in ok)
    s_all = sum(r["n_single"] for r in ok)
    m_del = sum(r["del_multi"] for r in ok)
    m_all = sum(r["n_multi"] for r in ok)
    print(f"单线团（目标该删）: 删 {s_del} / {s_all} 点  ({100.0*s_del/max(s_all,1):.1f}%)")
    print(f"多线团（目标该留）: 误删 {m_del} / {m_all} 点  ({100.0*m_del/max(m_all,1):.1f}%)")
    print(f"输出合计: 留 {sum(r['n_kept'] for r in ok)}  删 {sum(r['n_rem'] for r in ok)}")


if __name__ == "__main__":
    main()
