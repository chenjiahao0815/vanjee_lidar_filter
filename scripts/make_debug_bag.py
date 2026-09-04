#!/usr/bin/env python3
"""抽帧 -> 喂给当前编译出的滤波节点 -> 把结果和判定配色烧进一个新包，供 RViz 循环回放。

写出的话题（颜色已烧进点云 rgb 字段，不依赖 RViz 配色）：
  /dbg/raw      所有有限点，深灰，给个环境参照
  /dbg/kept     节点输出（保留下来的点），绿
  /dbg/removed  以输出反推的被删点，红
  /dbg/verdict  判定配色，只含直通内的点：
                  红   = 正确删（单线团被删）
                  黄   = 漏删（单线团没删掉）  <- 最该看的
                  品红 = 误删（多线团被删）    <- 最该看的
                  绿   = 正确留（多线团保留）
                  灰蓝 = 立方体外，未参与判定
  /dbg/markers  立方体线框（青）、直通框线框（橙）、每帧统计文字

时间戳重写成每帧固定间隔的合成时间轴，方便 `--loop` / `--start-paused` 逐帧步进。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_random_frames import (  # noqa: E402
    CUBE,
    LIM,
    clusters,
    cloud_xyz_ring,
    in_cube,
    in_pass,
    open_bag,
    start_node,
    stamp_key,
    xyz_set,
)

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402
from rosbag2_py import (  # noqa: E402
    ConverterOptions,
    SequentialWriter,
    StorageFilter,
    StorageOptions,
    TopicMetadata,
)
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

RGB_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])

C_RAW = (60, 60, 60)
C_KEPT = (40, 220, 80)
C_REMOVED = (240, 40, 40)
C_OK_DEL = (240, 40, 40)      # 正确删
C_LEAK = (255, 235, 0)        # 漏删
C_FALSE_DEL = (255, 0, 220)   # 误删
C_OK_KEEP = (40, 220, 80)     # 正确留
C_UNJUDGED = (90, 120, 170)   # 立方体外

TOPICS = {
    "/dbg/raw": "sensor_msgs/msg/PointCloud2",
    "/dbg/kept": "sensor_msgs/msg/PointCloud2",
    "/dbg/removed": "sensor_msgs/msg/PointCloud2",
    "/dbg/verdict": "sensor_msgs/msg/PointCloud2",
    "/dbg/markers": "visualization_msgs/msg/MarkerArray",
}


def pack(rgb: Tuple[int, int, int]) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def rgb_cloud(header: Header, xyz: np.ndarray, colors: np.ndarray) -> PointCloud2:
    n = len(xyz)
    arr = np.zeros(n, dtype=RGB_DT)
    if n:
        arr["x"] = xyz[:, 0]
        arr["y"] = xyz[:, 1]
        arr["z"] = xyz[:, 2]
        arr["rgb"] = colors
    m = PointCloud2()
    m.header = header
    m.height = 1
    m.width = n
    m.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        # 字节里放的是打包的 uint32，声明成 FLOAT32 是 PCL 的惯例，RViz 认这个
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    m.is_bigendian = False
    m.point_step = RGB_DT.itemsize
    m.row_step = RGB_DT.itemsize * n
    m.data = arr.tobytes()
    m.is_dense = True
    return m


def box_marker(header: Header, ns: str, mid: int,
               xr: Tuple[float, float], yr: Tuple[float, float], zr: Tuple[float, float],
               rgb: Tuple[int, int, int], width: float = 0.015) -> Marker:
    from geometry_msgs.msg import Point
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = mid
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = width
    m.color.r = rgb[0] / 255.0
    m.color.g = rgb[1] / 255.0
    m.color.b = rgb[2] / 255.0
    m.color.a = 0.9
    c = [(xr[i], yr[j], zr[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    # 顶点索引 = i*4 + j*2 + k
    edges = [(0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        for v in (c[a], c[b]):
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            m.points.append(p)
    return m


def text_marker(header: Header, ns: str, mid: int, text: str,
                pos: Tuple[float, float, float], rgb: Tuple[int, int, int],
                size: float = 0.12) -> Marker:
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
    m.scale.z = size
    m.color.r = rgb[0] / 255.0
    m.color.g = rgb[1] / 255.0
    m.color.b = rgb[2] / 255.0
    m.color.a = 1.0
    m.text = text
    return m


def pick_frames(bag_uri: str, k: int, seed: int, extras: List[int]):
    """蓄水池抽样 k 帧（和评估脚本同 seed 同结果），再强行并入 extras 指定的原序号。"""
    import random
    rng = random.Random(seed)
    Lidar = get_message("sensor_msgs/msg/PointCloud2")
    reader = open_bag(bag_uri)
    reader.set_filter(StorageFilter(topics=["/vanjee/lidar"]))
    want = set(extras)
    reservoir: List[Tuple[int, int, bytes]] = []
    forced: Dict[int, Tuple[int, int, bytes]] = {}
    n = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != "/vanjee/lidar":
            continue
        n += 1
        item = (n, t, data)
        if n in want:
            forced[n] = item
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(1, n)
            if j <= k:
                reservoir[j - 1] = item
    picked: Dict[int, Tuple[int, int, bytes]] = {it[0]: it for it in reservoir}
    picked.update(forced)
    out = []
    for idx in sorted(picked):
        _, t, data = picked[idx]
        out.append((idx, t, deserialize_message(data, Lidar)))
    return n, out


def open_writer(uri: str) -> SequentialWriter:
    w = SequentialWriter()
    w.open(StorageOptions(uri=uri, storage_id="sqlite3"),
           ConverterOptions(input_serialization_format="cdr",
                            output_serialization_format="cdr"))
    for name, typ in TOPICS.items():
        try:
            meta = TopicMetadata(name=name, type=typ, serialization_format="cdr")
        except TypeError:
            meta = TopicMetadata(id=0, name=name, type=typ, serialization_format="cdr")
        w.create_topic(meta)
    return w


def wait_output(node, pub, msg, kept_box, timeout=12.0) -> Optional[PointCloud2]:
    import rclpy
    kept_box.clear()
    key = stamp_key(msg)
    pub.publish(msg)
    t0 = time.time()
    published = 1
    while time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        for m in kept_box:
            if stamp_key(m) == key:
                return m
        if time.time() - t0 > 1.5 * published:
            pub.publish(msg)
            published += 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/tmp/vanjee_bag_read")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="/home/linux/vehicle_total/vajee_lidar/wanjee_bag/vanjee_debug_view")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--extra", default="306,317,327", help="强行并入的原帧号，逗号分隔")
    ap.add_argument("--domain", type=int, default=91)
    ap.add_argument("--period", type=float, default=1.0, help="回放里每帧间隔秒")
    args = ap.parse_args()

    extras = [int(s) for s in args.extra.split(",") if s.strip()]
    total, frames = pick_frames(args.bag, args.n, args.seed, extras)
    print(f"bag 共 {total} 帧，取 {len(frames)} 帧: "
          + ", ".join(str(i) for i, _, _ in frames))

    if os.path.exists(args.out):
        print(f"输出目录已存在，先删掉: {args.out}")
        subprocess.run(["rm", "-rf", args.out], check=True)

    proc = start_node(args.params, args.domain)
    time.sleep(3.0)
    if proc.poll() is not None:
        print("节点启动失败:\n" + (proc.stdout.read() if proc.stdout else ""))
        sys.exit(1)

    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    import rclpy
    from rclpy.node import Node as RclNode

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    rclpy.init()
    node = RclNode("make_debug_bag_client")
    pub = node.create_publisher(PointCloud2, "/eval/lidar", qos)
    kept_box: List[PointCloud2] = []
    sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=30)
    node.create_subscription(PointCloud2, "/eval/passthrough", kept_box.append, sub_qos)

    t_wait = time.time()
    while pub.get_subscription_count() < 1 and time.time() - t_wait < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() < 1:
        print("等待节点订阅超时")
        proc.terminate()
        rclpy.shutdown()
        sys.exit(1)
    time.sleep(0.3)

    # 第一趟：只喂帧收结果。写包放到第二趟，否则写包耗时会挤掉节点回包的窗口
    outputs: Dict[int, np.ndarray] = {}
    try:
        pending = list(frames)
        for attempt in (1, 2, 3):
            if not pending:
                break
            if attempt > 1:
                print(f"  第 {attempt} 轮重试 {len(pending)} 帧: "
                      + ", ".join(str(i) for i, _, _ in pending))
            still: List[Tuple[int, int, PointCloud2]] = []
            for idx, t_ns, msg in pending:
                kept_msg = wait_output(node, pub, msg, kept_box)
                if kept_msg is None:
                    still.append((idx, t_ns, msg))
                    continue
                kept_all = cloud_xyz_ring(kept_msg)
                outputs[idx] = kept_all[:, :3] if len(kept_all) else np.zeros((0, 3))
            pending = still
        if pending:
            print("  最终仍超时: " + ", ".join(str(i) for i, _, _ in pending))
    finally:
        node.destroy_node()
        rclpy.shutdown()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    # 第二趟：算判定、写包
    writer = open_writer(args.out)
    t_base = int(time.time()) * 1_000_000_000
    dt = int(args.period * 1e9)

    tot = dict(leak=0, single=0, fdel=0, multi=0, unjudged=0, frames=0)
    try:
        fi = -1
        for idx, _t_ns, msg in frames:
            if idx not in outputs:
                continue
            fi += 1
            src = cloud_xyz_ring(msg)
            if len(src) == 0:
                continue
            P = src[:, :3]
            RG = src[:, 3].astype(int)
            pass_m = np.array([in_pass(p) for p in P], dtype=bool)
            cube_m = np.array([in_cube(p) for p in P], dtype=bool)
            work = pass_m & cube_m

            kept_xyz = outputs[idx]
            ks = xyz_set(kept_xyz)

            Pw, RGw = P[work], RG[work]
            del_mask = np.ones(len(Pw), dtype=bool)
            for i in range(len(Pw)):
                keyp = tuple(np.round(Pw[i] * 1e4).astype(np.int64).tolist())
                if keyp in ks:
                    del_mask[i] = False

            vcolor = np.zeros(len(Pw), dtype=np.uint32)
            n_leak = n_single = n_fdel = n_multi = 0
            for cidx, rings in clusters(Pw, RGw):
                single = len(rings) <= 1
                for i in cidx:
                    if single:
                        n_single += 1
                        if del_mask[i]:
                            vcolor[i] = pack(C_OK_DEL)
                        else:
                            vcolor[i] = pack(C_LEAK)
                            n_leak += 1
                    else:
                        n_multi += 1
                        if del_mask[i]:
                            vcolor[i] = pack(C_FALSE_DEL)
                            n_fdel += 1
                        else:
                            vcolor[i] = pack(C_OK_KEEP)

            outside = pass_m & ~cube_m
            Po = P[outside]
            v_xyz = np.vstack([Pw, Po]) if len(Po) else Pw
            v_col = np.concatenate([vcolor, np.full(len(Po), pack(C_UNJUDGED), dtype=np.uint32)])

            stamp_ns = t_base + fi * dt
            hdr = Header()
            hdr.frame_id = msg.header.frame_id
            hdr.stamp.sec = stamp_ns // 1_000_000_000
            hdr.stamp.nanosec = stamp_ns % 1_000_000_000

            rem_xyz = Pw[del_mask]
            writer.write("/dbg/raw", serialize_message(
                rgb_cloud(hdr, P, np.full(len(P), pack(C_RAW), dtype=np.uint32))), stamp_ns)
            writer.write("/dbg/kept", serialize_message(
                rgb_cloud(hdr, kept_xyz, np.full(len(kept_xyz), pack(C_KEPT), dtype=np.uint32))), stamp_ns)
            writer.write("/dbg/removed", serialize_message(
                rgb_cloud(hdr, rem_xyz, np.full(len(rem_xyz), pack(C_REMOVED), dtype=np.uint32))), stamp_ns)
            writer.write("/dbg/verdict", serialize_message(rgb_cloud(hdr, v_xyz, v_col)), stamp_ns)

            ma = MarkerArray()
            ma.markers.append(box_marker(hdr, "cube", 0,
                                         (-CUBE[0], CUBE[0]), (-CUBE[1], CUBE[1]),
                                         (-CUBE[2], CUBE[2]), (0, 220, 220)))
            ma.markers.append(box_marker(hdr, "pass", 1,
                                         LIM["x"], LIM["y"], LIM["z"], (255, 150, 0)))
            info = (f"#{idx}  pass&cube={len(Pw)}  del={int(del_mask.sum())}\n"
                    f"leak(黄) {n_leak}/{n_single}   false-del(品红) {n_fdel}/{n_multi}\n"
                    f"outside cube(灰蓝) {len(Po)}")
            ma.markers.append(text_marker(hdr, "info", 2, info,
                                          (0.0, 0.0, float(CUBE[2]) + 0.35), (255, 255, 255)))
            writer.write("/dbg/markers", serialize_message(ma), stamp_ns)

            tot["leak"] += n_leak
            tot["single"] += n_single
            tot["fdel"] += n_fdel
            tot["multi"] += n_multi
            tot["unjudged"] += len(Po)
            tot["frames"] += 1
            print(f"  [{fi + 1:02d}] #{idx:<5d} 回放t=+{fi * args.period:.1f}s "
                  f"pass&cube={len(Pw):5d} del={int(del_mask.sum()):5d} "
                  f"漏删={n_leak:4d}/{n_single:<5d} 误删={n_fdel:4d}/{n_multi:<5d} 框外={len(Po):4d}")
    finally:
        del writer

    print(f"\n写入 {tot['frames']} 帧 -> {args.out}")
    print(f"漏删 {tot['leak']}/{tot['single']}   误删 {tot['fdel']}/{tot['multi']}   "
          f"立方体外未判定 {tot['unjudged']}")
    print(f"\n回放:\n  ros2 bag play {args.out} -l -p"
          f"\n  (空格 播放/暂停, s 单步一条消息)")


if __name__ == "__main__":
    main()
