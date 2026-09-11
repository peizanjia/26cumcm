# -*- coding: utf-8 -*-
"""
问题 3 · 机器狗自动定位与清除策略（全向干扰源）

通过 HTTP+JSON 与官方模拟器通信（默认 http://127.0.0.1:2026）。
仅调用 4 条指令：/enter、/measure、/clear、/exit。

策略概述
--------
1. 检测（覆盖保证）：目标区为半径 1800 m 圆盘，干扰源有效接收半径下限 1000 m。
   取 7 个观测点（原点 + 半径 1732.05 m 的正六边形顶点），使圆盘内任一点都落在
   至少一个观测点的 1000 m 邻域内，从而保证每个全向干扰源至少被一个观测点检测到。
   在每个观测点对 1..20 全部频道测向，记录每个频道在各观测点的示向度。
2. 定位：对同一频道 ≥2 个示向度做最小二乘交会（线法向拟合），得到位置估计；
   仅 1 个示向度时记为"仅方位角"，后续用追踪法逼近。
3. 清除：按"最近邻贪心"顺序依次清除。有位置估计的频道先到估计点再精细逼近；
   仅方位角的频道从检测它的观测点做"二分追踪"逼近。逼近中一旦 /measure 返回
   near（≤5 m）或方向反转（越过目标），即收敛，然后 /clear（20 m 内必成功）。
   逼近失败时用 5×5 局部网格 /clear 兜底。
4. 验证：对检测到的频道在全部观测点复查，凡仍测到信号（未清除）再逼近清除一次，
   确保"所有干扰源被清除"。

所有请求逐次串行、每个新动作新 request_id；网络错误复用原 request_id 重试。
运行时把指令序列与响应完整写入 JSONL 日志，供复盘与支撑材料使用。
"""

import argparse
import json
import math
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ---------------- 常量 ----------------
BASE_URL = "http://127.0.0.1:2026"
ARENA_R = 1800.0          # 目标区域半径（米）
CLEAR_R = 20.0            # 清除半径
NEAR_R = 5.0              # “距离过近”阈值
SPEED = 5.0               # 移动速度 m/s
CHANNELS = list(range(1, 21))
MIN_EFFECTIVE_R = 1000.0  # 有效接收半径下限
RING_R = 1500.0           # 正六边形观测环半径（最远覆盖 ≈ 902 m < 1000，余量更足且路径更短）


# ---------------- 几何工具 ----------------
def angdiff(a, b):
    """两个角度(度)的带符号差，范围 (-180, 180]。"""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def midpoint(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def advance(pos, theta_deg, step):
    th = math.radians(theta_deg)
    return (pos[0] + step * math.cos(th), pos[1] + step * math.sin(th))


def make_vantages():
    """7 个观测点：原点 + 半径 RING_R 的正六边形顶点。"""
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60.0 * k)
        pts.append((RING_R * math.cos(a), RING_R * math.sin(a)))
    return pts


def triangulate(bearings):
    """
    最小二乘交会：bearings 为 [(x, y, theta_deg), ...]。
    每条示向度确定一条过 (x,y)、方向 theta 的直线，法向 n=(-sin,cos)，
    目标点 Q 应满足 n·Q = n·P。对多条直线做线性最小二乘。
    返回 (qx, qy)；若几何退化（近似平行）或解出区域明显外，返回 None。
    """
    A11 = A12 = A22 = 0.0
    b1 = b2 = 0.0
    for (x, y, t) in bearings:
        th = math.radians(t)
        nx, ny = -math.sin(th), math.cos(th)
        A11 += nx * nx
        A12 += nx * ny
        A22 += ny * ny
        c = nx * x + ny * y
        b1 += nx * c
        b2 += ny * c
    det = A11 * A22 - A12 * A12
    if abs(det) < 1e-6:
        return None
    qx = (A22 * b1 - A12 * b2) / det
    qy = (A11 * b2 - A12 * b1) / det
    if math.hypot(qx, qy) > ARENA_R + 800.0:
        return None
    return (qx, qy)


# ---------------- 机器人 ----------------
class Robot:
    def __init__(self, robot_id, base_url=BASE_URL, log_path=None):
        self.robot_id = robot_id
        self.base_url = base_url
        self.counter = 0
        self.log_lines = []
        self.cur_pos = (0.0, 0.0)
        self.cur_channel = 1
        self.log_path = log_path
        self.measure_count = 0
        self.clear_count = 0
        self.cleared_channels = []

    # ---- 基础通信 ----
    def _next_rid(self, tag):
        self.counter += 1
        return f"{tag}-{self.counter}"

    def _base(self, rid):
        return {"arena_id": "default", "robot_id": self.robot_id, "request_id": rid}

    def _post_once(self, path, payload):
        req = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_retry(self, path, payload, retries=5):
        """网络错误时复用同一 request_id 重试（幂等）。"""
        last = None
        for i in range(retries):
            try:
                return self._post_once(path, payload)
            except HTTPError as e:
                # 结构化错误（400 等）：读回体并返回，交由上层判断
                try:
                    body = json.loads(e.read().decode("utf-8"))
                except Exception:
                    body = None
                if body is not None:
                    return body
                raise
            except (URLError, OSError) as e:
                last = e
                time.sleep(0.25)
        raise RuntimeError(f"网络错误：{last}")

    def _log(self, path, payload, resp):
        self.log_lines.append({"path": path, "req": payload, "resp": resp})

    # ---- 四类指令 ----
    def enter(self, max_wait=900.0):
        """轮询进入，直到接口就绪并 accepted=true。"""
        payload = self._base("enter")
        start = time.time()
        while time.time() - start < max_wait:
            try:
                r = self._post_once("/enter", payload)
                if r.get("accepted") is True:
                    self._log("/enter", payload, r)
                    return r
                # accepted=false：接口刚开但被拒（如重复进入）；打印并稍候
                print(f"[enter] accepted=false: {r}", flush=True)
            except (URLError, OSError):
                pass  # 接口尚未开放（倒计时/非测试期），继续轮询
            time.sleep(1.0)
        raise RuntimeError("进入失败：接口长时间未就绪（请确认模拟器已开始测试且倒计时结束）")

    def measure(self, channel, pos):
        self.measure_count += 1
        rid = self._next_rid("measure")
        payload = self._base(rid)
        payload["position"] = {"x": pos[0], "y": pos[1]}
        payload["channel"] = channel
        r = self.post_retry("/measure", payload)
        self._log("/measure", payload, r)
        if r.get("accepted") is not True:
            print(f"[measure] accepted=false: {r}", flush=True)
            return None
        self.cur_pos = (pos[0], pos[1])
        self.cur_channel = channel
        return r

    def clear(self, channel, pos):
        self.clear_count += 1
        rid = self._next_rid("clear")
        payload = self._base(rid)
        payload["position"] = {"x": pos[0], "y": pos[1]}
        payload["channel"] = channel
        r = self.post_retry("/clear", payload)
        self._log("/clear", payload, r)
        if r.get("accepted") is not True:
            print(f"[clear] accepted=false: {r}", flush=True)
            return None
        self.cur_pos = (pos[0], pos[1])
        return r

    def exit(self):
        rid = self._next_rid("exit")
        payload = self._base(rid)
        r = self.post_retry("/exit", payload)
        self._log("/exit", payload, r)
        return r

    # ---- 检测 ----
    def detection_sweep(self):
        """
        遍历 7 个观测点，每个点测 1..20 频道。
        返回 detections: {channel: {"bearings": [(x,y,theta),...], "near": [(x,y),...]}}
        """
        vantages = make_vantages()
        detections = {}
        for vi, v in enumerate(vantages):
            print(f"[sweep] 观测点 {vi + 1}/{len(vantages)} @ ({v[0]:.1f},{v[1]:.1f})", flush=True)
            for c in CHANNELS:
                r = self.measure(c, v)
                if r is None:
                    continue
                mr = r.get("measure_result")
                if mr == "direction":
                    d = detections.setdefault(c, {"bearings": [], "near": []})
                    d["bearings"].append((v[0], v[1], r["svd_deg"]))
                elif mr == "near":
                    d = detections.setdefault(c, {"bearings": [], "near": []})
                    d["near"].append((v[0], v[1]))
        return detections

    # ---- 清除 ----
    def clear_channel(self, c, d, est):
        """清除单个频道，返回是否成功。"""
        if d["near"]:
            r = self.clear(c, d["near"][0])
            return r is not None and r.get("clear_result") == "success"
        if est is not None:
            return self.fine_clear(c, est)
        x, y, _t = d["bearings"][0]
        return self.pursue(c, (x, y))

    def pursue(self, c, start, step=400.0, max_iter=80):
        """从较远处按示向度二分追踪逼近目标，越过目标即转精细逼近。"""
        pos = start
        prev_pos = None
        prev_theta = None
        for _ in range(max_iter):
            r = self.measure(c, pos)
            if r is None:
                return False
            mr = r.get("measure_result")
            if mr == "near":
                cr = self.clear(c, pos)
                return cr is not None and cr.get("clear_result") == "success"
            if mr == "no_signal":
                return self.local_sweep(c, pos)
            theta = r["svd_deg"]
            if prev_theta is not None and abs(angdiff(theta, prev_theta)) > 120.0:
                # 方向反转 ≈ 越过了目标，目标位于两位置之间
                return self.fine_clear(c, midpoint(prev_pos, pos))
            prev_pos = pos
            prev_theta = theta
            pos = advance(pos, theta, step)
        cr = self.clear(c, pos)
        return cr is not None and cr.get("clear_result") == "success"

    def fine_clear(self, c, pos, step=12.0, max_iter=60):
        """从较近位置（估计点）做小步长示向度追踪并清除。"""
        prev_pos = None
        prev_theta = None
        for _ in range(max_iter):
            r = self.measure(c, pos)
            if r is None:
                return False
            mr = r.get("measure_result")
            if mr == "near":
                cr = self.clear(c, pos)
                return cr is not None and cr.get("clear_result") == "success"
            if mr == "no_signal":
                return self.local_sweep(c, pos)
            theta = r["svd_deg"]
            if prev_theta is not None and abs(angdiff(theta, prev_theta)) > 120.0:
                pos = midpoint(prev_pos, pos)
                prev_pos = None
                prev_theta = None
                continue
            prev_pos = pos
            prev_theta = theta
            pos = advance(pos, theta, step)
        cr = self.clear(c, pos)
        return cr is not None and cr.get("clear_result") == "success"

    def local_sweep(self, c, center, spacing=20.0, half=2):
        """在 center 周围 5×5 网格尝试 /clear（覆盖 ±40 m）。"""
        x0, y0 = center
        pts = [(x0 + i * spacing, y0 + j * spacing)
               for i in range(-half, half + 1) for j in range(-half, half + 1)]
        pts.sort(key=lambda p: math.hypot(p[0] - x0, p[1] - y0))
        for p in pts:
            r = self.clear(c, p)
            if r is not None and r.get("clear_result") == "success":
                return True
        return False

    # ---- 主流程 ----
    def run(self):
        enter = self.enter()
        print(f"[enter] 剩余现实时间 {enter.get('remaining_real_duration_s')} s", flush=True)

        detections = self.detection_sweep()
        active = sorted(detections.keys())
        print(f"[detect] 检测到频道: {active}（共 {len(active)} 个）", flush=True)

        # 定位
        estimates = {}
        for c in active:
            d = detections[c]
            if len(d["bearings"]) >= 2:
                estimates[c] = triangulate(d["bearings"])
            else:
                estimates[c] = None
        have_est = sum(1 for c in active if estimates.get(c) is not None)
        print(f"[localize] 有交会位置估计的频道: {have_est}/{len(active)}", flush=True)

        # 清除（最近邻贪心）
        cleared = set()
        failed = []
        remaining = set(active)
        cur = self.cur_pos

        def target_pos(c):
            e = estimates.get(c)
            if e is not None:
                return e
            d = detections[c]
            if d["near"]:
                return d["near"][0]
            if d["bearings"]:
                x, y, t = d["bearings"][0]
                th = math.radians(t)
                return (x + 900.0 * math.cos(th), y + 900.0 * math.sin(th))
            return cur

        while remaining:
            c = min(remaining, key=lambda cc: math.hypot(
                target_pos(cc)[0] - cur[0], target_pos(cc)[1] - cur[1]))
            remaining.discard(c)
            ok = self.clear_channel(c, detections[c], estimates[c])
            if ok:
                cleared.add(c)
                self.cleared_channels.append(c)
                cur = self.cur_pos
                print(f"[clear] 频道 {c} 已清除 ({len(cleared)}/{len(active)})", flush=True)
            else:
                failed.append(c)
                print(f"[clear] 频道 {c} 清除失败，稍后重试", flush=True)

        # 重试失败频道：回到其检测观测点重新追踪
        for c in failed:
            d = detections[c]
            if d["bearings"]:
                x, y, _t = d["bearings"][0]
                ok = self.pursue(c, (x, y))
            elif d["near"]:
                ok = self.clear_channel(c, d, None)
            else:
                ok = False
            if ok and c not in cleared:
                cleared.add(c)
                self.cleared_channels.append(c)
                print(f"[retry] 频道 {c} 重试清除成功", flush=True)

        # 验证：复查全部观测点上的已检测频道，凡仍测到信号即再逼近清除
        self.verify(active, cleared, detections)

        exit_r = self.exit()
        vt = exit_r.get("virtual_time_s", 0) if exit_r else 0
        print(f"[exit] {exit_r}", flush=True)
        return {
            "active": active,
            "cleared": sorted(cleared),
            "cleared_count": len(cleared),
            "virtual_time_s": vt,
            "measure_count": self.measure_count,
            "clear_count": self.clear_count,
        }

    def verify(self, active, cleared, detections):
        """复查：对尚未清除的已检测频道，在全部观测点重新测向并再逼近清除。"""
        pending = set(active) - cleared
        if not pending:
            print("[verify] 无待复查频道，跳过复查", flush=True)
            return
        print(f"[verify] 复查频道: {sorted(pending)}", flush=True)
        vantages = make_vantages()
        for v in vantages:
            for c in sorted(pending):
                if c in self.cleared_channels:
                    continue
                r = self.measure(c, v)
                if r is None:
                    continue
                mr = r.get("measure_result")
                if mr == "near":
                    ok = self.clear(c, v)
                    if ok is not None and ok.get("clear_result") == "success":
                        cleared.add(c)
                        self.cleared_channels.append(c)
                        print(f"[verify] 频道 {c} 复查清除成功", flush=True)
                        continue
                elif mr == "direction":
                    ok = self.pursue(c, (v[0], v[1]))
                    if ok:
                        cleared.add(c)
                        self.cleared_channels.append(c)
                        print(f"[verify] 频道 {c} 复查清除成功", flush=True)

    # ---- 日志 ----
    def dump_log(self):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[log] 已写入 {self.log_path}（{len(self.log_lines)} 条）", flush=True)


def main():
    ap = argparse.ArgumentParser(description="问题3 机器狗策略")
    ap.add_argument("--robot-id", required=True, help="参赛队号（robot_id）")
    ap.add_argument("--base-url", default=BASE_URL, help="模拟器接口地址")
    ap.add_argument("--log-dir", default=None, help="日志目录")
    ap.add_argument("--runs", type=int, default=1, help="连续演练次数（每次结束后等待下一次开始）")
    args = ap.parse_args()

    # 控制台改 UTF-8，避免 Windows GBK 乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    log_dir = args.log_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "summary.csv")

    import csv as _csv
    fieldnames = ["run", "active", "cleared", "ratio", "virtual_time_s",
                  "avg_time_s", "measure_count", "clear_count", "log"]
    first_write = not os.path.exists(summary_path)

    all_stats = []
    for i in range(1, args.runs + 1):
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"run_{ts}.jsonl")
        robot = Robot(args.robot_id, base_url=args.base_url, log_path=log_path)
        print(f"\n########## 第 {i}/{args.runs} 次演练 ##########", flush=True)
        try:
            summary = robot.run()
            n = summary.get("cleared_count", len(summary["cleared"]))
            act = len(summary["active"])
            avg = (summary["virtual_time_s"] / n) if n else 0.0
            print("===== 本次小结 =====", flush=True)
            print(f"检测频道数: {act}  清除频道数: {n}", flush=True)
            print(f"虚拟总时间: {summary['virtual_time_s']:.3f} s  平均定位清除时间: {avg:.3f} s", flush=True)
            print(f"measure 次数: {summary['measure_count']}  clear 次数: {summary['clear_count']}", flush=True)
            row = {
                "run": i,
                "active": act,
                "cleared": n,
                "ratio": (n / act) if act else 0.0,
                "virtual_time_s": round(summary["virtual_time_s"], 3),
                "avg_time_s": round(avg, 3),
                "measure_count": summary["measure_count"],
                "clear_count": summary["clear_count"],
                "log": os.path.basename(log_path),
            }
            all_stats.append(row)
            with open(summary_path, "a", encoding="utf-8", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                if first_write:
                    w.writeheader()
                    first_write = False
                w.writerow(row)
        finally:
            robot.dump_log()

    if len(all_stats) > 1:
        tot_act = sum(r["active"] for r in all_stats)
        tot_clear = sum(r["cleared"] for r in all_stats)
        avg_all = sum(r["avg_time_s"] for r in all_stats) / len(all_stats)
        print("\n===== 演练汇总 =====", flush=True)
        print(f"共 {len(all_stats)} 次演练；清除 {tot_clear}/{tot_act}；"
              f"平均定位清除时间均值 {avg_all:.3f} s", flush=True)
        print(f"统计已追加到 {summary_path}", flush=True)


if __name__ == "__main__":
    main()
