"""
🛰️ AI OBJECT INTELLIGENCE SYSTEM
================================
Pipeline:  Detection → Classification → Tracking → Behavior Analysis

Single-file Streamlit app. No API keys. Runs a realistic synthetic sensor
feed (roads, intersection, traffic lights, pedestrians, vehicles, animals),
pre-warms 15 seconds of history so the dashboard is ALIVE on launch.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Object Intelligence System", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    .block-container{padding-top:1.1rem;}
    div[data-testid="stToolbar"]{display:none;}
    footer{visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════
# 1. CONSTANTS & HELPERS
# ════════════════════════════════════════════════════════════════
SEED = 7                      # curated demo scene (same great first impression)
W, H = 960, 540               # frame size (px)
PPM = 10.0                    # pixels per meter  → world = 96 m × 54 m
WORLD_W, WORLD_H = W / PPM, H / PPM
ROAD_HALF, SIDE = 6.0, 2.5
CX, CY = WORLD_W / 2, WORLD_H / 2
H_ROAD = (CY - ROAD_HALF, CY + ROAD_HALF)     # horizontal road band (y)
V_ROAD = (CX - ROAD_HALF, CX + ROAD_HALF)     # vertical road band (x)
CROSS_HALF = 3.5
SPEED_LIMIT = 16.0            # m/s
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASS_META = {
    "person":  dict(emoji="🚶", label="Person",  rgb=(79, 195, 247),  size=(0.9, 0.9), conf=0.90),
    "car":     dict(emoji="🚗", label="Car",     rgb=(255, 183, 77),  size=(4.4, 2.2), conf=0.93),
    "bicycle": dict(emoji="🚲", label="Bicycle", rgb=(129, 199, 132), size=(1.9, 0.9), conf=0.87),
    "dog":     dict(emoji="🐕", label="Dog",     rgb=(240, 98, 146),  size=(1.1, 0.6), conf=0.85),
    "bird":    dict(emoji="🕊️", label="Bird",    rgb=(186, 104, 200), size=(0.7, 0.5), conf=0.82),
}
KIND_W = [("person", 0.34), ("car", 0.28), ("bicycle", 0.13), ("dog", 0.12), ("bird", 0.13)]

BEHAVIOR_COLOR = {
    "walking": "#4CAF50", "running": "#FF9800", "idle": "#9E9E9E",
    "loitering": "#F44336", "crossing": "#00BCD4", "jaywalking": "#E91E63",
    "moving": "#66BB6A", "stopped": "#FFC107", "parked": "#78909C",
    "turning": "#2196F3", "speeding": "#F44336",
    "cruising": "#26A69A", "sniffing": "#AB47BC", "flying": "#29B6F6",
}

PARK = (3.5, 3.5, V_ROAD[0] - SIDE - 1.5, H_ROAD[0] - SIDE - 1.5)

PED_ZONES = [
    (2, 2, V_ROAD[0] - SIDE - 0.5, H_ROAD[0] - SIDE - 0.5),
    (V_ROAD[1] + SIDE + 0.5, 2, WORLD_W - 2, H_ROAD[0] - SIDE - 0.5),
    (2, H_ROAD[1] + SIDE + 0.5, V_ROAD[0] - SIDE - 0.5, WORLD_H - 2),
    (V_ROAD[1] + SIDE + 0.5, H_ROAD[1] + SIDE + 0.5, WORLD_W - 2, WORLD_H - 2),
    (2, H_ROAD[0] - SIDE - 0.3, V_ROAD[0] - SIDE - 0.5, H_ROAD[0] - 0.5),
    (V_ROAD[1] + SIDE + 0.5, H_ROAD[0] - SIDE - 0.3, WORLD_W - 2, H_ROAD[0] - 0.5),
    (2, H_ROAD[1] + 0.5, V_ROAD[0] - SIDE - 0.5, H_ROAD[1] + SIDE + 0.3),
    (V_ROAD[1] + SIDE + 0.5, H_ROAD[1] + 0.5, WORLD_W - 2, H_ROAD[1] + SIDE + 0.3),
]


def P(m):
    return int(round(m * PPM))


def clamp(v, a, b):
    return max(a, min(b, v))


def ang_diff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def ang_lerp(a, b, k):
    return a + ang_diff(a, b) * k


def bearing_of(vx, vy):
    if math.hypot(vx, vy) < 0.15:
        return 0.0
    return (math.degrees(math.atan2(vx, -vy)) + 360.0) % 360.0


def compass_str(vx, vy):
    if math.hypot(vx, vy) < 0.15:
        return "static"
    deg = bearing_of(vx, vy)
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return f"{names[int((deg + 22.5) // 45) % 8]} {deg:.0f}°"


def fmt_t(t):
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(c):
    return "#%02x%02x%02x" % tuple(c)


def rand_ped_point(rng):
    z = rng.choice(PED_ZONES)
    return rng.uniform(z[0], z[2]), rng.uniform(z[1], z[3])


def light_state(t):
    c = t % 24.0
    if c < 11:   return {"h": "green", "v": "red"}
    if c < 12:   return {"h": "yellow", "v": "red"}
    if c < 23:   return {"h": "red", "v": "green"}
    return {"h": "red", "v": "yellow"}


def _lane(axis, d):
    if axis == "h":
        return CY + 3.0 if d > 0 else CY - 3.0
    return CX + 3.0 if d > 0 else CX - 3.0


def _stop(axis, d):
    if axis == "h":
        return V_ROAD[0] - 1.5 if d > 0 else V_ROAD[1] + 1.5
    return H_ROAD[0] - 1.5 if d > 0 else H_ROAD[1] + 1.5


# ════════════════════════════════════════════════════════════════
# 2. GROUND-TRUTH WORLD SIMULATION (synthetic sensor feed)
# ════════════════════════════════════════════════════════════════
@dataclass
class Entity:
    eid: int
    kind: str
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    s: dict = field(default_factory=dict)
    dead: bool = False


class World:
    def __init__(self, rng):
        self.rng = rng
        self.t = 0.0
        self.entities: list[Entity] = []
        self.next_eid = 1

    # ---------- spawning ----------
    def add(self, kind, **kw):
        e = Entity(self.next_eid, kind)
        self.next_eid += 1
        self.entities.append(e)
        rng = self.rng
        if kind == "car":
            if kw.get("parked"):
                e.x, e.y = rng.uniform(8, 36), H_ROAD[1] - 1.0
                e.heading = 0.0
                e.s = dict(axis="h", dir=1, lane=e.y, cruise=0.0,
                           stop=_stop("h", 1), decided=True)
            else:
                axis = kw.get("axis") or rng.choice(["h", "v"])
                d = kw.get("dir") or rng.choice([-1, 1])
                lane = _lane(axis, d)
                if axis == "h":
                    e.x = -4.0 if d > 0 else WORLD_W + 4.0
                    e.y = lane + rng.uniform(-0.3, 0.3)
                    e.heading = 0.0 if d > 0 else math.pi
                else:
                    e.y = -4.0 if d > 0 else WORLD_H + 4.0
                    e.x = lane + rng.uniform(-0.3, 0.3)
                    e.heading = math.pi / 2 if d > 0 else -math.pi / 2
                cruise = rng.uniform(16.5, 19.0) if rng.random() < 0.10 else rng.uniform(6.5, 12.0)
                e.s = dict(axis=axis, dir=d, lane=lane, cruise=kw.get("cruise", cruise),
                           stop=_stop(axis, d), decided=False)
        elif kind == "person":
            e.x, e.y = kw.get("pos") or rand_ped_point(rng)
            runner = kw.get("runner", rng.random() < 0.15)
            spd = rng.uniform(2.7, 3.6) if runner else rng.uniform(1.0, 1.7)
            e.s = dict(wp=None, pause=self.t + rng.uniform(0, 1.5), mode="roam", spd=spd,
                       loiterer=kw.get("loiterer", rng.random() < 0.12),
                       loiter_until=0.0, lc=None, cross_wps=None)
            e.heading = rng.uniform(-math.pi, math.pi)
            if kw.get("crossing"):
                self._start_cross(e, jay=False)
        elif kind == "bicycle":
            d = rng.choice([-1, 1])
            e.x = rng.uniform(10, WORLD_W - 10)
            e.s = dict(dir=d, y=(H_ROAD[0] + 0.9) if d > 0 else (H_ROAD[1] - 0.9),
                       spd=rng.uniform(3.2, 5.4), pause=0.0)
            e.y = e.s["y"]
            e.heading = 0.0 if d > 0 else math.pi
        elif kind == "dog":
            e.x = rng.uniform(PARK[0], PARK[2])
            e.y = rng.uniform(PARK[1], PARK[3])
            e.s = dict(wp=None, pause=self.t + rng.uniform(0, 1.0), spd=rng.uniform(1.6, 3.2))
        elif kind == "bird":
            side = rng.choice([-1, 1])
            e.x = -4.0 if side > 0 else WORLD_W + 4.0
            e.y = rng.uniform(3, 13)
            e.vx = side * rng.uniform(5, 9)
            e.s = dict(ph=rng.uniform(0, 6.28))
            e.heading = 0.0 if side > 0 else math.pi
        return e

    # ---------- pedestrian decisions ----------
    def _start_cross(self, e, jay=False):
        rng = self.rng
        s = e.s
        if rng.random() < 0.72 or jay:
            if jay:
                a, b = rng.choice([(6, CX - 9), (CX + 9, WORLD_W - 6)])
                x0 = rng.uniform(a, b)
            else:
                x0 = CX + rng.uniform(-2.5, 2.5)
            if e.y < CY:
                entry, exitp = (x0, H_ROAD[0] - 1.0), (x0, H_ROAD[1] + 1.0)
            else:
                entry, exitp = (x0, H_ROAD[1] + 1.0), (x0, H_ROAD[0] - 1.0)
        else:
            y0 = CY + rng.uniform(-2.5, 2.5)
            if e.x < CX:
                entry, exitp = (V_ROAD[0] - 1.0, y0), (V_ROAD[1] + 1.0, y0)
            else:
                entry, exitp = (V_ROAD[1] + 1.0, y0), (V_ROAD[0] - 1.0, y0)
        s["mode"] = "cross"
        s["wp"] = entry
        s["cross_wps"] = [exitp]

    def _ped_decide(self, e):
        rng = self.rng
        s = e.s
        t = self.t
        if s["mode"] == "loiter":
            if t < s["loiter_until"]:
                cx_, cy_ = s["lc"]
                s["wp"] = (clamp(cx_ + rng.uniform(-2.5, 2.5), 2, WORLD_W - 2),
                           clamp(cy_ + rng.uniform(-2.5, 2.5), 2, WORLD_H - 2))
                return
            s["mode"] = "roam"
        if s["mode"] == "cross":
            s["mode"] = "roam"
        r = rng.random()
        if s["loiterer"] and r < 0.55:
            s["mode"] = "loiter"
            s["loiter_until"] = t + rng.uniform(12, 22)
            s["lc"] = (e.x + rng.uniform(-2, 2), e.y + rng.uniform(-2, 2))
            s["wp"] = s["lc"]
            return
        if r < 0.60:
            self._start_cross(e, jay=(rng.random() < 0.22))
            return
        if r < 0.70:
            s["pause"] = t + rng.uniform(1.5, 4.0)
            return
        s["wp"] = rand_ped_point(rng)

    # ---------- per-kind updates ----------
    def _plan_turn(self, e):
        rng = self.rng
        s = e.s
        new_axis = "v" if s["axis"] == "h" else "h"
        new_dir = rng.choice([-1, 1])
        new_lane = _lane(new_axis, new_dir)
        Pc = (CX, CY)
        if new_axis == "v":
            P1 = (new_lane, CY + new_dir * 9)
        else:
            P1 = (CX + new_dir * 9, new_lane)
        P0 = (e.x, e.y)
        length = math.dist(P0, Pc) + math.dist(Pc, P1)
        v = max(2.2, 0.45 * s["cruise"])
        s["turn"] = dict(P0=P0, Pc=Pc, P1=P1, T=length / v, v=v, t=0.0,
                         axis=new_axis, dir=new_dir, lane=new_lane)

    def _upd_car(self, e, dt):
        s = e.s
        rng = self.rng
        if s["cruise"] <= 0:                      # parked
            e.vx = e.vy = 0.0
            return
        if s.get("turn"):                          # bezier turn in progress
            trn = s["turn"]
            trn["t"] += dt
            p = min(1.0, trn["t"] / trn["T"])
            q = 1.0 - p
            P0, Pc, P1 = trn["P0"], trn["Pc"], trn["P1"]
            e.x = q * q * P0[0] + 2 * q * p * Pc[0] + p * p * P1[0]
            e.y = q * q * P0[1] + 2 * q * p * Pc[1] + p * p * P1[1]
            dx = 2 * q * (Pc[0] - P0[0]) + 2 * p * (P1[0] - Pc[0])
            dy = 2 * q * (Pc[1] - P0[1]) + 2 * p * (P1[1] - Pc[1])
            n = math.hypot(dx, dy) or 1.0
            e.vx, e.vy = dx / n * trn["v"], dy / n * trn["v"]
            e.heading = math.atan2(e.vy, e.vx)
            if p >= 1.0:
                s["axis"], s["dir"], s["lane"] = trn["axis"], trn["dir"], trn["lane"]
                s["stop"] = _stop(s["axis"], s["dir"])
                del s["turn"]
                s["decided"] = True
            return

        axis, d, cruise = s["axis"], s["dir"], s["cruise"]
        along = e.x if axis == "h" else e.y
        center = CX if axis == "h" else CY
        dist_c = (center - along) * d
        if dist_c > 22:
            s["decided"] = False
        if not s.get("decided") and 4 < dist_c < 14:
            s["decided"] = True
            if light_state(self.t)[axis] == "green" and rng.random() < 0.30:
                self._plan_turn(e)
                return

        desired = cruise
        if light_state(self.t)[axis] in ("red", "yellow"):
            dist_s = (s["stop"] - along) * d
            if dist_s > 0:
                va = (e.vx if axis == "h" else e.vy) * d
                if dist_s < va * va / 8.0 + 0.8:
                    desired = 0.0
        for o in self.entities:                    # car-following gap
            if o is e or o.kind != "car" or o.s.get("turn"):
                continue
            if o.s.get("axis") == axis and o.s.get("dir") == d and abs(o.s.get("lane", 0) - s["lane"]) < 1:
                gap = ((o.x if axis == "h" else o.y) - along) * d
                if 0 < gap < 6:
                    desired = 0.0 if gap < 2.5 else min(desired, (gap - 2.5) / 3.5 * cruise)

        va = (e.vx if axis == "h" else e.vy) * d
        va = min(desired, va + 3.2 * dt) if va < desired else max(desired, va - 5.5 * dt)
        lat = e.y if axis == "h" else e.x
        vlat = clamp((s["lane"] - lat) * 2.0, -1.4, 1.4)
        along += va * dt
        lat += vlat * dt
        if axis == "h":
            e.x, e.y, e.vx, e.vy = along, lat, va * d, vlat
        else:
            e.x, e.y, e.vx, e.vy = lat, along, vlat, va * d
        if abs(va) > 0.3 or abs(vlat) > 0.3:
            e.heading = math.atan2(e.vy, e.vx)

        m = 6.0                                    # toroidal wrap
        if axis == "h":
            if e.x < -m:            e.x += WORLD_W + 2 * m
            elif e.x > WORLD_W + m: e.x -= WORLD_W + 2 * m
        else:
            if e.y < -m:            e.y += WORLD_H + 2 * m
            elif e.y > WORLD_H + m: e.y -= WORLD_H + 2 * m

    def _upd_person(self, e, dt):
        s = e.s
        t = self.t
        if s["wp"] is None:
            if t < s["pause"]:
                k = max(0.0, 1 - dt * 4)
                e.vx *= k
                e.vy *= k
                return
            self._ped_decide(e)
            if s["wp"] is None:
                return
        wx, wy = s["wp"]
        dx, dy = wx - e.x, wy - e.y
        d = math.hypot(dx, dy)
        if d < 0.55:
            if s["mode"] == "cross" and s["cross_wps"]:
                s["wp"] = s["cross_wps"].pop(0)
            else:
                s["wp"] = None
                if s["mode"] == "cross":
                    s["mode"] = "roam"
                if self.rng.random() < 0.30:
                    s["pause"] = t + self.rng.uniform(0.6, 2.5)
            return
        spd = s["spd"] * (0.55 if s["mode"] == "loiter" else 1.0)
        tvx, tvy = dx / d * spd, dy / d * spd
        k = min(1.0, dt * 3.0)
        e.vx += (tvx - e.vx) * k
        e.vy += (tvy - e.vy) * k
        e.x += e.vx * dt
        e.y += e.vy * dt
        if math.hypot(e.vx, e.vy) > 0.25:
            e.heading = math.atan2(e.vy, e.vx)

    def _upd_bike(self, e, dt):
        s = e.s
        if self.t < s["pause"]:
            k = max(0.0, 1 - dt * 4)
            e.vx *= k
            e.vy *= k
            return
        e.vx += (s["dir"] * s["spd"] - e.vx) * min(1.0, dt * 2.5)
        e.x += e.vx * dt
        e.y = s["y"]
        if e.x < 4 and s["dir"] < 0:
            s["dir"] = 1
            if self.rng.random() < 0.3:
                s["pause"] = self.t + self.rng.uniform(1, 3)
        elif e.x > WORLD_W - 4 and s["dir"] > 0:
            s["dir"] = -1
            if self.rng.random() < 0.3:
                s["pause"] = self.t + self.rng.uniform(1, 3)
        e.heading = 0.0 if s["dir"] > 0 else math.pi

    def _upd_dog(self, e, dt):
        s = e.s
        if s["wp"] is None:
            if self.t < s["pause"]:
                k = max(0.0, 1 - dt * 4)
                e.vx *= k
                e.vy *= k
                return
            s["wp"] = (self.rng.uniform(PARK[0], PARK[2]), self.rng.uniform(PARK[1], PARK[3]))
        wx, wy = s["wp"]
        dx, dy = wx - e.x, wy - e.y
        d = math.hypot(dx, dy)
        if d < 0.6:
            s["wp"] = None
            if self.rng.random() < 0.35:
                s["pause"] = self.t + self.rng.uniform(1.0, 2.5)
            return
        tvx, tvy = dx / d * s["spd"], dy / d * s["spd"]
        k = min(1.0, dt * 3.0)
        e.vx += (tvx - e.vx) * k
        e.vy += (tvy - e.vy) * k
        e.x += e.vx * dt
        e.y += e.vy * dt
        if math.hypot(e.vx, e.vy) > 0.25:
            e.heading = math.atan2(e.vy, e.vx)

    def _upd_bird(self, e, dt):
        e.x += e.vx * dt
        e.y += 0.7 * math.sin(self.t * 1.6 + e.s["ph"]) * dt
        e.heading = 0.0 if e.vx > 0 else math.pi
        if e.x < -8 or e.x > WORLD_W + 8:
            e.dead = True

    def step(self, dt):
        self.t += dt
        for e in self.entities:
            {"car": self._upd_car, "person": self._upd_person, "bicycle": self._upd_bike,
             "dog": self._upd_dog, "bird": self._upd_bird}[e.kind](e, dt)
        self.entities = [e for e in self.entities if not e.dead]


# ════════════════════════════════════════════════════════════════
# 3. DETECTION + CLASSIFICATION LAYER (simulated neural detector)
# ════════════════════════════════════════════════════════════════
@dataclass
class Det:
    cls: str
    conf: float
    x: float
    y: float
    w: float
    h: float


def make_detections(world, noise, rng):
    dets = []
    for e in world.entities:
        if rng.random() < 0.02 + 0.05 * noise:            # missed detection
            continue
        cls = e.kind
        if rng.random() < 0.015:                          # class confusion
            cls = rng.choice([k for k in CLASS_META if k != cls])
        meta = CLASS_META[e.kind]
        conf = clamp(meta["conf"] + rng.gauss(0, 0.035 + 0.05 * noise), 0.42, 0.99)
        sig = 0.10 + 0.45 * noise
        L, Wd = meta["size"]
        c, sn = abs(math.cos(e.heading)), abs(math.sin(e.heading))
        bw = (c * L + sn * Wd) * 1.15 * (1 + rng.gauss(0, 0.04))
        bh = (sn * L + c * Wd) * 1.15 * (1 + rng.gauss(0, 0.04))
        dets.append(Det(cls, conf, e.x + rng.gauss(0, sig), e.y + rng.gauss(0, sig), bw, bh))
    return dets


# ════════════════════════════════════════════════════════════════
# 4. TRACKING LAYER (predict → gate → greedy associate → smooth)
# ════════════════════════════════════════════════════════════════
class Track:
    __slots__ = ("id", "cls", "cls_votes", "x", "y", "vx", "vy", "conf", "hits", "miss",
                 "first_t", "last_t", "heading", "speed", "turn_rate", "behavior",
                 "pending", "pend_n", "segments", "pos_hist", "speed_hist",
                 "still_since", "rad_ok_since", "alert_cd")

    def __init__(self, tid, t, det):
        self.id = tid
        self.cls = det.cls
        self.cls_votes = {det.cls: 1}
        self.x, self.y = det.x, det.y
        self.vx = self.vy = 0.0
        self.conf = det.conf
        self.hits, self.miss = 1, 0
        self.first_t = self.last_t = t
        self.heading = 0.0
        self.speed = 0.0
        self.turn_rate = 0.0
        self.behavior = "—"
        self.pending, self.pend_n = None, 0
        self.segments = []
        self.pos_hist = deque(maxlen=110)
        self.speed_hist = deque(maxlen=100)
        self.still_since = t
        self.rad_ok_since = None
        self.alert_cd = {}


class Tracker:
    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, dets, t, dt):
        events = []
        cand = []
        for i, d in enumerate(dets):
            for tr in self.tracks.values():
                if tr.miss > 10:
                    continue
                px_, py_ = tr.x + tr.vx * dt, tr.y + tr.vy * dt
                dist = math.hypot(d.x - px_, d.y - py_)
                gate = 2.6 + 0.6 * tr.speed + max(d.w, d.h) * 0.5
                if dist < gate:
                    cand.append((dist, i, tr))
        cand.sort(key=lambda z: z[0])
        used_d, used_t = set(), set()
        for dist, i, tr in cand:
            if i in used_d or tr.id in used_t:
                continue
            used_d.add(i)
            used_t.add(tr.id)
            self._merge(tr, dets[i], t)
        for i, d in enumerate(dets):
            if i not in used_d:
                self.tracks[self.next_id] = Track(self.next_id, t, d)
                events.append(("new", self.next_id, d.cls))
                self.next_id += 1
        for tr in list(self.tracks.values()):
            if tr.id in used_t:
                continue
            tr.miss += 1
            tr.speed *= 0.9
            if tr.miss > (16 if tr.hits >= 3 else 5):
                del self.tracks[tr.id]
                if tr.hits >= 3:
                    events.append(("lost", -1, tr))
        return events

    def _merge(self, tr, d, t):
        dt_ = max(1e-3, t - tr.last_t)
        predx, predy = tr.x + tr.vx * dt_, tr.y + tr.vy * dt_
        nx, ny = predx * 0.35 + d.x * 0.65, predy * 0.35 + d.y * 0.65
        ivx, ivy = (nx - tr.x) / dt_, (ny - tr.y) / dt_
        tr.vx += (ivx - tr.vx) * 0.35
        tr.vy += (ivy - tr.vy) * 0.35
        tr.x, tr.y = nx, ny
        tr.conf += (d.conf - tr.conf) * 0.3
        tr.cls_votes[d.cls] = tr.cls_votes.get(d.cls, 0) + 1
        tr.cls = max(tr.cls_votes, key=tr.cls_votes.get)   # classification refinement
        tr.hits += 1
        tr.miss = 0
        tr.last_t = t
        tr.speed = math.hypot(tr.vx, tr.vy)
        if tr.speed > 0.3:
            th = math.atan2(tr.vy, tr.vx)
            inst = ang_diff(tr.heading, th) / dt_
            tr.turn_rate += (clamp(inst, -3, 3) - tr.turn_rate) * 0.25
            tr.heading = ang_lerp(tr.heading, th, min(1.0, dt_ * 6))
        else:
            tr.turn_rate *= 0.8
        tr.pos_hist.append((t, tr.x, tr.y))
        tr.speed_hist.append(tr.speed)


# ════════════════════════════════════════════════════════════════
# 5. ENGINE — orchestrates world → detections → tracker → behavior
# ════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.world = World(self.rng)
        self.tracker = Tracker()
        self.t = 0.0
        self.frame = 0
        self.total_det = 0
        self.total_cls = 0
        self.log = deque(maxlen=40000)
        self.alerts = deque(maxlen=80)
        self.speed_trend = deque(maxlen=360)
        self.new_times = deque(maxlen=200)
        self.archived = deque(maxlen=6)
        self.last_counts = {"det": 0, "cls": 0, "trk": 0, "beh": 0}
        self.last_dets: list[Det] = []
        self.pipe_ms = 0.0
        self.fps = 0.0
        self._wall = None
        self._seed_scene()

    # ---------- curated demo scene + pre-warm ----------
    def _seed_scene(self):
        w = self.world
        w.add("car", parked=True)
        w.add("person", loiterer=True)
        w.add("person", runner=True)
        w.add("person", crossing=True)
        w.add("person"); w.add("person")
        w.add("car"); w.add("car"); w.add("car")
        w.add("bicycle"); w.add("dog"); w.add("bird"); w.add("bird")
        for _ in range(150):                     # 15 s of hidden history
            self.step(0.1, 0.25, 14)
        while len(self.alerts) > 6:
            self.alerts.popleft()

    # ---------- population control ----------
    def _manage_pop(self, target):
        alive = len(self.world.entities)
        if alive < target:
            for _ in range(min(2, target - alive)):
                kind = self.rng.choices([k for k, _ in KIND_W], weights=[w for _, w in KIND_W])[0]
                self.world.add(kind)
        elif alive > target + 4 and self.rng.random() < 0.15:
            cands = [e for e in self.world.entities if e.kind in ("bird", "car")]
            if cands:
                self.rng.choice(cands).dead = True

    def _push_alert(self, sev, typ, track, msg):
        self.alerts.append({"t": self.t, "sev": sev, "type": typ, "track": track, "msg": msg})

    # ---------- one simulation step ----------
    def step(self, sim_dt, noise, target):
        t0 = time.perf_counter()
        self.t += sim_dt
        self.frame += 1
        self.world.step(sim_dt)
        self._manage_pop(target)

        raw = make_detections(self.world, noise, self.rng)                    # DETECTION
        cls = [d for d in raw if d.conf >= 0.50]                              # CLASSIFICATION
        self.last_dets = cls
        events = self.tracker.update(cls, self.t, sim_dt)                     # TRACKING
        for kind, tid, payload in events:
            if kind == "new":
                self.new_times.append(self.t)
                self._push_alert("ℹ️", "acquired", f"#{tid:02d}", f"Track #{tid:02d} acquired ({payload})")
            else:
                tr = payload
                self.archived.append({"id": tr.id, "cls": tr.cls, "segs": list(tr.segments), "end": self.t})
                self._push_alert("ℹ️", "lost", f"#{tr.id:02d}", f"Track #{tr.id:02d} lost ({tr.cls})")

        confirmed = 0
        for tr in self.tracker.tracks.values():
            self.analyze(tr)                                                  # BEHAVIOR
            if tr.hits >= 3:
                confirmed += 1
                self.log.append((round(self.t, 2), tr.id, tr.cls, round(tr.x, 2), round(tr.y, 2),
                                 round(tr.speed, 2), round(bearing_of(tr.vx, tr.vy), 1),
                                 tr.behavior, round(tr.conf, 3)))

        self.last_counts = {"det": len(raw), "cls": len(cls), "trk": confirmed, "beh": confirmed}
        self.total_det += len(raw)
        self.total_cls += len(cls)
        sp = [tr.speed for tr in self.tracker.tracks.values() if tr.hits >= 3 and tr.speed > 0.4]
        self.speed_trend.append((self.t, sum(sp) / len(sp) if sp else 0.0))
        self.pipe_ms = self.pipe_ms * 0.7 + (time.perf_counter() - t0) * 1000 * 0.3

    # ---------- live entry point (real-time paced) ----------
    def tick(self, speed, noise, target):
        now = time.perf_counter()
        if self._wall is None:
            self._wall = now
        real = min(0.12, max(1e-4, now - self._wall))
        self._wall = now
        inst = 1.0 / real
        self.fps += (inst - self.fps) * 0.1
        self.step(real * speed, noise, target)

    # ---------- behavior state machine ----------
    def _commit(self, tr, state):
        t = self.t
        if tr.segments:
            tr.segments[-1]["t1"] = t
        tr.segments.append({"t0": t, "t1": t, "state": state})
        if len(tr.segments) > 80:
            tr.segments.pop(0)
        tr.behavior = state
        if state == "loitering":
            self._alert_track(tr, "loitering", "🚨", f"Person #{tr.id:02d} loitering near ({tr.x:.0f},{tr.y:.0f}) m")
        elif state == "speeding":
            self._alert_track(tr, "speeding", "🚨", f"Car #{tr.id:02d} at {tr.speed:.1f} m/s (limit {SPEED_LIMIT:.0f})")
        elif state == "jaywalking":
            self._alert_track(tr, "jaywalking", "⚠️", f"Person #{tr.id:02d} jaywalking across roadway")
        elif state == "parked":
            self._alert_track(tr, "parked", "ℹ️", f"Car #{tr.id:02d} parked for >10 s")

    def _alert_track(self, tr, typ, sev, msg):
        if self.t - tr.alert_cd.get(typ, -99) > 25:
            tr.alert_cd[typ] = self.t
            self._push_alert(sev, typ, f"#{tr.id:02d}", msg)

    def analyze(self, tr):
        t = self.t
        sp = tr.speed
        hist = [p for p in tr.pos_hist if t - p[0] <= 5.0]
        if len(hist) >= 2:
            cx_ = sum(p[1] for p in hist) / len(hist)
            cy_ = sum(p[2] for p in hist) / len(hist)
            radius = max(math.hypot(p[1] - cx_, p[2] - cy_) for p in hist)
            span = max(1e-3, t - hist[0][0])
            avg_sp = math.hypot(tr.x - hist[0][1], tr.y - hist[0][2]) / span
        else:
            radius, avg_sp = 99.0, sp

        state = None
        if tr.cls == "person":
            in_h = H_ROAD[0] + 0.3 <= tr.y <= H_ROAD[1] - 0.3
            in_v = V_ROAD[0] + 0.3 <= tr.x <= V_ROAD[1] - 0.3
            if in_h or in_v:
                state = "crossing"
                if in_h and not (CX - CROSS_HALF - 1.5 <= tr.x <= CX + CROSS_HALF + 1.5):
                    state = "jaywalking"
                if in_v and not (CY - CROSS_HALF - 1.5 <= tr.y <= CY + CROSS_HALF + 1.5):
                    state = "jaywalking"
            else:
                if radius < 3.0 and avg_sp < 0.85:
                    if tr.rad_ok_since is None:
                        tr.rad_ok_since = t
                    if t - tr.rad_ok_since > 6.0:
                        state = "loitering"
                else:
                    tr.rad_ok_since = None
                if state is None:
                    if sp < 0.3:
                        if t - tr.still_since > 1.4:
                            state = "idle"
                    else:
                        tr.still_since = t
                    if state is None:
                        state = "running" if sp > 2.3 else "walking"
                if sp >= 0.3:
                    tr.still_since = t
        elif tr.cls == "car":
            if sp > SPEED_LIMIT:
                state = "speeding"
            elif abs(tr.turn_rate) > 0.55 and sp > 1.0:
                state = "turning"
            elif sp < 0.35:
                state = "parked" if t - tr.still_since > 10 else "stopped"
            else:
                tr.still_since = t
                state = "moving"
        elif tr.cls == "bicycle":
            if sp < 0.3:
                state = "stopped"
            else:
                tr.still_since = t
                state = "cruising"
        elif tr.cls == "dog":
            state = "sniffing" if sp < 0.3 else ("running" if sp > 2.6 else "walking")
        elif tr.cls == "bird":
            state = "flying"

        if state is None:
            return
        if tr.behavior == "—":
            self._commit(tr, state)
            return
        if state != tr.behavior:
            if state == tr.pending:
                tr.pend_n += 1
            else:
                tr.pending, tr.pend_n = state, 1
            if tr.pend_n >= 4:                       # hysteresis: anti-flicker
                self._commit(tr, state)
                tr.pending, tr.pend_n = None, 0
        else:
            tr.pending, tr.pend_n = None, 0
            if tr.segments:
                tr.segments[-1]["t1"] = t


# ════════════════════════════════════════════════════════════════
# 6. RENDERING — video-style frame with AI overlay
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def build_base():
    rng = np.random.default_rng(11)
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (25, 33, 28)
    g = rng.integers(-6, 7, (H, W))
    img = np.clip(img.astype(np.int16) + g[..., None], 0, 255).astype(np.uint8)
    cv2.rectangle(img, (P(2), P(2)), (P(PARK[2]), P(PARK[3])), (31, 45, 34), -1)
    for _ in range(14):                              # park trees
        x = rng.uniform(PARK[0] + 1, PARK[2] - 1)
        y = rng.uniform(PARK[1] + 1, PARK[3] - 1)
        r = int(rng.uniform(7, 14))
        cv2.circle(img, (P(x), P(y)), r, (int(rng.uniform(28, 40)), int(rng.uniform(52, 70)), int(rng.uniform(30, 42))), -1)
        cv2.circle(img, (P(x), P(y)), r, (20, 30, 22), 1)
    cv2.circle(img, (P(20), P(11)), 16, (52, 60, 66), -1)   # fountain
    cv2.circle(img, (P(20), P(11)), 10, (70, 110, 130), -1)
    cv2.rectangle(img, (P(V_ROAD[0] - SIDE), 0), (P(V_ROAD[1] + SIDE), H), (60, 64, 70), -1)
    cv2.rectangle(img, (0, P(H_ROAD[0] - SIDE)), (W, P(H_ROAD[1] + SIDE)), (60, 64, 70), -1)
    cv2.rectangle(img, (0, P(H_ROAD[0])), (W, P(H_ROAD[1])), (41, 45, 53), -1)
    cv2.rectangle(img, (P(V_ROAD[0]), 0), (P(V_ROAD[1]), H), (41, 45, 53), -1)
    for yy in (H_ROAD[0] + 0.25, H_ROAD[1] - 0.25):
        cv2.line(img, (0, P(yy)), (P(V_ROAD[0]), P(yy)), (120, 124, 130), 1)
        cv2.line(img, (P(V_ROAD[1]), P(yy)), (W, P(yy)), (120, 124, 130), 1)
    for xx in (V_ROAD[0] + 0.25, V_ROAD[1] - 0.25):
        cv2.line(img, (P(xx), 0), (P(xx), P(H_ROAD[0])), (120, 124, 130), 1)
        cv2.line(img, (P(xx), P(H_ROAD[1])), (P(xx), H), (120, 124, 130), 1)
    seg = 26
    gap = 20
    xs = 0
    while xs < W:                                    # dashed center lines
        x1 = min(xs + seg, W)
        if not (V_ROAD[0] * PPM - 14 < xs < V_ROAD[1] * PPM + 14):
            cv2.line(img, (xs, P(CY)), (x1, P(CY)), (190, 160, 60), 2)
        xs += seg + gap
    ys = 0
    while ys < H:
        y1 = min(ys + seg, H)
        if not (H_ROAD[0] * PPM - 14 < ys < H_ROAD[1] * PPM + 14):
            cv2.line(img, (P(CX), ys), (P(CX), y1), (190, 160, 60), 2)
        ys += seg + gap
    y = H_ROAD[0] + 0.5                              # zebra crosswalks
    while y < H_ROAD[1] - 0.8:
        cv2.rectangle(img, (P(CX - CROSS_HALF), P(y)), (P(CX + CROSS_HALF), P(y + 0.55)), (185, 188, 195), -1)
        y += 1.0
    x = V_ROAD[0] + 0.5
    while x < V_ROAD[1] - 0.8:
        cv2.rectangle(img, (P(x), P(CY - CROSS_HALF)), (P(x + 0.55), P(CY + CROSS_HALF)), (185, 188, 195), -1)
        x += 1.0
    sl = (200, 200, 205)                             # stop lines
    cv2.rectangle(img, (P(V_ROAD[0] - 1.6), P(CY + 0.2)), (P(V_ROAD[0] - 1.2), P(H_ROAD[1] - 0.2)), sl, -1)
    cv2.rectangle(img, (P(V_ROAD[1] + 1.2), P(H_ROAD[0] + 0.2)), (P(V_ROAD[1] + 1.6), P(CY - 0.2)), sl, -1)
    cv2.rectangle(img, (P(CX + 0.2), P(H_ROAD[0] - 1.6)), (P(V_ROAD[1] - 0.2), P(H_ROAD[0] - 1.2)), sl, -1)
    cv2.rectangle(img, (P(V_ROAD[0] + 0.2), P(H_ROAD[1] + 1.2)), (P(CX - 0.2), P(H_ROAD[1] + 1.6)), sl, -1)
    return img


BASE = build_base()


def _text(img, txt, pos, scale=0.42, color=(225, 228, 232), thick=1):
    cv2.putText(img, txt, (pos[0] + 1, pos[1] + 1), FONT, scale, (12, 14, 18), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, pos, FONT, scale, color, thick, cv2.LINE_AA)


def draw_frame(eng, opts, focus):
    img = BASE.copy()
    ls = light_state(eng.t)
    cols = {"green": (88, 205, 120), "yellow": (240, 198, 84), "red": (238, 92, 92)}
    lx, ly = P(V_ROAD[0] - 2.6), P(H_ROAD[0] - 2.6)
    cv2.circle(img, (lx, ly), 5, cols[ls["h"]], -1)
    _text(img, "EW", (lx + 9, ly + 4), 0.4, (210, 214, 220))
    cv2.circle(img, (lx, ly + 16), 5, cols[ls["v"]], -1)
    _text(img, "NS", (lx + 9, ly + 20), 0.4, (210, 214, 220))

    if opts.get("zones"):
        cv2.rectangle(img, (P(CX - CROSS_HALF), P(H_ROAD[0])), (P(CX + CROSS_HALF), P(H_ROAD[1])), (80, 200, 220), 1)
        cv2.rectangle(img, (P(V_ROAD[0]), P(CY - CROSS_HALF)), (P(V_ROAD[1]), P(CY + CROSS_HALF)), (80, 200, 220), 1)

    if opts.get("det"):                              # raw detection layer
        for d in eng.last_dets:
            x0, y0 = P(d.x - d.w / 2), P(d.y - d.h / 2)
            cv2.rectangle(img, (x0, y0), (P(d.x + d.w / 2), P(d.y + d.h / 2)), (150, 155, 160), 1)
            cv2.circle(img, (P(d.x), P(d.y)), 2, (150, 155, 160), -1)

    tracks = sorted([tr for tr in eng.tracker.tracks.values() if tr.hits >= 3], key=lambda z: z.id)
    for tr in tracks:
        meta = CLASS_META[tr.cls]
        cx_, cy_ = P(tr.x), P(tr.y)
        if opts.get("trail"):                        # fading trail
            pts = [(P(x), P(y)) for (_, x, y) in list(tr.pos_hist)[::2]]
            for i in range(1, len(pts)):
                a = i / max(1, len(pts) - 1)
                col = tuple(int(c * (0.25 + 0.75 * a)) for c in meta["rgb"])
                cv2.line(img, pts[i - 1], pts[i], col, 1 if a < 0.5 else 2, cv2.LINE_AA)
        if opts.get("vec") and tr.speed > 0.4:       # velocity vector
            cv2.arrowedLine(img, (cx_, cy_), (int(cx_ + tr.vx * 4.5), int(cy_ + tr.vy * 4.5)),
                            (235, 240, 245), 2, tipLength=0.25)
        if opts.get("box"):                          # bbox + labels
            L, Wd = meta["size"]
            c, sn = abs(math.cos(tr.heading)), abs(math.sin(tr.heading))
            bw = (c * L + sn * Wd) * 1.25 * PPM
            bh = (sn * L + c * Wd) * 1.25 * PPM
            x0, y0 = int(cx_ - bw / 2), int(cy_ - bh / 2)
            x1, y1 = int(cx_ + bw / 2), int(cy_ + bh / 2)
            is_focus = focus == tr.id
            if is_focus:
                cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (245, 248, 252), 1)
            cv2.rectangle(img, (x0, y0), (x1, y1), meta["rgb"], 3 if is_focus else 2)
            if opts.get("label"):
                txt = f"#{tr.id:02d} {meta['label']} {tr.conf:.0%}"
                (th, tw), _ = cv2.getTextSize(txt, FONT, 0.42, 1)
                cv2.rectangle(img, (x0, y0 - th - 8), (x0 + tw + 6, y0), meta["rgb"], -1)
                cv2.putText(img, txt, (x0 + 3, y0 - 5), FONT, 0.42, (15, 18, 22), 1, cv2.LINE_AA)
                bt = tr.behavior.upper()
                bc = hex2rgb(BEHAVIOR_COLOR.get(tr.behavior, "#888888"))
                (bth, btw), _ = cv2.getTextSize(bt, FONT, 0.36, 1)
                cv2.rectangle(img, (x0, y1 + 3), (x0 + btw + 8, y1 + bth + 9), bc, -1)
                cv2.putText(img, bt, (x0 + 4, y1 + bth + 5), FONT, 0.36, (16, 18, 22), 1, cv2.LINE_AA)

    if opts.get("hud"):
        _text(img, f"T+{fmt_t(eng.t)}   FRAME {eng.frame}   PIPE {eng.pipe_ms:.1f} ms", (14, 22), 0.45)
        _text(img, "CAM-01  •  SYNTHETIC FEED 960x540  •  AI OVERLAY", (W - 330, 22), 0.42, (170, 176, 184))
        L, m, col = 26, 8, (90, 96, 104)
        for sx, sy, dx, dy in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
            cv2.line(img, (sx, sy), (sx + dx * L, sy), col, 2)
            cv2.line(img, (sx, sy), (sx, sy + dy * L), col, 2)
        cv2.line(img, (m, H - m - 14), (m + 100, H - m - 14), (220, 224, 228), 2)
        _text(img, "10 m", (m + 38, H - m - 20), 0.4)
    return img


def show_image(img):
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# 7. EXPORT BUILDERS
# ════════════════════════════════════════════════════════════════
def track_card(eng, tr):
    return {
        "track_id": tr.id, "class": tr.cls, "confidence": round(tr.conf, 3),
        "x_m": round(tr.x, 2), "y_m": round(tr.y, 2),
        "velocity_ms": round(tr.speed, 2), "vx": round(tr.vx, 2), "vy": round(tr.vy, 2),
        "direction": compass_str(tr.vx, tr.vy), "heading_deg": round(bearing_of(tr.vx, tr.vy), 1),
        "behavior": tr.behavior, "age_s": round(eng.t - tr.first_t, 1),
    }


def confirmed_tracks(eng):
    return sorted([tr for tr in eng.tracker.tracks.values() if tr.hits >= 3], key=lambda z: z.id)


def snapshot_csv(eng):
    rows = [track_card(eng, tr) for tr in confirmed_tracks(eng)]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def log_csv(eng):
    df = pd.DataFrame(list(eng.log),
                      columns=["sim_time_s", "track_id", "class", "x_m", "y_m",
                               "speed_ms", "heading_deg", "behavior", "confidence"])
    return df.to_csv(index=False).encode("utf-8")


def json_report(eng):
    tracks = confirmed_tracks(eng)
    report = {
        "system": "AI Object Intelligence System",
        "pipeline": ["Detection", "Classification", "Tracking", "Behavior Analysis"],
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_time_s": round(eng.t, 1), "frames": eng.frame,
        "last_frame_counts": eng.last_counts,
        "objects": [track_card(eng, tr) for tr in tracks],
        "alerts": list(eng.alerts)[-50:],
        "behavior_segments": {f"#{tr.id:02d}": tr.segments for tr in tracks},
    }
    return json.dumps(report, indent=2).encode("utf-8")


def png_snapshot(eng):
    img = draw_frame(eng, {"box": True, "label": True, "trail": True, "vec": True,
                           "det": False, "hud": True, "zones": False}, None)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def time_in_state(tr, now, win=60):
    agg = {}
    for sg in tr.segments:
        t0 = max(sg["t0"], now - win)
        t1 = min(sg["t1"], now)
        if t1 > t0:
            agg[sg["state"]] = agg.get(sg["state"], 0.0) + (t1 - t0)
    return agg


def sparkline(hist):
    vals = list(hist)[-84:][::6]
    if len(vals) < 2:
        return "·"
    mx = max(max(vals), 0.1)
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int(v / mx * 7))] for v in vals)


# ════════════════════════════════════════════════════════════════
# 8. STREAMLIT UI
# ════════════════════════════════════════════════════════════════
def _fragment(run_every):
    def deco(f):
        if hasattr(st, "fragment"):
            return st.fragment(run_every=run_every)(f)
        return f
    return deco


if "engine" not in st.session_state:
    st.session_state.engine = Engine(SEED)     # pre-warmed → never a blank dashboard
st.session_state.setdefault("focus", None)

# ---------------- sidebar: mission control ----------------
with st.sidebar:
    st.markdown("## 🎛️ Mission Control")
    st.toggle("🔴 Live simulation", value=True, key="running")
    st.slider("Simulation speed", 0.5, 3.0, 1.0, 0.25, key="speed", format="%.2f×")
    st.slider("Target objects", 0, 24, 14, key="target")
    st.slider("Sensor noise", 0.0, 1.0, 0.25, 0.05, key="noise",
              help="Detection jitter, misses and confidence noise")

    eng0 = st.session_state.engine
    opts_ids = [None] + [tr.id for tr in confirmed_tracks(eng0)]

    def _fmt(v):
        if v is None:
            return "🌐 Auto (no focus)"
        tr = eng0.tracker.tracks.get(v)
        if tr is None:
            return f"#{v:02d}"
        return f"{CLASS_META[tr.cls]['emoji']} #{v:02d} · {tr.cls} · {tr.behavior}"

    idx = opts_ids.index(st.session_state.focus) if st.session_state.focus in opts_ids else 0
    sel = st.selectbox("🎯 Focus object", opts_ids, index=idx, format_func=_fmt)
    if sel != st.session_state.focus:
        st.session_state.focus = sel

    with st.expander("🎨 Overlays", expanded=False):
        st.checkbox("Bounding boxes", value=True, key="ov_box")
        st.checkbox("Labels", value=True, key="ov_lab")
        st.checkbox("Trails", value=True, key="ov_trail")
        st.checkbox("Velocity vectors", value=True, key="ov_vec")
        st.checkbox("Raw detections", value=False, key="ov_det")
        st.checkbox("Crosswalk zones", value=False, key="ov_zone")
        st.checkbox("HUD", value=True, key="ov_hud")

    st.caption("Spawn object")
    sb = st.columns(5)
    for col, (k, em) in zip(sb, [("person", "🚶"), ("car", "🚗"), ("bicycle", "🚲"),
                                 ("dog", "🐕"), ("bird", "🕊️")]):
        if col.button(em, key=f"spawn_{k}"):
            st.session_state.engine.world.add(k)

    if st.button("🔄 Reset scene (reload demo data)", use_container_width=True):
        st.session_state.engine = Engine(SEED)
        st.session_state.focus = None

    with st.expander("ℹ️ About / Pipeline"):
        st.markdown(
            """
            **Detection → Classification → Tracking → Behavior Analysis**

            - **Detection** — simulated neural detector (bbox + confidence + noise)
            - **Classification** — confidence gating + majority-vote refinement per track
            - **Tracking** — motion-predicted gated association, EMA smoothing, track IDs
            - **Behavior** — rule-based state machine with hysteresis (walking, loitering,
              crossing, jaywalking, turning, speeding, parked…)

            Fully local. **No API keys.** Scene auto-loads 15 s of pre-warmed history.
            """
        )

# ---------------- header ----------------
running_now = st.session_state.get("running", True)
st.markdown(
    f"### 🛰️ Object Intelligence System &nbsp;"
    f"<span style='color:{'#ff4b4b' if running_now else '#9aa0a6'};font-size:0.9rem'>"
    f"{'● LIVE' if running_now else '⏸ PAUSED'}</span>",
    unsafe_allow_html=True,
)
st.caption("Detection → Classification → Tracking → Behavior Analysis · unified intelligence per object · synthetic feed")


# ---------------- live panel (≈8 Hz) ----------------
@_fragment(0.12)
def live_panel():
    eng = st.session_state.engine
    if st.session_state.get("running", True):
        eng.tick(st.session_state.get("speed", 1.0),
                 st.session_state.get("noise", 0.25),
                 st.session_state.get("target", 14))
    opts = {"box": st.session_state.get("ov_box", True),
            "label": st.session_state.get("ov_lab", True),
            "trail": st.session_state.get("ov_trail", True),
            "vec": st.session_state.get("ov_vec", True),
            "det": st.session_state.get("ov_det", False),
            "zones": st.session_state.get("ov_zone", False),
            "hud": st.session_state.get("ov_hud", True)}
    focus = st.session_state.get("focus")
    img = draw_frame(eng, opts, focus)

    tracks = confirmed_tracks(eng)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("🎯 Active tracks", len(tracks))
    k2.metric("🔍 Detections", f"{eng.total_det:,}", f"+{eng.last_counts['det']}/frame")
    k3.metric("🆔 Unique IDs", eng.tracker.next_id - 1)
    crit = sum(1 for a in eng.alerts if a["sev"] == "🚨")
    k4.metric("🚨 Alerts", len(eng.alerts), f"{crit} critical")
    k5.metric("⏱ Sim clock", fmt_t(eng.t))
    k6.metric("⚙️ FPS", f"{eng.fps:.0f}")

    lc = eng.last_counts
    st.caption(
        f"**Pipeline:** 🔍 Detection `{lc['det']}` → 🏷️ Classification `{lc['cls']}` → "
        f"🎯 Tracking `{lc['trk']}` → 🧠 Behavior `{lc['beh']}` · latency `{eng.pipe_ms:.1f} ms`"
    )
    show_image(img)

    st.markdown("#### 🧠 Unified Object-Intelligence Cards")
    if not tracks:
        st.info("No confirmed tracks — raise *Target objects* or lower *Sensor noise*.")
    cols = st.columns(4)
    for i, tr in enumerate(tracks[:16]):
        meta = CLASS_META[tr.cls]
        bcol = BEHAVIOR_COLOR.get(tr.behavior, "#888888")
        alert_icon = " 🚨" if tr.behavior in ("loitering", "speeding", "jaywalking") else ""
        with cols[i % 4], st.container(border=True):
            st.markdown(
                f"{meta['emoji']} **{meta['label']}** `#{tr.id:02d}`{alert_icon}&nbsp; "
                f"<span style='background:{bcol};color:#111;padding:2px 10px;border-radius:12px;"
                f"font-size:0.72rem;font-weight:700'>{tr.behavior.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.progress(float(tr.conf), text=f"confidence {tr.conf:.0%}")
            a, b = st.columns(2)
            a.markdown(f"📍 ({tr.x:.1f}, {tr.y:.1f}) m")
            b.markdown(f"🚀 {tr.speed:.1f} m/s")
            st.caption(f"🧭 {compass_str(tr.vx, tr.vy)} · ⏱ {eng.t - tr.first_t:.0f}s · "
                       f"speed {sparkline(tr.speed_hist)}")
            if st.button("🎯 Focus", key=f"foc_{tr.id}", use_container_width=True):
                st.session_state.focus = tr.id
    if len(tracks) > 16:
        st.caption(f"+ {len(tracks) - 16} more tracks (see timeline below)")

    recent = list(eng.alerts)[-3:]
    if recent:
        st.caption("📟 " + " · ".join(f"{a['sev']} {fmt_t(a['t'])} {a['msg']}" for a in recent))


live_panel()


# ---------------- analytics panel (1 Hz) ----------------
@_fragment(1.0)
def analytics_panel():
    eng = st.session_state.engine
    tracks = confirmed_tracks(eng)

    st.markdown("#### 📈 Statistics")
    c1, c2, c3, c4 = st.columns(4)
    cfg = {"displayModeBar": False}
    if tracks:
        dfc = pd.DataFrame({"class": [CLASS_META[tr.cls]["label"] for tr in tracks]})
        cnt = dfc["class"].value_counts().reset_index()
        cnt.columns = ["class", "n"]
        fig1 = px.pie(cnt, values="n", names="class", hole=0.55, color="class",
                      color_discrete_map={m["label"]: rgb2hex(m["rgb"]) for m in CLASS_META.values()})
        fig1.update_layout(template="plotly_dark", height=240, margin=dict(l=8, r=8, t=24, b=8),
                           title="Classes", legend=dict(orientation="h", y=-0.15))
        c1.plotly_chart(fig1, use_container_width=True, config=cfg)

        dfb = pd.DataFrame({"behavior": [tr.behavior for tr in tracks]})
        bcnt = dfb["behavior"].value_counts().reset_index()
        bcnt.columns = ["behavior", "n"]
        fig2 = px.bar(bcnt, x="n", y="behavior", orientation="h", color="behavior",
                      color_discrete_map=BEHAVIOR_COLOR)
        fig2.update_layout(template="plotly_dark", height=240, margin=dict(l=8, r=8, t=24, b=8),
                           title="Behavior mix", showlegend=False)
        c2.plotly_chart(fig2, use_container_width=True, config=cfg)

        dfs = pd.DataFrame(list(eng.speed_trend), columns=["t", "v"])
        dfs["t"] = dfs["t"] - eng.t
        fig3 = px.line(dfs, x="t", y="v")
        fig3.update_layout(template="plotly_dark", height=240, margin=dict(l=8, r=8, t=24, b=8),
                           title="Avg speed of moving objects")
        c3.plotly_chart(fig3, use_container_width=True, config=cfg)

        fig4 = px.histogram(x=[tr.conf for tr in tracks], nbins=12)
        fig4.update_layout(template="plotly_dark", height=240, margin=dict(l=8, r=8, t=24, b=8),
                           title="Confidence distribution")
        fig4.update_xaxes(title="confidence")
        c4.plotly_chart(fig4, use_container_width=True, config=cfg)
    else:
        c1.info("Waiting for tracks…")

    st.markdown("#### 🕒 Behavior Timeline (last 60 s) — click a bar to focus that object")
    win = eng.t - 60
    rows = []
    for tr in tracks:
        for sg in tr.segments:
            if sg["t1"] < win:
                continue
            rows.append(dict(id=tr.id, label=f"#{tr.id:02d} · {tr.cls}",
                             t0=max(sg["t0"], win), t1=sg["t1"], state=sg["state"]))
    for a in eng.archived:
        if a["end"] < win:
            continue
        for sg in a["segs"]:
            if sg["t1"] < win:
                continue
            rows.append(dict(id=a["id"], label=f"#{a['id']:02d} · {a['cls']} ✦",
                             t0=max(sg["t0"], win), t1=min(sg["t1"], eng.t), state=sg["state"]))
    if rows:
        fig = go.Figure()
        for r in rows:
            fig.add_trace(go.Bar(
                x=[max(0.15, r["t1"] - r["t0"])], y=[r["label"]], base=[r["t0"]],
                orientation="h", marker=dict(color=BEHAVIOR_COLOR.get(r["state"], "#777")),
                customdata=[r["id"]], width=0.6,
                hovertemplate=f"<b>%{{y}}</b> · {r['state']}<br>%{{x:.1f}} s<extra></extra>"))
        n_lab = len({r["label"] for r in rows})
        fig.update_layout(barmode="overlay", template="plotly_dark",
                          height=min(430, 90 + 26 * n_lab),
                          margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
        fig.update_xaxes(range=[win, eng.t], title="sim time (s)")
        fig.update_yaxes(autorange="reversed")
        try:
            ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="tl_chart")
            if ev is not None and getattr(ev, "selection", None) and ev.selection.points:
                cd = ev.selection.points[0].get("customdata")
                try:
                    tid = int(cd[0]) if isinstance(cd, (list, tuple)) else int(cd)
                    st.session_state.focus = tid
                except Exception:
                    pass
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No behavior segments yet.")

    focus = st.session_state.get("focus")
    tr = eng.tracker.tracks.get(focus) if focus is not None else None
    if tr is not None and tr.hits >= 3:
        st.markdown("#### 🔬 Focus Detail")
        meta = CLASS_META[tr.cls]
        bcol = BEHAVIOR_COLOR.get(tr.behavior, "#888888")
        fl, fr = st.columns([1, 1.2])
        with fl:
            st.markdown(
                f"## {meta['emoji']} {meta['label']} `#{tr.id:02d}` &nbsp;"
                f"<span style='background:{bcol};color:#111;padding:3px 12px;border-radius:12px;"
                f"font-weight:700'>{tr.behavior.upper()}</span>", unsafe_allow_html=True)
            fc = track_card(eng, tr)
            st.markdown(
                f"| field | value |\n|---|---|\n"
                f"| Confidence | **{fc['confidence']:.0%}** |\n"
                f"| Position | ({fc['x_m']}, {fc['y_m']}) m |\n"
                f"| Velocity | {fc['velocity_ms']} m/s (vx {fc['vx']}, vy {fc['vy']}) |\n"
                f"| Direction | {fc['direction']} |\n"
                f"| Track age | {fc['age_s']} s |")
            hist = list(tr.pos_hist)
            sp = list(tr.speed_hist)
            n = min(len(hist), len(sp))
            if n > 5:
                figv = go.Figure(go.Scatter(x=[h[0] - eng.t for h in hist[-n:]], y=sp[-n:],
                                            mode="lines", line=dict(color="#4fc3f7", width=2),
                                            fill="tozeroy", fillcolor="rgba(79,195,247,0.12)"))
                figv.update_layout(template="plotly_dark", height=200,
                                   margin=dict(l=8, r=8, t=20, b=8), title="Speed (m/s)")
                st.plotly_chart(figv, use_container_width=True, config=cfg)
        with fr:
            agg = time_in_state(tr, eng.t)
            if agg:
                md = "**Time in state (last 60 s)**<br>"
                for state_, v in sorted(agg.items(), key=lambda z: -z[1]):
                    md += (f"<span style='color:{BEHAVIOR_COLOR.get(state_, '#888')}'>●</span> "
                           f"{state_}: <b>{v:.1f} s</b><br>")
                st.markdown(md, unsafe_allow_html=True)
            segs = [s for s in tr.segments if s["t1"] > eng.t - 60]
            if segs:
                stdf = pd.DataFrame(segs)
                stdf["start"] = (stdf["t0"] - eng.t).round(1)
                stdf["dur_s"] = (stdf["t1"] - stdf["t0"]).round(1)
                st.dataframe(stdf[["start", "dur_s", "state"]].tail(14),
                             hide_index=True, use_container_width=True, height=220)

    st.markdown("#### 🚨 Alert Feed")
    if eng.alerts:
        adf = pd.DataFrame(list(eng.alerts)[::-1])
        adf["time"] = adf["t"].map(fmt_t)
        st.dataframe(adf[["time", "sev", "type", "track", "msg"]],
                     hide_index=True, use_container_width=True, height=190)
    else:
        st.caption("No alerts yet.")

    st.markdown("#### 💾 Export")
    d1, d2, d3, d4 = st.columns(4)
    stamp = int(eng.t)
    d1.download_button("📄 Snapshot CSV", snapshot_csv(eng),
                       file_name=f"intel_snapshot_{stamp}s.csv", mime="text/csv",
                       use_container_width=True)
    d2.download_button("🗂 Track log CSV", log_csv(eng),
                       file_name=f"intel_tracklog_{stamp}s.csv", mime="text/csv",
                       use_container_width=True)
    d3.download_button("🧾 JSON report", json_report(eng),
                       file_name=f"intel_report_{stamp}s.json", mime="application/json",
                       use_container_width=True)
    d4.download_button("🖼 Frame PNG", png_snapshot(eng),
                       file_name=f"intel_frame_{stamp}s.png", mime="image/png",
                       use_container_width=True)
    st.caption("Exports reflect the live simulation state at render time.")


analytics_panel()