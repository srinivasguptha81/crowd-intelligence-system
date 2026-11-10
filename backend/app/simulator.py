"""Physics-based synthetic crowd video stream generator for offline lab testing (demo://*)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class _Agent:
    x: float
    y: float
    vx: float
    vy: float
    w: int
    h: int
    color: Tuple[int, int, int]
    fallen: bool = False


class SyntheticCrowdStream:
    """
    Generates 640x360 synthetic venue concourse frames simulating:
    - 'crowd-sim': cyclical normal flow -> counter-flow -> panic scattering -> crush buildup
    - 'panic-sim': sudden outward radial panic dispersion + fallen individual
    - 'crush-sim': high-density bottleneck convergence leading to movement stagnation
    """

    def __init__(self, mode: str = "crowd-sim", width: int = 640, height: int = 360) -> None:
        self.mode = mode.replace("demo://", "").lower()
        self.width = width
        self.height = height
        self.tick = 0
        self.rng = random.Random(42)
        self.agents: list[_Agent] = []
        self._init_agents(18 if self.mode != "crush-sim" else 28)

    def _init_agents(self, count: int) -> None:
        self.agents.clear()
        for i in range(count):
            y = self.rng.uniform(70, self.height - 65)
            scale = 0.65 + 0.55 * (y / self.height)
            w = int(20 * scale)
            h = int(48 * scale)
            x = self.rng.uniform(40, self.width - 40)
            vx = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(1.4, 2.6)
            vy = self.rng.uniform(-0.5, 0.5)
            col = (
                self.rng.randint(135, 220),
                self.rng.randint(135, 220),
                self.rng.randint(150, 235),
            )
            self.agents.append(_Agent(x=x, y=y, vx=vx, vy=vy, w=w, h=h, color=col))

    def next_frame(self) -> tuple[np.ndarray, list[list[int]]]:
        self.tick += 1
        phase_len = 35
        if self.mode == "panic-sim":
            stage = "panic" if (self.tick % 60) >= 15 else "normal"
        elif self.mode == "crush-sim":
            stage = "crush" if (self.tick % 70) >= 12 else "normal"
        else:
            cycle = self.tick % (phase_len * 4)
            if cycle < phase_len:
                stage = "normal"
            elif cycle < phase_len * 2:
                stage = "counterflow"
            elif cycle < phase_len * 3:
                stage = "panic"
            else:
                stage = "crush"

        cx_mid, cy_mid = self.width * 0.5, self.height * 0.52

        # Adjust target agent population dynamically for crush buildup vs normal flow
        target_n = 30 if stage == "crush" else 18
        while len(self.agents) < target_n:
            y = self.rng.uniform(90, self.height - 70)
            scale = 0.65 + 0.55 * (y / self.height)
            self.agents.append(
                _Agent(
                    x=self.rng.uniform(80, self.width - 80),
                    y=y,
                    vx=self.rng.uniform(-2.0, 2.0),
                    vy=self.rng.uniform(-0.8, 0.8),
                    w=int(20 * scale),
                    h=int(48 * scale),
                    color=(175, 190, 210),
                )
            )
        if len(self.agents) > target_n:
            self.agents = self.agents[:target_n]

        frame = np.full((self.height, self.width, 3), (28, 34, 44), dtype=np.uint8)
        # Draw floor perspective lines for visual reference
        cv2.line(frame, (0, int(self.height * 0.33)), (self.width, int(self.height * 0.33)), (46, 56, 70), 1)
        cv2.line(frame, (0, int(self.height * 0.66)), (self.width, int(self.height * 0.66)), (46, 56, 70), 1)

        boxes: list[list[int]] = []
        for idx, ag in enumerate(self.agents):
            ag.fallen = False
            if stage == "normal":
                # Smooth horizontal corridor walk
                if abs(ag.vx) < 1.0:
                    ag.vx = 2.0 if idx % 2 == 0 else 1.8
                ag.vx = abs(ag.vx)
                ag.vy *= 0.8
            elif stage == "counterflow":
                # Two strong opposing streams colliding in center
                ag.vx = 4.2 if idx % 2 == 0 else -4.2
                ag.vy = math.sin(self.tick * 0.2 + idx) * 1.2
            elif stage == "panic":
                # Radial outward scattering from concourse center + 1 fallen individual
                dx = ag.x - cx_mid
                dy = ag.y - cy_mid
                dist = max(10.0, math.hypot(dx, dy))
                ag.vx = (dx / dist) * 8.5 + self.rng.uniform(-1.5, 1.5)
                ag.vy = (dy / dist) * 6.5 + self.rng.uniform(-1.5, 1.5)
                if idx == 0:
                    ag.fallen = True
                    ag.vx = 0.2
                    ag.vy = 0.1
            elif stage == "crush":
                # Converge tightly into center-right bottleneck and stagnate
                tx, ty = self.width * 0.56, self.height * 0.52
                dx = tx - ag.x + self.rng.uniform(-28, 28)
                dy = ty - ag.y + self.rng.uniform(-22, 22)
                dist = math.hypot(dx, dy)
                if dist > 48:
                    ag.vx = (dx / dist) * 3.4
                    ag.vy = (dy / dist) * 2.4
                else:
                    # High compression micro-tremor deadlock (near-zero flow!)
                    ag.vx = self.rng.uniform(-0.35, 0.35)
                    ag.vy = self.rng.uniform(-0.35, 0.35)

            ag.x += ag.vx
            ag.y += ag.vy

            if ag.x < 20:
                ag.x = self.width - 25 if stage in {"normal", "counterflow"} else 22
            elif ag.x > self.width - 20:
                ag.x = 25 if stage in {"normal", "counterflow"} else self.width - 22
            if ag.y < 52:
                ag.y = 54
                ag.vy = abs(ag.vy)
            elif ag.y > self.height - 25:
                ag.y = self.height - 27
                ag.vy = -abs(ag.vy)

            bw = ag.h if ag.fallen else ag.w
            bh = max(14, int(ag.w * 0.85)) if ag.fallen else ag.h
            x1 = max(2, int(ag.x - bw / 2))
            y1 = max(40, int(ag.y - bh / 2))
            x2 = min(self.width - 2, x1 + bw)
            y2 = min(self.height - 2, y1 + bh)

            # Render synthetic pedestrian silhouette (torso + head) for rich optical flow textures
            cv2.rectangle(frame, (x1, y1 + bh // 4), (x2, y2), ag.color, -1)
            cv2.circle(frame, ((x1 + x2) // 2, y1 + max(4, bh // 6)), max(4, min(bw, bh) // 4), (215, 200, 185), -1)
            boxes.append([x1, y1, x2, y2])

        cv2.putText(
            frame,
            f"SIMULATED STREAM ({stage.upper()}) - NON-CCTV TEST SIGNAL",
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 220),
            1,
            cv2.LINE_AA,
        )
        return frame, boxes

