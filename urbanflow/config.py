"""UrbanFlow AI Configuration Settings."""
from __future__ import annotations

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "urbanflow" / "web"

# Server Settings
HOST = os.getenv("URBANFLOW_HOST", "127.0.0.1")
PORT = int(os.getenv("URBANFLOW_PORT", "8001"))
DEBUG = os.getenv("URBANFLOW_DEBUG", "false").lower() in ("true", "1", "yes")

# Simulation Settings
DEFAULT_SIMULATOR = os.getenv("URBANFLOW_SIMULATOR", "kinematic")  # "kinematic" or "sumo"
SIMULATION_TICK_RATE = float(os.getenv("URBANFLOW_TICK_RATE", "0.05"))  # 20 Hz
STREAM_BROADCAST_RATE = float(os.getenv("URBANFLOW_STREAM_RATE", "0.05"))  # 20 Hz
DEFAULT_NETWORK_NAME = "central_corridor"
