"""UrbanFlow AI — Live 3D Digital Twin & Command Center Demo Launcher.

Starts the Uvicorn ASGI server and streams real-time traffic micro-simulation
to the Three.js 3D WebGL Command Center.
"""
from __future__ import annotations

import os
import shutil
import signal
import sys
import socket
import urllib.request
import json
import subprocess
import time
import webbrowser
from pathlib import Path
import uvicorn

# Ensure urbanflow is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from urbanflow.config import HOST, PORT


def is_port_bound(host: str, port: int) -> bool:
    """Checks whether the specified TCP host:port is currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def check_existing_server(host: str, port: int) -> bool:
    """Checks whether the currently bound port is an active UrbanFlow server."""
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/api/status",
            headers={"User-Agent": "UrbanFlowLauncher"},
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            return data.get("platform") == "UrbanFlow AI"
    except Exception:
        return False


def kill_process_on_port(port: int) -> None:
    """Terminate the listener after the caller verifies it is UrbanFlow."""
    try:
        if sys.platform != "win32":
            if shutil.which("lsof"):
                output = subprocess.check_output(
                    ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], text=True
                )
                for pid in set(output.split()):
                    os.kill(int(pid), signal.SIGTERM)
            elif shutil.which("fuser"):
                subprocess.run(["fuser", "-k", "-TERM", f"{port}/tcp"], check=True)
            else:
                raise OSError("Restart requires lsof or fuser on this platform.")
            return
        out = subprocess.check_output(["netstat", "-ano"], text=True)
        for line in out.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                pid = int(parts[4])
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"  [RESTART] Cleanly terminated previous process on port {port} (PID: {pid})")
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"  [RESTART] {error}", file=sys.stderr)


def main():
    url = f"http://{HOST}:{PORT}"

    # Handle port collisions and restarts
    if is_port_bound(HOST, PORT):
        if ("--restart" in sys.argv or "-r" in sys.argv) and check_existing_server(HOST, PORT):
            kill_process_on_port(PORT)
            time.sleep(1.0)
            if is_port_bound(HOST, PORT):
                sys.exit(f"Port {PORT} is still occupied. Stop the previous server and retry.")
        elif check_existing_server(HOST, PORT):
            print("=" * 72)
            print("  URBANFLOW AI IS ALREADY RUNNING!")
            print("=" * 72)
            print(f"  Command Center is active at: {url}")
            print(f"  Live WebSocket endpoint   : ws://{HOST}:{PORT}/ws/live")
            print("  To restart cleanly, run   : python run_demo.py --restart")
            print("=" * 72)
            if "--open" in sys.argv or "-o" in sys.argv or "--no-browser" not in sys.argv:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            return
        else:
            print("=" * 72)
            print(f"  [ERROR] Port {PORT} is currently in use by another application.")
            print(f"  Please free port {PORT} or specify a different port via URBANFLOW_PORT.")
            print("=" * 72)
            sys.exit(1)

    print("=" * 72)
    print("  URBANFLOW AI — INTELLIGENT TRAFFIC MANAGEMENT PLATFORM")
    print("  Geographic 3D Urban Digital Twin & Real-Time Simulation Engine")
    print("=" * 72)
    print(f"  Command Center URL: {url}")
    print(f"  API Documentation : {url}/api/status")
    print(f"  Live WebSocket    : ws://{HOST}:{PORT}/ws/live")
    print("-" * 72)
    print("  Starting Uvicorn ASGI server... Press Ctrl+C to stop.")
    print("=" * 72)

    # Optional auto-open browser
    if "--open" in sys.argv or "-o" in sys.argv or "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    uvicorn.run(
        "urbanflow.server.app:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
