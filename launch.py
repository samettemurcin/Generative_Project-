"""
launch.py — Start the web demo (static site + Streamlit) with one command.

Usage:
    python launch.py            # opens browser automatically
    python launch.py --no-open  # skip auto-open
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"
WEB_PORT = 8080
STREAMLIT_PORT = 8501
HF_HOME = r"D:\ml-cache\huggingface"


def serve_web() -> None:
    """Serve the static web/ directory on WEB_PORT."""
    os.chdir(WEB_DIR)

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
            pass  # suppress request logs

    server = HTTPServer(("localhost", WEB_PORT), QuietHandler)
    print(f"[web]       http://localhost:{WEB_PORT}")
    server.serve_forever()


def start_streamlit() -> subprocess.Popen:
    """Launch Streamlit as a subprocess."""
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "streamlit_app.py"),
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    print(f"[streamlit] http://localhost:{STREAMLIT_PORT}")
    return proc


def wait_for_streamlit(timeout: int = 30) -> bool:
    """Poll until Streamlit is accepting connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", STREAMLIT_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    print("Starting demo servers…")

    # Static web server in a daemon thread
    t = threading.Thread(target=serve_web, daemon=True)
    t.start()

    # Streamlit subprocess
    st_proc = start_streamlit()

    print("Waiting for Streamlit to be ready…")
    ready = wait_for_streamlit(timeout=60)
    if not ready:
        print("[warn] Streamlit didn't respond in 60s — opening browser anyway.")

    url = f"http://localhost:{WEB_PORT}"
    print(f"\nDemo ready → {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        st_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down…")
        st_proc.terminate()
        st_proc.wait()


if __name__ == "__main__":
    main()