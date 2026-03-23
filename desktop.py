"""
Desktop Launcher

Wraps the Streamlit medical imaging app in a native desktop window
using pywebview. Starts the Streamlit server as a subprocess and
opens a native OS window pointing to it.
"""

import atexit
import signal
import socket
import subprocess
import sys
import time


def find_free_port():
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(port, timeout=30):
    """Wait until the Streamlit server is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def main():
    try:
        import webview
    except ImportError:
        print("pywebview is required for desktop mode.")
        print("Install it with: pip install pywebview")
        sys.exit(1)

    port = find_free_port()

    # Start Streamlit as a subprocess
    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def cleanup():
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Wait for Streamlit to start
    if not wait_for_server(port):
        print("Error: Streamlit server failed to start within 30 seconds.")
        cleanup()
        sys.exit(1)

    url = f"http://localhost:{port}"

    # Create native window
    webview.create_window(
        title="Medical Imaging Analysis",
        url=url,
        width=1280,
        height=900,
        min_size=(800, 600),
    )

    webview.start()
    cleanup()


if __name__ == "__main__":
    main()
