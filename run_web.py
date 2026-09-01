import os
import sys
import subprocess
import webbrowser

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(root_dir, "Emotion Monitor")
    app_file = os.path.join(app_dir, "server.py")
    if not os.path.exists(app_file):
        app_file = os.path.join(app_dir, "app.py")

    if not os.path.exists(app_file):
        print(f"Error: Could not locate server entrypoint at {app_file}")
        sys.exit(1)

    print("=" * 65)
    print("      STARTING EMOTION & ATTENTION MONITOR WEB SERVER")
    print("=" * 65)
    print("  Localhost URL: http://localhost:5000")
    print("  To stop server: Press Ctrl+C")
    print("=" * 65)

    # Launch browser after a short delay
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:5000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        subprocess.run([sys.executable, app_file], cwd=app_dir)
    except KeyboardInterrupt:
        print("\n[OK] Server stopped gracefully.")

if __name__ == "__main__":
    main()
