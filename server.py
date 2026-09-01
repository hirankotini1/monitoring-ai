"""
Root Server Entrypoint for Cloud & Render Deployment
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(root_dir, "Emotion Monitor")
if os.path.exists(sub_dir) and sub_dir not in sys.path:
    sys.path.insert(0, sub_dir)
    os.chdir(sub_dir)

from server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
