import os
import threading
from gui import build_gui
from flask_app import run_flask

def start_flask():
    # Start Flask app in Docker
    if os.environ.get("RUNNING_IN_DOCKER"):
        run_flask()

if __name__ == "__main__":
    # Create a thread to run Flask
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Run the GUI only on the local machine
    if not os.environ.get("RUNNING_IN_DOCKER"):
        build_gui()
