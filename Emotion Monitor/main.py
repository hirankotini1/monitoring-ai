import os
import sys
import subprocess


def main():
    project_folder = os.path.dirname(os.path.abspath(__file__))
    dashboard_file = os.path.join(project_folder, "dashboard.py")

    if not os.path.exists(dashboard_file):
        print("ERROR: dashboard.py was not found.")
        print(f"Expected location: {dashboard_file}")
        sys.exit(1)

    print("=" * 60)
    print("              EMOTION MONITOR")
    print("=" * 60)
    print("Starting dashboard...")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, dashboard_file],
            cwd=project_folder
        )

        if result.returncode != 0:
            print("\nDashboard stopped with an error.")
            print(f"Exit code: {result.returncode}")

    except KeyboardInterrupt:
        print("\nEmotion Monitor stopped.")

    except Exception as error:
        print(f"\nERROR: {error}")

    finally:
        print("\nEmotion Monitor closed.")


if __name__ == "__main__":
    main()