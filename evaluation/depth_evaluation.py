import pandas as pd
import matplotlib.pyplot as plt

def plot_depth_deviation(csv_file: pd.DataFrame, tracker_id: int):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Filter data for the given tracker_id
    tracker_data = df[df["tracker_id"] == tracker_id]

    # Plot the deviation over frames
    plt.figure(figsize=(12, 6))
    plt.scatter(tracker_data["Frame"], tracker_data["depth"], marker="o", linestyle="-", color="b")
    plt.plot(tracker_data["Frame"], tracker_data["estimated_depth"], marker="o", linestyle="-", color="r")
    plt.title(f"Depth Deviation for Tracker ID {tracker_id}")
    plt.xlabel("Frame")
    plt.ylabel("Depth [m]")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_file = "X:\Life\TFG\Coding\Project_v2\data\Man_Standing_in_a_crowd\objects_detections_processed.csv"
    tracker_id = 1
    plot_depth_deviation(csv_file, tracker_id)