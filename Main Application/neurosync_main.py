import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import subprocess
from flask import send_file, jsonify
import threading
import os
from dash.exceptions import PreventUpdate
import dash_daq as daq

# Initialize Tkinter root
root = tk.Tk()
root.title("EEG, HRV, and Video Capture Platform")

# Define Frames for better organization
left_frame = tk.Frame(root)
left_frame.grid(row=0, column=0, sticky="nsew")

right_frame = tk.Frame(root)
right_frame.grid(row=0, column=1, sticky="nsew")

# Configure weight so that both frames expand properly
root.grid_columnconfigure(0, weight=2)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# Create a video label in GUI to show video frames
video_label = tk.Label(root)
video_label.grid(row=0, column=1, rowspan=10, padx=10, pady=10)

# Control Buttons in left_frame
button_frame = tk.Frame(left_frame)
button_frame.pack()

# Global variables
eeg_data = {ch: [] for ch in ["AF3", "AF4", "F3", "F4", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2", "C3", "C4"]}
hrv_data = []
timestamps = []
hrv_timestamps = []  # Independent timestamps for HRV
stimulus_active = False  # Flag to indicate active stimulus
stimulus_effect_duration = 2  # Duration of stimulus effect in seconds
selected_channels = {ch: tk.BooleanVar(value=True) for ch in eeg_data.keys()}  # Default: All channels selected
stimuli_log = []  # To log stimulus type for each timestamp
sampling_window = 1000  # Number of data points to display (e.g., 1 second of HRV data)
total_recording_time = None  # Global variable to store total recording time

# Global variables
running = False
video_capture = None
video_writer = None
experiment_folder = ""
experiment_start_time = None
start_interval = None
end_interval = None


# Define unique properties for EEG channels
channel_properties = {
    "AF3": {"frequency": 10, "offset": 0, "amplitude": 50},
    "AF4": {"frequency": 12, "offset": 100, "amplitude": 50},
    "F3": {"frequency": 8, "offset": 200, "amplitude": 50},
    "F4": {"frequency": 14, "offset": 300, "amplitude": 50},
    "F7": {"frequency": 6, "offset": 400, "amplitude": 50},
    "F8": {"frequency": 16, "offset": 500, "amplitude": 50},
    "T7": {"frequency": 7, "offset": 600, "amplitude": 50},
    "T8": {"frequency": 9, "offset": 700, "amplitude": 50},
    "P7": {"frequency": 11, "offset": 800, "amplitude": 50},
    "P8": {"frequency": 13, "offset": 900, "amplitude": 50},
    "O1": {"frequency": 15, "offset": 1000, "amplitude": 50},
    "O2": {"frequency": 17, "offset": 1100, "amplitude": 50},
    "C3": {"frequency": 18, "offset": 1200, "amplitude": 50},
    "C4": {"frequency": 20, "offset": 1300, "amplitude": 50},
}

# Global variable to track if the experiment is stopped
experiment_stopped = True  # Ensure visualization is not blocked on startup
data_lock = threading.Lock()
experiment_start_flag = threading.Event()
video_capture = None
video_writer = None
video_running = False

def convert_video_to_browser_friendly_format(input_video_path):
    # Define the output video path by appending '_converted' to the filename
    output_video_path = input_video_path.replace(".mp4", "_converted.mp4")

    # Provide the full path to ffmpeg executable
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Change this to your actual path

    # Command to convert the video using ffmpeg
    command = [
        ffmpeg_path,  # Use the full path here
        '-i', input_video_path,
        '-r', '10',  # Set the output frame rate to match the original recording frame rate (10 fps)
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '192k',
        output_video_path
    ]

    try:
        # Run the ffmpeg command to convert the video
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

    return output_video_path

# Function to trigger a stimulus with relative timestamps
def trigger_stimulus(stimulus_type):
    global stimulus_active, stimulus_type_active, stimuli_log
    current_time = time.time()
    relative_time = current_time - experiment_start_time  # Calculate relative time
    stimulus_active = True
    stimulus_type_active = stimulus_type  # Set the active stimulus type

    stimuli_log.append({"start_time": relative_time, "type": stimulus_type})
    print(f"{stimulus_type} stimulus triggered at {relative_time}")

    def deactivate_stimulus():
        global stimulus_active, stimulus_type_active
        time.sleep(stimulus_effect_duration)
        stimulus_active = False
        stimulus_type_active = None  # Reset the stimulus type

        # Update the end time in the log for the last stimulus entry
        stimuli_log[-1]["end_time"] = relative_time + stimulus_effect_duration

    threading.Thread(target=deactivate_stimulus, daemon=True).start()

# Function to start the Dash visualization
def visualize_data_with_dash():
    threading.Thread(target=run_dash_app, daemon=True).start()

def save_experiment_data_to_csv(eeg_data, ecg_data, ecg_timestamps, experiment_folder):
    # Ensure EEG DataFrame
    if not isinstance(eeg_data, pd.DataFrame):
        eeg_data = pd.DataFrame(eeg_data)

    # Construct file paths for CSV files
    eeg_file_path = os.path.join(experiment_folder, "eeg_data.csv")
    ecg_file_path = os.path.join(experiment_folder, "ecg_data.csv")

    # Save EEG data with timestamp and stimulus
    eeg_data.to_csv(eeg_file_path, index=False)

    # Create a DataFrame for ECG that also includes the stimuli during the timestamps
    ecg_df = pd.DataFrame({
        'Timestamp': ecg_timestamps,
        'ECG': ecg_data
    })

    # Remove duplicate timestamps from EEG data to ensure proper reindexing
    eeg_data = eeg_data.loc[~eeg_data['Timestamp'].duplicated(keep='first')]

    # Merge Internal Clock and Stimulus to ECG Data
    merged_df = pd.merge_asof(
        ecg_df.sort_values("Timestamp"),
        eeg_data[["Timestamp", "Internal Clock", "Stimulus"]].sort_values("Timestamp"),
        on="Timestamp",
        direction="nearest"
    )

    # Fill missing values for Stimulus (if any)
    merged_df["Stimulus"].fillna("None", inplace=True)

    # Save ECG data with merged columns
    merged_df.to_csv(ecg_file_path, index=False)

    print(f"EEG data saved to {eeg_file_path}")
    print(f"ECG data saved to {ecg_file_path}")

# Function to start the experiment
def start_experiment():
    global running, experiment_folder, experiment_start_time
    if running:
        messagebox.showwarning("Warning", "Experiment is already running!")
        return
    running = True
    experiment_folder = "data/experiment_" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(experiment_folder, exist_ok=True)

    # Set the start time after threads are ready
    experiment_start_time = time.time() + 2  # Add a 2-second delay to ensure all threads are fully ready

    # Start all threads but wait until the start flag
    threading.Thread(target=capture_video, daemon=True).start()
    threading.Thread(target=simulate_eeg, daemon=True).start()
    threading.Thread(target=simulate_hrv, daemon=True).start()

    # Let all threads start simultaneously after 1-second delay
    experiment_start_flag.set()

    messagebox.showinfo("Experiment Started", "Experiment has begun!")

def initialize_video():
    global video_capture, video_running
    # Open the video capture when the application starts
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return
    video_running = True
    # Start the thread that continuously captures video frames
    threading.Thread(target=show_video_feed, daemon=True).start()

def show_video_feed():
    global video_capture, video_running
    while video_running:
        ret, frame = video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        time.sleep(0.1)  # Update the frame every 100ms

# Function to handle video capture
def capture_video():
    global video_capture, video_writer, experiment_start_time, running, video_file_path

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    video_file_path = os.path.join(experiment_folder, "experiment_video.mp4")
    video_writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    # Wait until experiment officially starts
    experiment_start_flag.wait()

    start_time = time.time()
    frame_counter = 0

    while running:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Calculate relative time and timestamp
        current_time = time.time()
        relative_time = current_time - experiment_start_time
        timestamp_text = f"Time: {relative_time:.2f}s"

        # Add the timestamp to the video frame
        cv2.putText(frame, timestamp_text, (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Write frame to video
        video_writer.write(frame)

        # Control the frame rate (ensure 10 fps)
        frame_counter += 1
        elapsed_time = time.time() - start_time
        sleep_time = (frame_counter / 10) - elapsed_time  # 10 fps
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Release resources
    video_capture.release()
    video_writer.release()

    if os.path.exists(video_file_path) and os.path.getsize(video_file_path) > 0:
        print(f"Video saved successfully at {video_file_path}")
    else:
        print("Failed to save video, please check the path and video writer setup.")

# Function to simulate HRV data
def simulate_hrv():
    global hrv_data, hrv_timestamps
    sampling_rate = 256  # Adjusted to match HRV sampling rate
    anomaly_triggered = False

    while running:
        current_time = time.time()
        relative_time = current_time - experiment_start_time  # Calculate relative time

        # Introduce random anomalies
        if int(relative_time) % 15 == 0 and relative_time > 0:  # Every 10 seconds
            anomaly_triggered = True
        else:
            anomaly_triggered = np.random.uniform(0, 1) < 0.1  # Slightly higher probability

        hrv_value = 60 + 5 * np.sin(2 * np.pi * 0.2 * current_time % 1)  # HR oscillates around 60 BPM
        if stimulus_active or anomaly_triggered:
            hrv_value += np.random.uniform(-10, 10)  # Larger range for anomalies

        hrv_data.append(hrv_value)
        hrv_timestamps.append(current_time - experiment_start_time)  # Store relative timestamp
        time.sleep(1 / sampling_rate)

# Example in simulate_eeg()
def simulate_eeg():
    global eeg_data, timestamps, internal_clock
    anomaly_triggered = False
    while running:
        current_time = time.time()
        relative_time = current_time - experiment_start_time  # Calculate relative time

        # Decide randomly to introduce an anomaly
        if int(relative_time) % 15 == 0 and relative_time > 0:  # Every 10 seconds
            anomaly_triggered = True
        else:
            anomaly_triggered = np.random.uniform(0, 1) < 0.1  # Slightly higher probability

        data_row = {"Timestamp": relative_time}
        for ch, props in channel_properties.items():
            if selected_channels[ch].get():
                eeg_value = (
                    props["amplitude"] * np.sin(2 * np.pi * props["frequency"] * relative_time % 1) +
                    np.random.normal(0, 5)
                )
                if stimulus_active or anomaly_triggered:
                    eeg_value += np.random.uniform(-100, 100)  # Larger range to represent anomaly
                data_row[ch] = eeg_value + props["offset"]
            else:
                data_row[ch] = None

        data_row["Stimulus"] = stimulus_type_active if stimulus_active else "None"
        relative_time = current_time - experiment_start_time  # Calculate relative time
        data_row["Internal Clock"] = relative_time
        internal_clock = data_row["Internal Clock"]

        # Acquire lock before modifying shared data
        with data_lock:
            eeg_data = pd.concat([eeg_data, pd.DataFrame([data_row])], ignore_index=True) if isinstance(eeg_data, pd.DataFrame) else pd.DataFrame([data_row])
            timestamps.append(relative_time)

        time.sleep(1 / 256)

# Stop the experiment and video capture
def stop_experiment():
    global running, video_running, total_recording_time
    running = False
    video_running = False

    # Calculate the total recording time
    if experiment_start_time is not None:
        total_recording_time = time.time() - experiment_start_time

    # Make sure to release video writer resources if running
    if video_writer:
        video_writer.release()

    if video_capture:
        video_capture.release()

    messagebox.showinfo("Experiment Stopped", "Experiment has ended.")
    save_experiment_data_to_csv(eeg_data, hrv_data, hrv_timestamps, experiment_folder)

# Real-time plotting
def update_plot(frame):
    if not running:
        return

    # Use total_recording_time for the x-axis limit if available
    time_window = total_recording_time if total_recording_time else 1.0  # Default to 1 second if not yet set

    # Calculate the number of samples to display for EEG
    eeg_samples_to_display = int(time_window * 256)  # Adjust to reflect new sampling rate

    # Update EEG plot
    ax_eeg.clear()
    for ch in [key for key in eeg_data.keys() if key not in ["Timestamp", "Stimulus", "Internal Clock"]]:
        if selected_channels[ch].get():
            y_data = eeg_data[ch][-eeg_samples_to_display:]
            x_data = np.linspace(0, time_window, len(y_data))  # Generate time axis based on samples
            ax_eeg.plot(x_data, y_data, label=ch)
    ax_eeg.set_title("EEG Channels (With Stimulus Effects)")
    ax_eeg.set_xlim(0, time_window)
    ax_eeg.legend(loc="upper right")
    ax_eeg.grid(True)

    # Calculate the number of samples to display for ECG
    ecg_samples_to_display = int(time_window * 256)  # Adjust to reflect new sampling rate


    # Update HRV plot
    ax_hrv.clear()
    ax_hrv.plot(hrv_data[-50:], color="red", label="HRV")
    ax_hrv.set_title("HRV Signal")
    ax_hrv.grid(True)

    fig.canvas.draw()

app = dash.Dash(__name__)

@app.server.route('/video/<path:path>')
def serve_video(path):
    # Dynamically determine the video directory based on the given path.
    # It will look in the folder where the videos were saved.
    video_directory = os.path.dirname(video_file_path)
    video_path = os.path.join(video_directory, path)

    # Ensure that the video exists
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4', conditional=True)
    else:
        # If not found, return a 404 error
        return jsonify({"error": "Video not found"}), 404

# Function to start the Dash app in a separate thread
def run_dash_app():
    global video_file_path

    app.clientside_callback(
        """
        function(interval, n_clicks) {
            if (interval) {
                var video = document.getElementById('video-player');
                var start = interval.start;
                var end = interval.end;

                // Event to continuously monitor the time
                video.ontimeupdate = function() {
                    if (video.currentTime < start || video.currentTime > end) {
                        video.currentTime = start; // Reset to start of the interval if out of bounds
                    }
                };

                video.currentTime = start; // Start at the beginning of the interval
                video.play();

                // Update the range display text
                return `Selected Interval: ${start.toFixed(1)}s - ${end.toFixed(1)}s`;
            }
            return '';
        }
        """,
        [Input('selected-interval-store', 'data')],
        [Input('video-player', 'n_clicks')]
    )

    # Load EEG, ECG, and video data
    eeg_data_path = os.path.join(experiment_folder, 'eeg_data.csv')
    ecg_data_path = os.path.join(experiment_folder, 'ecg_data.csv')

    # Read the dataframes
    eeg_data_df = pd.read_csv(eeg_data_path)
    ecg_data_df = pd.read_csv(ecg_data_path)

    # Debug print statements to verify the data
    print("EEG Data Sample:")
    print(eeg_data_df.head())
    print("ECG Data Sample:")
    print(ecg_data_df.head())

    converted_video_path = convert_video_to_browser_friendly_format(video_file_path)
    video_file_name = os.path.basename(converted_video_path)

    # Define video URL endpoint
    video_url = f'/video/{video_file_name}'

    # Dropdown options for annotations
    ANNOTATION_OPTIONS = [
    {'label': 'Peak in EEG Activity', 'value': 'Peak in EEG Activity'},
    {'label': 'Dip in HRV', 'value': 'Dip in HRV'},
    {'label': 'High Variability in EEG', 'value': 'High Variability in EEG'},
    {'label': 'Correlation Between HRV and EEG', 'value': 'Correlation Between HRV and EEG'},
    {'label': 'Sudden Dip in EEG Activity', 'value': 'Sudden Dip in EEG Activity'},
    {'label': 'Peak in HRV with Concurrent EEG Increase', 'value': 'Peak in HRV with Concurrent EEG Increase'}
]


    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='eeg-hrv-graph',
                style={'height': '800px'},
                config={'modeBarButtonsToAdd': ['select2d', 'lasso2d']},
            ),
        ], style={'flex': '2', 'padding': '10px', "background": "#EEE", "flex-wrap" : "wrap"}),

        html.Div([
            html.Video(
                id="video-player",
                controls=True,
                src='/video/experiment_video.mp4',
                style={"width": "100%"},
            ),
            dcc.RangeSlider(
                id="video-seek-range-slider",
                min=0,
                max=total_recording_time if total_recording_time else 0,
                step=0.1,
                value=[0, total_recording_time if total_recording_time else 0],
                marks={
                    0: {"label": "0", "style": {"color": "black", "font-family": "Arial"}},
                    total_recording_time if total_recording_time else 0: {
                        "label": f"{total_recording_time:.1f}" if total_recording_time else "0",
                        "style": {"color": "black", "font-family": "Arial"}}
                },
                className="custom-slider",
            ),
            html.Div(
                id="range-slider-value-display",
                style={"textAlign": "center", "marginTop": "-10px", "font-family": "Arial", "font-weight": "bold",
                       'font-size': '13px'}
            ),
            html.Div([
                html.Div([
                daq.ColorPicker(
                    id='annotation-color-picker',
                    label='Select Annotation Color',
                    value={'hex': '#2E91E5'},  # Default color (blue)
                    style={'margin-bottom': '10px', "font-family": "Arial", "font-weight": "bold"}
                ),
                html.Div([
                html.P(
                    "Select your desired annotation range using the slider above, then enter your annotation below and click 'Add Annotation'",
                    style={"fontSize": "14px", "color": "#OOO", "marginBottom": "5px", "display": "block",
                           "width": "100%", "font-family": "Arial", "font-weight": "bold"}
                ),
                dcc.Dropdown(
                    id='annotation-dropdown',
                    options=ANNOTATION_OPTIONS,
                    placeholder='Select an annotation type',
                    style={'width': '100%', "font-family": "Arial", "font-weight": "bold", "font-size": "12px"}
                ),
                html.Button(
                    'Add Annotation',
                    id='add-annotation-button',
                    n_clicks=0,
                    style={
                        "padding": "10px 15px",
                        "borderRadius": "5px",
                        "border": "none",
                        "backgroundColor": "#673AB7",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "transition": "background-color 0.3s",
                        "font-weight": "bold"
                    }),
                ], style= {
                        "width": "100%",
                        "display": "flex",
                        "flex-direction": "column",
                        "flex-wrap": "nowrap",
                        "align-content": "space-between",
                        "justify - content": "flex-start",
                        "row-gap": "10px",
                        "padding": "0px 10px"
                    }
                    )],
                    style={
                        "display" : "flex"
                    }),
                dcc.Loading(
                    id="loading-llm-output",
                    type="default",
                    children=[
                        html.Div(
                            id='llm-annotation-suggestions',
                            style={
                                "marginTop": "20px",
                                "font-size": "13px",
                                "line-height": "16px",
                                "padding": "10px",
                                "border": "1px solid #ccc",
                                "borderRadius": "0px",
                                "white-space": "pre-wrap",
                                "background-color": "#fff"
                            }
                        )
                    ],
                ),
                html.Button(
                    'Add to Plot',
                    id='add-to-plot-button',
                    n_clicks=0,
                    style={
                        "padding": "10px 15px",
                        "borderRadius": "5px",
                        "border": "none",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "font-weight": "bold",
                        "marginTop": "10px",
                        "display": "none"
                    }
                ),
                html.P(
                    "Select the annotation you want to delete and click 'Delete Annotation'",
                    style={"fontSize": "14px", "color": "#OOO", "marginBottom": "5px", "display": "block",
                           "width": "100%", "font-family": "Arial", "font-weight": "bold"}
                ),
                dcc.Dropdown(
                    id='annotation-selector',
                    placeholder='Select annotation to delete',
                    options=[],  # Populated dynamically with annotations
                    style={'width': '100%'}
                ),
                html.Button('Delete Annotation', id='delete-annotation-button',
                        style={"padding": "10px 15px",
                        "borderRadius": "5px",
                        "border": "none",
                        "backgroundColor": "red",
                        "color": "white",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "font-weight": "bold",
                        "marginTop": "10px"}),
                dcc.Store(id='static-annotations-store'),
                dcc.Store(id='annotations-store', data=[]),  # Store for annotations
                dcc.Store(id='selected-interval-store', data=[]),  # Store for selected interval
            ], style={
                "marginTop": "20px",
                "padding": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "backgroundColor": "#fff",
                "font-family": "Arial",
                "fontSize": "14px",
                "color": "#333",
                "row-gap": "10px",
                "display": "flex",
                "flex-direction": "column"
            }),

        ], style={'flex': '1', 'padding': '10px', "background": "#EEE"}),

        # Adding the Query Box for LLM integration
        html.Div([
            html.P(
                "Ask a question about the EEG or HRV data:",
                style={"fontSize": "14px", "color": "#000", "marginBottom": "5px", "display": "block",
                       "font-family": "Arial", "font-weight": "bold", "margin-bottom": "10px"}
            ),
            dcc.Input(
                id='query-input',
                type='text',
                placeholder='Enter your question here',
                style={
                    "padding": "10px",
                    "borderRadius": "5px",
                    "border": "1px solid #ccc",
                    "width": "70%",
                    "fontSize": "14px",
                    "marginRight": "10px"
                }
            ),
            html.Button(
                'Submit Query',
                id='submit-query-button',
                n_clicks=0,
                style={
                    "padding": "10px 15px",
                    "borderRadius": "5px",
                    "border": "none",
                    "backgroundColor": "rgb(255 201 102)",
                    "color": "black",
                    "cursor": "pointer",
                    "fontSize": "14px",
                    "transition": "background-color 0.3s",
                    "font-weight": "bold"
                }
            ),
            dcc.Loading(
                id="loading-llm-output-data-query",
                type="default",
                children=[
                    html.Pre(
                        id='query-response',
                        style={
                            "marginTop": "20px",
                            "padding": "10px",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px",
                            "backgroundColor": "#fff",
                            "font-family": "Arial",
                            "fontSize": "14px",
                            "color": "#333"
                        }
                    )]
            )
        ], style={
            "marginTop": "10px",
            "padding": "10px",
            "backgroundColor": "#EEE",
            "width" : "calc(100% - 20px)"
        }),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'width': '100%',
        'justifyContent': 'space-between',
        'flex-wrap': 'wrap',
        'marginTop': '20px'  # Adjust the value as needed
        })

    print("Serving video from:", converted_video_path)

    @app.callback(
        Output('selected-interval-store', 'data'),
        [Input('eeg-hrv-graph', 'selectedData'),
         Input('video-seek-range-slider', 'value')],
        [State('selected-interval-store', 'data')]
    )
    def update_selected_interval_store(selected_data, slider_value, current_interval):
        if selected_data and 'range' in selected_data:
            # Extract the range from the graph selection
            x_range = selected_data['range']['x']
            start_value, end_value = x_range[0], x_range[1]
            return {'start': start_value, 'end': end_value}

        elif slider_value:
            # If no graph selection, use slider values
            start_value, end_value = slider_value[0], slider_value[1]
            return {'start': start_value, 'end': end_value}

        # Default to the current interval if no input is provided
        return current_interval if current_interval else {'start': 0, 'end': 0}

    @app.callback(
        Output('range-slider-value-display', 'children'),
        [Input('selected-interval-store', 'data')]
    )
    def update_selected_interval_display(selected_interval):
        if selected_interval and 'start' in selected_interval and 'end' in selected_interval:
            start, end = selected_interval['start'], selected_interval['end']
            return f"Selected Interval: {start:.1f}s - {end:.1f}s"

        return "Selected Interval: 0.0s - 0.0s"

    @app.callback(
        Output('video-player', 'src'),
        Input('video-seek-range-slider', 'value')
    )
    def update_video_src(seek_time):
        return f"{video_url}#t={seek_time}"

    def save_annotations_to_csv(annotation):
        eeg_data_path = os.path.join(experiment_folder, 'eeg_data.csv')
        ecg_data_path = os.path.join(experiment_folder, 'ecg_data.csv')

        # Load the EEG and HRV data
        eeg_data_df = pd.read_csv(eeg_data_path)
        ecg_data_df = pd.read_csv(ecg_data_path)

        # Add annotation to EEG DataFrame
        annotation_row = {
            'Timestamp': (annotation['start'] + annotation['end']) / 2,
            'Annotation': annotation['text']
        }

        # Append the annotation to both DataFrames
        if 'Annotation' not in eeg_data_df.columns:
            eeg_data_df['Annotation'] = None

        if 'Annotation' not in ecg_data_df.columns:
            ecg_data_df['Annotation'] = None

        eeg_data_df.loc[(eeg_data_df['Internal Clock'] >= annotation['start']) &
                        (eeg_data_df['Internal Clock'] <= annotation['end']), 'Annotation'] = annotation['text']

        ecg_data_df.loc[(ecg_data_df['Timestamp'] >= annotation['start']) &
                        (ecg_data_df['Timestamp'] <= annotation['end']), 'Annotation'] = annotation['text']

        # Save the updated DataFrames back to CSV
        eeg_data_df.to_csv(eeg_data_path, index=False)
        ecg_data_df.to_csv(ecg_data_path, index=False)

        print(f"Annotation saved to EEG and ECG CSV files for interval {annotation['start']} to {annotation['end']}")

    @app.callback(
        Output('static-annotations-store', 'data'),
        [Input('add-annotation-button', 'n_clicks')],
        [State('selected-interval-store', 'data'),
         State('annotation-dropdown', 'value'),
         State('static-annotations-store', 'data'),
         State('annotation-color-picker', 'value')]
    )
    def add_static_annotation(n_clicks, selected_interval, annotation_value, current_annotations, color):
        if not n_clicks or not selected_interval or not annotation_value:
            raise PreventUpdate

        current_annotations = current_annotations or []
        start, end = selected_interval.get('start', 0), selected_interval.get('end', 0)

        if start != end:
            new_annotation = {'start': start, 'end': end, 'text': annotation_value, 'color': color['hex']}
            current_annotations.append(new_annotation)

        return current_annotations

    @app.callback(
        Output('annotations-store', 'data'),
        [Input('static-annotations-store', 'data')]
    )
    def update_combined_annotations(static_annotations):
        static_annotations = static_annotations or []
        combined_annotations = static_annotations
        print(f"Updated Combined Annotations: {combined_annotations}")
        return combined_annotations


    @app.callback(
        Output('annotation-selector', 'options'),
        [Input('annotations-store', 'data')]
    )
    def populate_annotation_selector(annotations_store):
        if not annotations_store:
            return []
        return [{'label': f"{annotation['text']} ({annotation['start']}s - {annotation['end']}s)",
                 'value': i} for i, annotation in enumerate(annotations_store)]

    @app.callback(
        Output('annotations-store', 'data', allow_duplicate=True),
        [Input('delete-annotation-button', 'n_clicks')],
        [State('annotation-selector', 'value'),
         State('annotations-store', 'data')],
        prevent_initial_call=True
    )
    def delete_annotation(n_clicks, selected_annotation, annotations_store):
        if not n_clicks or selected_annotation is None or not annotations_store:
            raise PreventUpdate

        # Remove the selected annotation
        annotations_store.pop(selected_annotation)
        print(f"Deleted Annotation Index: {selected_annotation}")
        return annotations_store

    # Adjusted update_graph function to rely solely on graph selection for user-defined annotations while preserving slider for visualization
    @app.callback(
        Output('eeg-hrv-graph', 'figure'),
        [Input('annotations-store', 'data'),  # Ensure it listens to changes in the annotations-store
         Input('video-seek-range-slider', 'value')],
        [State('annotation-dropdown', 'value'),
         State('selected-interval-store', 'data')]
    )
    def update_graph(annotations_store, value, annotation_text, selected_interval):
        start_value, end_value = value
        annotations_store = annotations_store or []

        # Save all annotations to the CSV file
        for annotation in annotations_store:
            if annotation:  # Ensure valid annotations
                save_annotations_to_csv(annotation)

        # Debugging Statements to ensure correct callback triggers
        print("Callback triggered.")
        print(f"Slider Value (start, end): {value}")
        print(f"Annotation Text: {annotation_text}")

        # Filter EEG and HRV data based on selected range
        eeg_filtered = eeg_data[
            (eeg_data['Internal Clock'] >= start_value) &
            (eeg_data['Internal Clock'] <= end_value)
            ]
        hrv_filtered = [val for i, val in enumerate(hrv_data) if start_value <= hrv_timestamps[i] <= end_value]
        hrv_filtered_timestamps = [t for t in hrv_timestamps if start_value <= t <= end_value]

        # Debug: Log filtered data size
        print(f"EEG Data Points Filtered: {len(eeg_filtered)}")
        print(f"HRV Data Points Filtered: {len(hrv_filtered)}")

        # Create subplots for EEG and HRV
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.15,
        )

        # Plot EEG data
        for column in eeg_filtered.columns[1:]:
            fig.add_trace(go.Scatter(
                x=eeg_filtered['Internal Clock'],
                y=eeg_filtered[column],
                mode='lines',
                name=f'EEG Channel {column}'
            ), row=1, col=1)

        # Plot HRV data
        fig.add_trace(go.Scatter(
            x=hrv_filtered_timestamps,
            y=hrv_filtered,
            mode='lines',
            line=dict(width=2, color='red'),
            name='HRV'
        ), row=2, col=1)

        # Handle Stimuli Rectangles
        stimuli_colors = {
            "Visual": "rgba(255, 165, 0, 0.6)",  # Orange
            "Auditory": "rgba(128, 0, 128, 0.6)",  # Purple
            "Cognitive": "rgba(0, 255, 0, 0.6)"  # Green
        }

        if 'Stimulus' in eeg_filtered.columns:
            for stimulus_type in eeg_filtered['Stimulus'].unique():
                if stimulus_type != "None":
                    stimulus_rows = eeg_filtered[eeg_filtered['Stimulus'] == stimulus_type]
                    if not stimulus_rows.empty:
                        start_time = None
                        gap_threshold = 0.4
                        for idx in range(len(stimulus_rows)):
                            current_time = stimulus_rows.iloc[idx]['Internal Clock']
                            if start_time is None:
                                start_time = current_time
                            is_last_row = idx == len(stimulus_rows) - 1
                            next_time = stimulus_rows.iloc[idx + 1]['Internal Clock'] if not is_last_row else None
                            if is_last_row or (next_time and next_time - current_time > gap_threshold):
                                end_time = current_time
                                color = stimuli_colors.get(stimulus_type, "rgba(128, 128, 128, 0.3)")
                                fig.add_vrect(
                                    x0=start_time,
                                    x1=end_time,
                                    fillcolor=color,
                                    opacity=0.3,
                                    layer="below",
                                    line_width=0,
                                    y0=0,
                                    y1=1,
                                    annotation_text=stimulus_type,
                                    annotation_position="top left"
                                )
                                start_time = None

        # Add legend items for stimuli colors
        for stimulus_type, color in stimuli_colors.items():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"{stimulus_type} Stimulus"
            ))

        # Add annotations from the store
        for annotation in annotations_store or []:
            label = annotation["text"][:10]  # Shortened label for vrect
            full_text = annotation.get("full_text", annotation["text"])  # Full text for hover

            # Add vertical rectangle with short label
            fig.add_vrect(
                x0=annotation["start"],
                x1=annotation["end"],
                fillcolor=annotation.get('color', 'gray'),  # Default to gray if no color is set
                opacity=0.3,
                line_width=0,
            )

            # Add hover annotation for the full text
            fig.add_annotation(
                x=(annotation["start"] + annotation["end"]) / 2,  # Center of vrect
                y=max(eeg_data),  # Adjust as per your graph
                text=label,  # Use short label here
                showarrow=False,
                font=dict(size=10, color="black"),
                align="center",
                yshift=20,  # Offset to prevent overlapping
                hovertext=full_text,  # Detailed text on hover
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            )

        print(f"Annotations Store: {annotations_store}")

        # Update Layout
        fig.update_layout(
            height=700,
            title_text='EEG and HRV Data with Stimuli',
            xaxis_title='Time (seconds)',
            yaxis_title='EEG Data',
            yaxis2_title='HRV Data',
            template='plotly_white',
            legend=dict(
                orientation="h",
                x=0,
                y=-0.2,
                xanchor="left",
                yanchor="top"
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # Update x-axis range based on slider
        fig.update_xaxes(range=[start_value, end_value])

        return fig

    app.run_server(debug=True, use_reloader=False)

# Create a frame for buttons in the left_frame
button_frame = tk.Frame(left_frame)
button_frame.pack(side=tk.TOP, pady=10)  # Place at the top with some vertical padding

start_button = tk.Button(button_frame, text="Start Experiment", bg="green", fg="white", command=start_experiment)
start_button.pack(side=tk.LEFT, padx=5, pady=5)

stop_button = tk.Button(button_frame, text="Stop Experiment", bg="red", fg="white", command=stop_experiment)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

visualize_button = tk.Button(button_frame, text="Visualize Data", command=visualize_data_with_dash)
visualize_button.pack(side=tk.LEFT, padx=5)

visual_stimulus_button = tk.Button(button_frame, text="Visual Stimulus", command= lambda: trigger_stimulus("Visual"))
visual_stimulus_button.pack(side=tk.LEFT, padx=5)

auditory_stimulus_button = tk.Button(button_frame, text="Auditory Stimulus", command= lambda: trigger_stimulus("Auditory"))
auditory_stimulus_button.pack(side=tk.LEFT, padx=5)

cognitive_stimulus_button = tk.Button(button_frame, text="Cognitive Stimulus", command= lambda: trigger_stimulus("Cognitive"))
cognitive_stimulus_button.pack(side=tk.LEFT, padx=5)

# EEG channel selection checkboxes in the left_frame
checkbox_frame = tk.Frame(left_frame)
checkbox_frame.pack(pady=10)

for region, channels in {
    "Frontal": ["AF3", "AF4", "F3", "F4", "F7", "F8"],
    "Temporal": ["T7", "T8"],
    "Parietal": ["P7", "P8"],
    "Occipital": ["O1", "O2"],
    "Central": ["C3", "C4"],
}.items():
    tk.Label(checkbox_frame, text=region).pack(side="left", padx=5)
    for ch in channels:
        tk.Checkbutton(checkbox_frame, text=ch, variable=selected_channels[ch]).pack(side="left")

# Matplotlib setup for plotting
fig, (ax_eeg, ax_hrv) = plt.subplots(2, 1, figsize=(10, 8))
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack()

ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)

# Initialize video feed on load
initialize_video()

root.mainloop()