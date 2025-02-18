import pandas as pd
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from flask import send_file, jsonify
import os
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash
import cv2
import subprocess


# Global variables
eeg_data = {ch: [] for ch in ["AF3", "AF4", "F3", "F4", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2", "C3", "C4"]}
hrv_data = []
timestamps = []
hrv_timestamps = []  # Independent timestamps for HRV
stimulus_active = False  # Flag to indicate active stimulus
stimulus_effect_duration = 2  # Duration of stimulus effect in seconds
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


def convert_video_to_browser_friendly_format(input_video_path):
    # Define the output video path by appending '_converted' to the filename
    output_video_path = input_video_path.replace(".mp4", "_converted.mp4")

    # Check if the converted video already exists
    if os.path.exists(output_video_path):
        #print(f"Converted video already exists: {output_video_path}")
        return output_video_path  # Skip conversion if the file already exists

    # Provide the full path to ffmpeg executable
    ffmpeg_path = shutil.which("ffmpeg")  # Automatically finds ffmpeg
    if not ffmpeg_path:
        raise FileNotFoundError("FFmpeg not found. Install it or provide its path.")


    # Command to convert the video using ffmpeg
    command = [
        ffmpeg_path,
        '-i', input_video_path,
        '-r', '10',  # Set the output frame rate to 10 fps
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
        #print(f"Conversion successful: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

    return output_video_path

def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def get_video_duration(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open the video file.")
        return 0  # Fallback to 0 if the file can't be opened
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0  # Avoid division by zero
    video.release()
    return duration

def create_dash_app(eeg_file_path, hrv_file_path, video_file_path):
    app = dash.Dash(__name__)

    hrv_data_df = pd.read_csv(hrv_file_path)

    global hrv_timestamps
    if 'Timestamp' in hrv_data_df.columns:
        hrv_data_df['Timestamp'] -= hrv_data_df['Timestamp'].min()
        hrv_timestamps = hrv_data_df['Timestamp'].tolist()
    else:
        raise KeyError("HRV data is missing the 'Timestamp' column.")

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

    # Calculate total recording time from the video
    global total_recording_time
    total_recording_time = get_video_duration(video_file_path)

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

        # Load the EEG and HRV data
        eeg_data_df = pd.read_csv(eeg_file_path)
        ecg_data_df = pd.read_csv(hrv_file_path)

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

        eeg_data_df.columns = eeg_data_df.columns.str.strip()
        if 'Internal Clock' not in eeg_data_df.columns:
            raise KeyError(f"'Internal Clock' column not found in EEG data. Available columns: {eeg_data.columns}")

        eeg_data_df.loc[(eeg_data_df['Internal Clock'] >= annotation['start']) &
                        (eeg_data_df['Internal Clock'] <= annotation['end']), 'Annotation'] = annotation['text']

        ecg_data_df.loc[(ecg_data_df['Timestamp'] >= annotation['start']) &
                        (ecg_data_df['Timestamp'] <= annotation['end']), 'Annotation'] = annotation['text']

        # Save the updated DataFrames back to CSV
        eeg_data_df.to_csv(eeg_file_path, index=False)
        ecg_data_df.to_csv(hrv_file_path, index=False)

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

        global eeg_file_path, hrv_file_path

        # Read the dataframes
        eeg_data = pd.read_csv(eeg_file_path)
        hrv_data = pd.read_csv(hrv_file_path)

        start_value, end_value = value
        annotations_store = annotations_store or []

        # Save all annotations to the CSV file
        for annotation in annotations_store:
            if annotation:  # Ensure valid annotations
                save_annotations_to_csv(annotation)

        # Filter EEG and HRV data based on selected range
        eeg_filtered = eeg_data[
            (eeg_data['Internal Clock'] >= start_value) &
            (eeg_data['Internal Clock'] <= end_value)
            ]
        hrv_filtered = hrv_data_df[
            (hrv_data_df['Timestamp'] >= start_value) & (hrv_data_df['Timestamp'] <= end_value)
            ]
        hrv_filtered_timestamps = [t for t in hrv_timestamps if start_value <= t <= end_value]

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
        hrv_y_values = hrv_filtered['ECG'].values
        fig.add_trace(go.Scatter(
            x=hrv_filtered_timestamps,
            y=hrv_y_values,
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

    return app

if __name__ == "__main__":
    eeg_file_path = "data/experiment_20250127_110148/eeg_data.csv"
    hrv_file_path = "data/experiment_20250127_110148/ecg_data.csv"
    video_file_path = "data\experiment_20250127_110148\experiment_video.mp4"

    eeg_data = pd.read_csv(eeg_file_path)
    hrv_data = pd.read_csv(hrv_file_path)

    app = create_dash_app(eeg_file_path, hrv_file_path, video_file_path)
    app.run_server(port=8052, use_reloader=True)
