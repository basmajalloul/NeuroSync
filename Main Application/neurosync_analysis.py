import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy.signal import find_peaks

from scipy.stats import pearsonr

# Load EEG and HRV Data
eeg_data_path = "data/eeg_data.csv"
hrv_data_path = "data/ecg_data.csv"

eeg_data = pd.read_csv(eeg_data_path)
hrv_data = pd.read_csv(hrv_data_path)

# print(eeg_data["Stimulus"].unique())
# print(eeg_data["Stimulus"].dtype)

# Filter EEG channels (exclude non-EEG columns)
eeg_channels = [col for col in eeg_data.columns if col not in ["Timestamp", "Stimulus", "Internal Clock", "Annotation"]]

# Sampling rate for EEG in Hz (adjust based on your data)
EEG_SAMPLING_RATE = 256

# Compute EEG Band Power
def compute_eeg_band_power(data, sampling_rate):
    freq, psd = welch(data, fs=sampling_rate, nperseg=1024)
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 50),
    }
    band_power = {}
    for band, (low, high) in bands.items():
        band_power[band] = trapezoid(psd[(freq >= low) & (freq < high)], freq[(freq >= low) & (freq < high)])
    return band_power

eeg_band_powers = {channel: compute_eeg_band_power(eeg_data[channel], EEG_SAMPLING_RATE) for channel in eeg_channels}

# Compute RR Intervals and HRV Metrics
def calculate_hrv_metrics(rr_intervals):
    """
    Calculate HRV metrics from RR intervals.
    Parameters:
    - rr_intervals: List or array of RR intervals (in ms).
    Returns:
    - metrics: Dictionary with Mean HR, RMSSD, and SDNN.
    """
    if len(rr_intervals) == 0:
        # print("No RR intervals to calculate metrics.")
        return {"MeanHR": np.nan, "RMSSD": np.nan, "SDNN": np.nan}

    # Convert RR intervals to seconds for HR calculation
    rr_intervals_sec = np.array(rr_intervals) / 1000.0

    # Mean Heart Rate (HR)
    mean_hr = 60.0 / np.mean(rr_intervals_sec)

    # RMSSD: Root Mean Square of Successive Differences
    diff_rr = np.diff(rr_intervals_sec)
    rmssd = np.sqrt(np.mean(diff_rr**2)) * 1000.0  # Convert back to ms

    # SDNN: Standard Deviation of NN Intervals
    sdnn = np.std(rr_intervals_sec) * 1000.0  # Convert back to ms

    return {"MeanHR": mean_hr, "RMSSD": rmssd, "SDNN": sdnn}

def calculate_rr_intervals(csv_path, sampling_rate):
    """
    Process raw ECG signal from a CSV file and calculate HRV metrics.
    Parameters:
    - csv_path: Path to the CSV file containing ECG data.
    - sampling_rate: Sampling rate of ECG signal in Hz.
    """
    # Load the CSV using pandas
    ecg_df = pd.read_csv(csv_path)

    # Ensure the column "Amplitude" exists and extract the signal
    if "ECG" not in ecg_df.columns:
        # print("The 'Amplitude' column is missing in the CSV.")
        return

    ecg_signal = ecg_df["ECG"].values
    #print(f"Loaded ECG signal with {len(ecg_signal)} samples.")

    # Step 1: Detect R-peaks
    peaks, _ = find_peaks(ecg_signal, distance=sampling_rate * 0.5)  # Adjust threshold if needed
    #print(f"Detected R-peaks: {peaks}")

    # Step 2: Calculate RR intervals (in ms)
    rr_intervals = np.diff(peaks) / sampling_rate * 1000.0
    #print(f"RR intervals (ms): {rr_intervals}")

    # Filter RR intervals for physiological plausibility (200-3000 ms)
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    #print(f"Filtered RR intervals (ms): {rr_intervals}")

    return rr_intervals

rr_intervals = calculate_rr_intervals(hrv_data_path, 10)

hrv_metrics = calculate_hrv_metrics(rr_intervals)

def extract_event_segments(eeg_data, event_label, pre_event=1, post_event=1):
    event_indices = eeg_data[eeg_data["Stimulus"] == event_label].index
    # print(f"Event indices found for '{event_label}': {event_indices}")

    if len(event_indices) == 0:
        # print(f"No events found for '{event_label}' in the Stimulus column.")
        return []

    segments = []
    for idx in event_indices:
        start_idx = max(idx - int(pre_event * EEG_SAMPLING_RATE), 0)
        end_idx = min(idx + int(post_event * EEG_SAMPLING_RATE), len(eeg_data) - 1)
        segment = eeg_data.iloc[start_idx:end_idx]
        segments.append(segment)

    return segments


# Dash App
app = dash.Dash(__name__)

app.layout = (

    html.Div([

        html.Div([
            html.H1("EEG and HRV Metrics Dashboard", style={'textAlign': 'center', "font-family": "Arial"}
        ),

        html.Div([
            html.H2("EEG Band Power Table"),
            html.Div([
                    html.Div(id="eeg-band-table", style={"margin": "0px"}),
                    html.Div(id="eeg-band-alerts", style={"marginTop": "0px"})
            ],
            className="insights-body"),
        ],
        style={'padding': "20px", "background": "#fff"},
        className="insights"),

        html.Div([
            html.H2("HRV Metrics"),
            html.Div([
                html.Div(id="hrv-metrics-cards"),
                html.Div(id="hrv-metrics-alerts")
            ],
            className="insights-body"),
        ],
        className="insights"),

        # Section for Statistical Summaries
        html.Div([
            html.H2("Statistical Summaries"),
            html.Div([
                html.Div(id="statistical-summaries"),
                html.Div(id="statistical-insights")
            ], className="insights-raw")
        ], className="insights"),

        html.Div([
            # EEG Band Power Heatmap

            html.Div([
                html.Div([
                    html.H2("EEG Band Power Heatmap"),
                    dcc.Graph(id="eeg-band-heatmap"),
                ], style={'margin': '20px'}),

                # Correlation Analysis Heatmap
                html.Div([
                    html.H2("Correlation Analysis"),
                    html.Div(id="correlation-heatmap"),
                ]),
                ]),

            # RR Interval Insights
            html.Div([
                html.H2("RR Interval Insights"),
                dcc.Graph(id="rr-interval-histogram"),
                dcc.Graph(id="rr-interval-smooth"),
            ]),

        ], className="insights-visual"),

        html.Div([
            html.H2("Event-Based Analysis"),
            dcc.Dropdown(
                id="event-selector",
                options=[{"label": "All", "value": "All"}] + [{"label": str(event), "value": str(event)} for event in
                                                              eeg_data["Stimulus"].dropna().unique()],
                value="All",  # Set "All" as the default
                placeholder="Select an event",
            ),
            dcc.Dropdown(
                id="channel-filter",
                options=[{"label": channel, "value": channel} for channel in eeg_channels],
                value=eeg_channels,
                multi=True,
                placeholder="Select EEG Channels",
                style={"margin": "20px"}
            ),
            dcc.RangeSlider(
                id="time-slider",
                min=eeg_data["Timestamp"].min(),
                max=eeg_data["Timestamp"].max(),
                step=1,
                value=[eeg_data["Timestamp"].min(), eeg_data["Timestamp"].max()],
                marks={int(ts): str(int(ts)) for ts in
                       np.linspace(eeg_data["Timestamp"].min(), eeg_data["Timestamp"].max(), 5)}
            ),
            dcc.Graph(id="eeg-plot"),
            dcc.Graph(id="hrv-plot"),
        ],
        className="insights"),

        # Trends Section

        html.Div([
            html.H2("Trends Analysis"),
            html.Div([
                html.Label("Select EEG Channels:"),
                dcc.Dropdown(
                    id="trends-eeg-power-bands",
                    options=[
                        {"label": "Delta", "value": "Delta"},
                        {"label": "Theta", "value": "Theta"},
                        {"label": "Alpha", "value": "Alpha"},
                        {"label": "Beta", "value": "Beta"},
                        {"label": "Gamma", "value": "Gamma"},
                    ],
                    multi=True,
                    value=["Delta", "Theta", "Alpha"],  # Default selection
                    placeholder="Select EEG Power Bands",
                ),
                html.Label("Select Aggregation Interval (seconds):"),
                dcc.Dropdown(
                    id="aggregation-interval",
                    options=[
                        {"label": "1 second", "value": 1},
                        {"label": "5 seconds", "value": 5},
                        {"label": "10 seconds", "value": 10},
                        {"label": "30 seconds", "value": 30},
                        {"label": "1 minute", "value": 60}
                    ],
                    value=1,  # Default interval
                ),
            ]),
            dcc.Graph(id="eeg-trends-graph"),
            dcc.Graph(id="hrv-trends-graph"),
        ],
        className='insights'),

        html.Div([
            html.Div(id="synchronization-validation")
        ],
         className="insights-sync"),

    ],
    style={'font-family': 'Arial', "background-color" : "#eee", "padding" : "30px"})

]))

@app.callback(
    Output("eeg-band-table", "children"),
    Input("correlation-heatmap", "id")
)
def update_eeg_band_table(trigger):
    header = ["Channel", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
    rows = [
        [channel] + [f"{powers.get(band, 0):.2f}" for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]]
        for channel, powers in eeg_band_powers.items()
    ]
    return html.Table(
        # Table Header
        [html.Tr([html.Th(col) for col in header])] +
        # Table Rows
        [html.Tr([html.Td(cell) for cell in row]) for row in rows],
        style={"width": "100%", "border": "1px solid black", "borderCollapse": "collapse"}
    )

@app.callback(
    Output("hrv-metrics-cards", "children"),
    Input("correlation-heatmap", "id")
)
def update_hrv_metrics_cards(trigger):
    return html.Div([
        html.Div([
            html.H4(f"Mean HR: {hrv_metrics['MeanHR']:.2f} BPM"),
            html.P("Mean Heart Rate"),
        ], style={"padding": "10px", "border": "1px solid black", "margin": "5px", "display": "inline-block"}),
        html.Div([
            html.H4(f"RMSSD: {hrv_metrics['RMSSD']:.2f} ms"),
            html.P("Root Mean Square of Successive Differences"),
        ], style={"padding": "10px", "border": "1px solid black", "margin": "5px", "display": "inline-block"}),
        html.Div([
            html.H4(f"SDNN: {hrv_metrics['SDNN']:.2f} ms"),
            html.P("Standard Deviation of NN Intervals"),
        ], style={"padding": "10px", "border": "1px solid black", "margin": "5px", "display": "inline-block"}),
    ])

@app.callback(
    Output("eeg-band-heatmap", "figure"),
    Input("correlation-heatmap", "id")
)
def update_band_heatmap(trigger):
    band_labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    heatmap_data = [
        [powers.get(band, 0) for band in band_labels]
        for powers in eeg_band_powers.values()
    ]
    return {
        "data": [go.Heatmap(z=heatmap_data, x=band_labels, y=list(eeg_band_powers.keys()))],
        "layout": {
            "title": "EEG Band Power Heatmap",
            "xaxis": {"title": "EEG Bands"},
            "yaxis": {"title": "Channels"},
        },
    }

@app.callback(
    Output("rr-interval-histogram", "figure"),
    Input("correlation-heatmap", "id")
)
def update_rr_histogram(trigger):
    return {
        "data": [go.Histogram(x=rr_intervals, nbinsx=50)],
        "layout": {
            "title": "RR Interval Distribution",
            "xaxis": {"title": "RR Interval (ms)"},
            "yaxis": {"title": "Frequency"},
        },
    }

@app.callback(
    Output("rr-interval-smooth", "figure"),
    Input("correlation-heatmap", "id")
)
def update_rr_smooth(trigger):
    window_size = 10
    smoothed_rr = pd.Series(rr_intervals).rolling(window=window_size).mean().dropna()
    return {
        "data": [go.Scatter(x=list(range(len(smoothed_rr))), y=smoothed_rr)],
        "layout": {
            "title": "Smoothed RR Intervals",
            "xaxis": {"title": "Time"},
            "yaxis": {"title": "Smoothed RR Interval (ms)"},
        },
    }

## Compute and Clean Combined Data for Correlation Analysis
eeg_band_powers_df = pd.DataFrame.from_dict(eeg_band_powers, orient='index')
hrv_metrics_df = pd.DataFrame(hrv_metrics, index=["HRV_Metrics"])

combined_data = pd.concat([eeg_data.iloc[:, 1:], hrv_data[['ECG']]], axis=1)

# Drop NaN values to avoid errors in correlation computation
combined_data = combined_data.dropna()
# Drop non-relevant columns like 'Stimulus', 'Annotations', and 'Internal Clock'
numeric_data = combined_data.drop(columns=['Stimulus', 'Annotation', 'Internal Clock'], errors='ignore')

# Ensure only numeric columns are selected
numeric_data = numeric_data.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()
# Display the correlation matrix
# print("Correlation Matrix:")
# print(correlation_matrix)

@app.callback(
    Output("correlation-heatmap", "children"),
    Input("eeg-band-table", "children")  # Example trigger, adjust as needed
)
def update_correlation_heatmap(eeg_data):
    # Save the heatmap to a BytesIO buffer
    buffer = io.BytesIO()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between EEG Channels and Raw ECG Values")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    # Return the image as an HTML Img element
    return html.Img(
        src=f"data:image/png;base64,{encoded_image}",
        style={"width": "100%", "height": "auto"}
    )


@app.callback(
    [Output("eeg-plot", "figure"), Output("hrv-plot", "figure")],
    [Input("event-selector", "value"),
     Input("channel-filter", "value"),
     Input("time-slider", "value")]
)
def update_event_plots(selected_event, selected_channels, time_range):
    if selected_event == "All":
        # Use the entire signal when "All" is selected
        eeg_segment = eeg_data[(eeg_data["Timestamp"] >= time_range[0]) & (eeg_data["Timestamp"] <= time_range[1])]
        hrv_segment = hrv_data[(hrv_data["Timestamp"] >= time_range[0]) & (hrv_data["Timestamp"] <= time_range[1])]
    else:
        # Extract EEG segments for the specific event
        eeg_segments = extract_event_segments(eeg_data, selected_event)
        if not eeg_segments:
            # print(f"No EEG segments found for the event '{selected_event}'.")
            return go.Figure(), go.Figure()

        # Select the first EEG segment
        eeg_segment = eeg_segments[0]
        eeg_segment = eeg_segment[
            (eeg_segment["Timestamp"] >= time_range[0]) & (eeg_segment["Timestamp"] <= time_range[1])]

        # Extract HRV data for the same time range
        start_time = eeg_segment["Timestamp"].iloc[0]
        end_time = eeg_segment["Timestamp"].iloc[-1]
        hrv_segment = hrv_data[(hrv_data["Timestamp"] >= start_time) & (hrv_data["Timestamp"] <= end_time)]

    # Create EEG figure
    eeg_fig = go.Figure()
    for channel in selected_channels:
        if channel in eeg_segment.columns:
            eeg_fig.add_trace(go.Scatter(
                x=eeg_segment["Timestamp"],
                y=eeg_segment[channel],
                mode="lines",
                name=f"EEG - {channel}"
            ))

    eeg_fig.update_layout(
        title=f"EEG Analysis for '{selected_event}'",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend_title="EEG Channels"
    )

    # Create HRV figure
    hrv_fig = go.Figure()
    if not hrv_segment.empty:
        hrv_fig.add_trace(go.Scatter(
            x=hrv_segment["Timestamp"],
            y=hrv_segment["ECG"],
            mode="lines",
            name="HRV (ECG)"
        ))

    hrv_fig.update_layout(
        title=f"HRV Analysis for '{selected_event}'",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend_title="HRV Metrics"
    )

    return eeg_fig, hrv_fig

@app.callback(
    [Output('eeg-trends-graph', 'figure'),
     Output('hrv-trends-graph', 'figure')],
    [
        Input('trends-eeg-power-bands', 'value'),
        Input('aggregation-interval', 'value'),
    ]
)
def update_trends(selected_bands, aggregation_interval):
    global eeg_data, hrv_data
    try:
        # print("Selected EEG Bands:", selected_bands)
        # print("Selected Aggregation Interval:", aggregation_interval)

        # Ensure bands and aggregation interval are selected
        if not selected_bands or not aggregation_interval:
            # print("No bands or aggregation interval selected!")
            return go.Figure(), go.Figure()

        # Define EEG frequency bands
        band_definitions = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 45),
        }

        # Ensure the selected bands exist in definitions
        if any(band not in band_definitions for band in selected_bands):
            # print("Invalid band selection!")
            return go.Figure(), go.Figure()

        # --- Process EEG Data ---
        # print("Processing EEG Data...")
        eeg_data = eeg_data.copy()  # Replace with your actual raw EEG variable
        eeg_sampling_rate = 256  # Replace with actual EEG sampling rate
        eeg_data['Time'] = (eeg_data.index / eeg_sampling_rate).astype(float)

        # Calculate EEG Power Bands
        eeg_power_data = pd.DataFrame()
        eeg_power_data['Time'] = eeg_data['Time']
        for band in selected_bands:
            low, high = band_definitions[band]
            # Convert frequency range to indices
            n = eeg_data.shape[0]  # Number of samples
            freq_resolution = eeg_sampling_rate / n
            low_idx = int(low / freq_resolution)
            high_idx = int(high / freq_resolution)

            band_power = eeg_data.select_dtypes(include=[np.number]).apply(
                lambda row: np.mean(np.square(np.abs(np.fft.fft(row.values)[low_idx:high_idx]))),
                axis=1
            )
            eeg_power_data[band] = band_power

        # Aggregate EEG Power Bands
        eeg_power_data['TimeBin'] = eeg_power_data['Time'] // aggregation_interval * aggregation_interval
        eeg_power_aggregated = eeg_power_data.groupby('TimeBin')[selected_bands].mean().reset_index()

        # print("EEG Power Aggregated:")
        print(eeg_power_aggregated.head())

        # --- Process HRV Data ---
        # print("Processing HRV Data...")
        hrv_data = hrv_data.copy()  # Replace with your actual raw HRV variable
        hrv_sampling_rate = 10  # Replace with actual HRV sampling rate
        hrv_data['Time'] = (hrv_data.index / hrv_sampling_rate).astype(float)

        # Aggregate HRV Metrics
        hrv_data['TimeBin'] = hrv_data['Time'] // aggregation_interval * aggregation_interval
        hrv_aggregated = hrv_data.groupby('TimeBin').apply(
            lambda group: pd.Series({
                'MeanHR': group['ECG'].mean(),
                'SDNN': group['ECG'].std()
            })
        ).reset_index()

        # print("HRV Aggregated:")
        # print(hrv_aggregated.head())

        # --- Create EEG Trends Plot ---
        # print("Creating EEG Trends Plot...")
        eeg_fig = go.Figure()
        for band in selected_bands:
            if band in eeg_power_aggregated.columns:
                eeg_fig.add_trace(go.Scatter(
                    x=eeg_power_aggregated['TimeBin'],
                    y=eeg_power_aggregated[band],
                    mode='lines',
                    name=f"EEG - {band}"
                ))

        eeg_fig.update_layout(
            title="EEG Trends",
            xaxis_title="Time (s)",
            yaxis_title="Power",
            legend_title="EEG Bands",
            template="plotly_white"
        )

        # --- Create HRV Trends Plot ---
        # print("Creating HRV Trends Plot...")
        hrv_fig = go.Figure()
        for metric in ['MeanHR', 'SDNN']:
            if metric in hrv_aggregated.columns:
                hrv_fig.add_trace(go.Scatter(
                    x=hrv_aggregated['TimeBin'],
                    y=hrv_aggregated[metric],
                    mode='lines',
                    name=f"HRV - {metric}",
                    line=dict(dash='dot')
                ))

        hrv_fig.update_layout(
            title="HRV Trends",
            xaxis_title="Time (s)",
            yaxis_title="Metric Value",
            legend_title="HRV Metrics",
            template="plotly_white"
        )

        # print("Trends Plots Successfully Created!")
        return eeg_fig, hrv_fig

    except Exception as e:
        # print("Error in update_trends:", e)
        return go.Figure(), go.Figure()

# Callback to generate statistical summaries
@app.callback(
    Output("statistical-summaries", "children"),
    [Input("statistical-summaries", "id")]
)
def update_statistical_summaries(trigger):
    # Calculate EEG statistics
    numeric_eeg = eeg_data.drop(columns=['Timestamp', 'Internal Clock'], errors='ignore')
    eeg_stats = numeric_eeg.describe().T[['mean', 'std', 'min', '50%', 'max']]
    eeg_stats.rename(columns={"50%": "median"}, inplace=True)

    # Calculate HRV statistics
    numeric_hrv = hrv_data.drop(columns=['Time','Timestamp', 'TimeBin', 'Internal Clock'], errors='ignore')
    hrv_stats = numeric_hrv.describe().T[['mean', 'std', 'min', '50%', 'max']]
    hrv_stats.rename(columns={"50%": "median"}, inplace=True)

    # Generate tables
    eeg_table = html.Div([
        html.H4("EEG Statistical Summaries"),
        html.Table([
            html.Tr([html.Th(col) for col in ["Metric"] + eeg_stats.columns.tolist()])
        ] + [
            html.Tr([html.Td(idx)] + [html.Td(f"{val:.2f}") for val in row])
            for idx, row in eeg_stats.iterrows()
        ], style={"width": "100%", "border": "1px solid black", "borderCollapse": "collapse"})
    ])

    hrv_table = html.Div([
        html.H4("HRV Statistical Summaries"),
        html.Table([
            html.Tr([html.Th(col) for col in ["Metric"] + hrv_stats.columns.tolist()])
        ] + [
            html.Tr([html.Td(idx)] + [html.Td(f"{val:.2f}") for val in row])
            for idx, row in hrv_stats.iterrows()
        ], style={"width": "100%", "border": "1px solid black", "borderCollapse": "collapse"})
    ])

    return [eeg_table, hrv_table]


@app.callback(
    Output("eeg-band-alerts", "children"),
    Input("eeg-band-table", "children")
)
def update_eeg_alerts(trigger):
    rows = []

    # Define benchmarks
    benchmarks = {
        "Gamma": {"high": 100, "low": 10, "note_high": "may indicate stress, intense mental activity, or noise artifacts.", "note_low": "could indicate underactivity or relaxation."},
        "Delta": {"high": 50, "low": 10, "note_high": "may indicate deep sleep or noise artifacts.", "note_low": "could indicate disrupted sleep or low relaxation."},
        "Alpha": {"high": 50, "low": 5, "note_high": "is associated with relaxation and focus.", "note_low": "could indicate low cognitive or relaxed activity."},
        "Beta": {"high": 35, "low": 12, "note_high": "is linked to active thinking, concentration, and anxiety.", "note_low": "could indicate drowsiness or reduced focus."},
        "Theta": {"high": 8, "low": 4, "note_high": "is linked to drowsiness or early sleep stages.", "note_low": "could indicate low activity or fatigue."}
    }

    # Iterate over each band and organize rows
    for band, thresholds in benchmarks.items():
        high_channels = []
        low_channels = []

        for channel, powers in eeg_band_powers.items():
            if "high" in thresholds and powers[band] > thresholds["high"]:
                high_channels.append(f"{channel} ({powers[band]:.2f} µV²)")
            if "low" in thresholds and powers[band] < thresholds["low"]:
                low_channels.append(f"{channel} ({powers[band]:.2f} µV²)")

        # Add rows for each band
        if high_channels or low_channels:
            rows.append(html.Tr([html.Td(band, rowSpan=2, style={"fontWeight": "bold"})]))
            if high_channels:
                rows.append(html.Tr([
                    html.Td(", ".join(high_channels)),
                    html.Td(f"High {band}: {thresholds['note_high']}")
                ]))
            if low_channels:
                rows.append(html.Tr([
                    html.Td(", ".join(low_channels)),
                    html.Td(f"Low {band}: {thresholds['note_low']}")
                ]))

    # Default if no alerts
    if not rows:
        rows.append(html.Tr([
            html.Td("No abnormalities detected.", colSpan=3, style={"textAlign": "center", "color": "green"})
        ]))

    # Construct table
    alerts_table = html.Table(
        children=[
            html.Thead(html.Tr([
                html.Th("Band"),
                html.Th("Channels (Power)"),
                html.Th("Insights")
            ])),
            html.Tbody(rows)
        ],
        className="alerts-table"
    )

    # Return the table
    return alerts_table

@app.callback(
    Output("hrv-metrics-alerts", "children"),
    Input("hrv-metrics-cards", "children")
)
def update_hrv_alerts(trigger):
    # Define benchmarks based on studies
    low_hr_threshold = 60  # BPM, normal range 60-100 BPM
    high_rmssd_threshold = 50  # ms, RMSSD above this suggests relaxation
    low_rmssd_threshold = 20  # ms, RMSSD below this suggests stress
    low_sdnn_threshold = 50  # ms, normal SDNN above 50 ms
    high_sdnn_threshold = 100  # ms, very high SDNN could indicate artifacts

    alerts = []
    insights = []

    # Generate HRV Alerts and Insights
    if hrv_metrics["MeanHR"] < low_hr_threshold:
        alerts.append(["Mean Heart Rate", f"{hrv_metrics['MeanHR']:.2f} BPM", "Low",
                        "Could indicate bradycardia, requiring medical attention."])
    else:
        insights.append(["Mean Heart Rate", f"{hrv_metrics['MeanHR']:.2f} BPM", "Normal",
                         "Within a healthy range."])

    if hrv_metrics["RMSSD"] > high_rmssd_threshold:
        alerts.append(["RMSSD", f"{hrv_metrics['RMSSD']:.2f} ms", "High",
                       "Suggests a relaxed state but excessive values could indicate artifacts."])
    elif hrv_metrics["RMSSD"] < low_rmssd_threshold:
        alerts.append(["RMSSD", f"{hrv_metrics['RMSSD']:.2f} ms", "Low",
                       "Low variability may reflect stress or autonomic dysfunction."])
    else:
        insights.append(["RMSSD", f"{hrv_metrics['RMSSD']:.2f} ms", "Normal",
                         "Reflects normal variability."])

    if hrv_metrics["SDNN"] < low_sdnn_threshold:
        alerts.append(["SDNN", f"{hrv_metrics['SDNN']:.2f} ms", "Low",
                       "Reduced autonomic nervous system flexibility, possibly due to stress or illness."])
    elif hrv_metrics["SDNN"] > high_sdnn_threshold:
        alerts.append(["SDNN", f"{hrv_metrics['SDNN']:.2f} ms", "High",
                       "High variability might suggest noise or artifacts."])
    else:
        insights.append(["SDNN", f"{hrv_metrics['SDNN']:.2f} ms", "Normal",
                         "Reflects a healthy autonomic nervous system."])

    # Default if no alerts
    if not alerts:
        alerts.append(["-", "No abnormalities detected", "-", "All metrics within normal range."])

    # Create an HRV table
    return html.Table(
        children=[
            html.Thead(
                html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Level"), html.Th("Insight")], className="table-head")
            ),
            html.Tbody(
                [html.Tr([html.Td(a[0]), html.Td(a[1]), html.Td(a[2]), html.Td(a[3])]) for a in alerts + insights]
            ),
        ],
        className="hrv-table",
    )

@app.callback(
    Output("statistical-insights", "children"),
    [Input("statistical-summaries", "id")]
)
def update_statistical_insights(trigger):
    global eeg_data, hrv_data

    if eeg_data is None or hrv_data is None:
        return html.Div("No data available for generating insights.", style={"color": "red"})

    try:
        # Benchmarks
        eeg_benchmarks = {
            "mean": (10, 100),  # µV
            "std": (10, 50),    # µV
        }

        hrv_benchmarks = {
            "mean": (500, 1500),  # ms
            "std": (50, 150),     # ms
        }

        # Filter EEG Signal Columns
        eeg_signal_columns = [col for col in eeg_data.columns if col not in ["Timestamp", "Internal Clock"]]
        eeg_stats = eeg_data[eeg_signal_columns].describe().T

        # Analyze EEG Data
        eeg_table_rows = []
        for channel in eeg_stats.index:
            mean_val = eeg_stats.loc[channel, "mean"]
            std_val = eeg_stats.loc[channel, "std"]

            if std_val > eeg_benchmarks["std"][1]:
                insight = "High variability: Possible noise or artifacts."
            elif mean_val < eeg_benchmarks["mean"][0]:
                insight = "Low activity: Underactivity or suppression."
            elif eeg_benchmarks["mean"][0] <= mean_val <= eeg_benchmarks["mean"][1]:
                insight = "Normal activity: Healthy baseline."
            else:
                insight = "Abnormal activity."

            eeg_table_rows.append([channel, f"{mean_val:.2f} µV", f"{std_val:.2f} µV", insight])

        # HRV Analysis
        hrv_stats = hrv_data.describe().T
        mean_hr = hrv_stats.loc["ECG", "mean"]
        std_hr = hrv_stats.loc["ECG", "std"]

        if mean_hr < hrv_benchmarks["mean"][0]:
            hrv_mean_insight = "Low mean heart rate: Possible bradycardia or measurement issue."
        elif mean_hr > hrv_benchmarks["mean"][1]:
            hrv_mean_insight = "High mean heart rate: Possible tachycardia or noise."
        else:
            hrv_mean_insight = "Normal mean heart rate: Healthy cardiovascular function."

        if std_hr > hrv_benchmarks["std"][1]:
            hrv_std_insight = "High variability: Possible stress or noise."
        elif std_hr < hrv_benchmarks["std"][0]:
            hrv_std_insight = "Low variability: Low HRV or autonomic dysfunction."
        else:
            hrv_std_insight = "Normal variability: Balanced autonomic function."

        hrv_table_rows = [
            ["Mean HR", f"{hrv_metrics['MeanHR']:.2f}", hrv_mean_insight],
            ["SD HR", f"{std_hr:.2f}", hrv_std_insight],
        ]

        # Generate Consolidated Table
        eeg_table = html.Table(
            children=[
                html.Thead(html.Tr([html.Th("Channel"), html.Th("Mean"), html.Th("STD"), html.Th("Insight")])),
                html.Tbody([html.Tr([html.Td(row[0]), html.Td(row[1]), html.Td(row[2]), html.Td(row[3])]) for row in eeg_table_rows]),
            ],
            className="eeg-stats-table",
        )

        hrv_table = html.Table(
            children=[
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Insight")])),
                html.Tbody([html.Tr([html.Td(row[0]), html.Td(row[1]), html.Td(row[2])]) for row in hrv_table_rows]),
            ],
            className="hrv-stats-table",
        )

        # Combine Tables and Headings
        return html.Div([
            html.H4("EEG Statistical Insights:"),
            eeg_table,
            html.H4("HRV Statistical Insights:"),
            hrv_table,
        ])

    except Exception as e:
        return html.Div(f"Error generating insights: {str(e)}", style={"color": "red"})

@app.callback(
    Output("synchronization-validation", "children"),
    [Input("statistical-summaries", "id")]  # Trigger update
)
def update_synchronization_metrics(trigger):
    global eeg_data, hrv_data

    eeg_data_local = eeg_data.copy()
    hrv_data_local = hrv_data.copy()

    if eeg_data_local is None or hrv_data_local is None:
        return html.Div("No data available for synchronization validation.", style={"color": "red"})

    try:
        # Ensure Time column exists and is properly formatted
        if "Time" not in eeg_data_local.columns:
            eeg_data_local["Time"] = eeg_data_local["Timestamp"]
        if "Time" not in hrv_data_local.columns:
            hrv_data_local["Time"] = hrv_data_local["Timestamp"]

        # Convert Time columns to datetime format
        eeg_data_local["Time"] = pd.to_datetime(eeg_data_local["Time"], unit="s")
        hrv_data_local["Time"] = pd.to_datetime(hrv_data_local["Time"], unit="s")

        # Remove duplicates and sort by Time
        eeg_data_2 = eeg_data_local.drop_duplicates(subset=["Time"]).sort_values("Time")
        hrv_data_2 = hrv_data_local.drop_duplicates(subset=["Time"]).sort_values("Time")

        # Merge the data based on nearest timestamps with a grace period
        merged_data = pd.merge_asof(
            eeg_data_2,
            hrv_data_2,
            on="Time",
            direction="nearest",
            tolerance=pd.Timedelta("1ms")  # Grace period of 1ms
        )

        # Calculate latency
        time_diffs = (merged_data["Time"].diff().dt.total_seconds() * 1000).dropna()
        latency = time_diffs.mean()

        time_intervals_eeg = eeg_data_local["Time"].diff().dt.total_seconds().dropna()
        time_intervals_hrv = hrv_data_local["Time"].diff().dt.total_seconds().dropna()

        # Ensure lengths match
        min_length = min(len(time_intervals_eeg), len(time_intervals_hrv))
        time_intervals_eeg = time_intervals_eeg[:min_length]
        time_intervals_hrv = time_intervals_hrv[:min_length]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((time_intervals_eeg - time_intervals_hrv) ** 2))

        # Define benchmarks for comparison
        benchmark_rmse = 10.0  # ms
        benchmark_latency = 50.0  # ms

        # Create RMSE plot
        rmse_fig = go.Figure()
        rmse_fig.add_trace(go.Bar(
            x=["RMSE"],
            y=[rmse],
            name="RMSE",
            marker_color="blue",
            text=[f"{rmse:.2f} ms"],  # Add value on the bar
            textposition="outside",  # Position text outside the bar
        ))
        rmse_fig.add_shape(
            type="line",
            x0=-0.5, x1=0.5, y0=benchmark_rmse, y1=benchmark_rmse,
            line=dict(color="red", dash="dash", width=2),
        )
        rmse_fig.update_layout(
            title="Root Mean Square Error (RMSE)",
            xaxis_title="Metric",
            yaxis_title="Value (ms)",
            showlegend=False,  # Remove legend for simplicity
        )

        latency_fig = go.Figure()
        latency_fig.add_trace(go.Bar(
            x=["Latency"],
            y=[latency],
            name="Latency",
            marker_color="green",
            text=[f"{latency:.2f} ms"],  # Add value on the bar
            textposition="outside",  # Position text outside the bar
        ))
        latency_fig.add_shape(
            type="line",
            x0=-0.5, x1=0.5, y0=benchmark_latency, y1=benchmark_latency,
            line=dict(color="red", dash="dash", width=2),
        )
        latency_fig.update_layout(
            title="Latency",
            xaxis_title="Metric",
            yaxis_title="Value (ms)",
            showlegend=False,  # Remove legend for simplicity
            yaxis=dict(range=[0, latency + 50]),  # Extend y-axis range
        )

        # Return side-by-side layout
        return html.Div(
            style={"display": "flex", "justify-content": "space-between"},
            children=[
                html.Div(dcc.Graph(figure=rmse_fig), style={"width": "48%"}),
                html.Div(dcc.Graph(figure=latency_fig), style={"width": "48%"}),
            ]
        )

    except Exception as e:
        return html.Div(f"Error calculating synchronization metrics: {e}", style={"color": "red"})

if __name__ == "__main__":
    app.run_server(port=8051, debug=False, use_reloader=True)
