import sys
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import gpxpy
from geopy.distance import geodesic
from datetime import timedelta
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from concurrent.futures import ThreadPoolExecutor

# Determine if running as a PyInstaller bundle
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

# Configure upload paths
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAPS_FOLDER'] = os.path.join(os.getcwd(), 'maps')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MAPS_FOLDER'], exist_ok=True)

def process_gpx(file_path):
    # Load and parse GPX file
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    # Extract GPX data into DataFrame
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'time': point.time
                })

    df = pd.DataFrame(points)

    # Calculate speed, distance, and duration
    speeds = []
    durations = []
    distances = []
    for i in range(1, len(df)):
        coord1 = (df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude'])
        coord2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance = geodesic(coord1, coord2).meters
        time_diff = (df.iloc[i]['time'] - df.iloc[i - 1]['time']).total_seconds()
        speed = (distance / time_diff) * 3.6 if time_diff > 0 else 0  # Convert m/s to km/h
        speeds.append(speed)
        durations.append(time_diff)
        distances.append(distance)

    df = df.iloc[1:]
    df['speed'] = speeds
    df['instant_speed'] = speeds
    df['duration'] = durations
    df['distance'] = distances

    # Segment data based on speed
    def extract_segments(dataframe, speed_min=8, speed_max=18, duration_threshold=15, min_segment_duration=15, speed_drop_tolerance=3, merge_tolerance=5):
        segments = []
        current_segment = []
        below_threshold_time = 0
        started = False
        speed_drop_counter = 0

        for i, row in dataframe.iterrows():
            if row['speed'] > speed_min:
                started = True
            if started:
                current_segment.append(row)
            if speed_min <= row['speed'] <= speed_max:
                speed_drop_counter = 0
                below_threshold_time = 0  # Reset counter if speed is within range
            else:
                speed_drop_counter += row['duration']
                if speed_drop_counter > speed_drop_tolerance:
                    below_threshold_time += speed_drop_counter
                    speed_drop_counter = 0

            # If the speed is below threshold for 3 or more seconds, finalize the segment
            if below_threshold_time >= 3 and len(current_segment) > 0 and pd.DataFrame(current_segment)['duration'].sum() >= min_segment_duration:
                segment_df = pd.DataFrame(current_segment)
                if segment_df['speed'].mean() >= speed_min:
                    segments.append(segment_df)
                current_segment = []
                below_threshold_time = 0

        if len(current_segment) > 0:
            segment_df = pd.DataFrame(current_segment)
            if segment_df['duration'].sum() >= duration_threshold:
                segments.append(segment_df)

        # Merge segments if they are close enough in time and space
        merged_segments = []
        if segments:
            current_merged_segment = segments[0]
            for i in range(1, len(segments)):
                prev_segment = current_merged_segment
                next_segment = segments[i]
                time_gap = (next_segment['time'].iloc[0] - prev_segment['time'].iloc[-1]).total_seconds()
                if time_gap <= merge_tolerance:
                    current_merged_segment = pd.concat([current_merged_segment, next_segment], ignore_index=True)
                else:
                    merged_segments.append(current_merged_segment)
                    current_merged_segment = next_segment
            merged_segments.append(current_merged_segment)
        else:
            merged_segments = segments

        return merged_segments

    # Extract segments using adjusted thresholds
    segments = extract_segments(df, speed_min=9, speed_max=16)
    all_segments_df = pd.concat(segments, ignore_index=True)
    all_segments_df['segment_id'] = 'All Segments'

    # Generate folium maps for each segment
    segment_html_files = []
    for i, segment in enumerate(segments):
        bounds = [(segment['latitude'].min(), segment['longitude'].min()), 
                  (segment['latitude'].max(), segment['longitude'].max())]
        segment_map = folium.Map(location=[segment['latitude'].mean(), segment['longitude'].mean()], control_scale=True)
        colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=9, vmax=16).to_step(n=200)

        # Plot the segment with specific color
        for j in range(len(segment) - 1):
            folium.PolyLine(
                locations=[(segment.iloc[j]['latitude'], segment.iloc[j]['longitude']),
                           (segment.iloc[j + 1]['latitude'], segment.iloc[j + 1]['longitude'])],
                color=colormap(segment.iloc[j]['speed']),
                weight=4,
                opacity=0.7,
                smooth_factor=2.0
            ).add_to(segment_map)
        folium.Marker(
            location=[segment.iloc[0]['latitude'], segment.iloc[0]['longitude']],
            popup='Start',
            icon=folium.Icon(color='green')
        ).add_to(segment_map)
        folium.Marker(
            location=[segment.iloc[-1]['latitude'], segment.iloc[-1]['longitude']],
            popup='End',
            icon=folium.Icon(color='red')
        ).add_to(segment_map)
        colormap.add_to(segment_map)

        # Fit the map to the bounds of the segment
        segment_map.fit_bounds(bounds)

        # Save map as HTML file
        html_file = os.path.join(app.config['MAPS_FOLDER'], f'segment_map_{i}.html')
        segment_map.save(html_file)
        segment_html_files.append(f'segment_map_{i}.html')  # Return only the filename

    # Generate folium map for all segments combined
    all_segments_bounds = [(all_segments_df['latitude'].min(), all_segments_df['longitude'].min()), 
                           (all_segments_df['latitude'].max(), all_segments_df['longitude'].max())]
    all_segments_map = folium.Map(location=[all_segments_df['latitude'].mean(), all_segments_df['longitude'].mean()], zoom_start=14)
    colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=9, vmax=16).to_step(n=200)

    # Plot all segments without connecting them
    for segment in segments:
        folium.PolyLine(
            locations=list(zip(segment['latitude'], segment['longitude'])),
            color=colormap(segment['instant_speed'].mean()),
            weight=4,
            opacity=0.7,
            smooth_factor=2.0
        ).add_to(all_segments_map)

    colormap.add_to(all_segments_map)  # Ensure the speed scale is preserved

    # Fit the map to the bounds of all segments
    all_segments_map.fit_bounds(all_segments_bounds)

    all_segments_html = 'all_segments_map.html'  # Return only the filename
    all_segments_map.save(os.path.join(app.config['MAPS_FOLDER'], all_segments_html))

    # Convert timedelta to a string for JSON serialization
    total_duration_str = str(timedelta(seconds=int(sum(segment['duration'].sum() for segment in segments))))
    
    segment_details = []
    for i, segment in enumerate(segments):
        segment_details.append({
            'distance': segment['distance'].sum(),
            'duration': str(timedelta(seconds=int(segment['duration'].sum()))),
            'average_speed': segment['speed'].mean()
    })
    # Return results
    return {
        'total_distance': sum(segment['distance'].sum() for segment in segments),
        'total_duration': total_duration_str,  # Use the string representation
        'total_segments': len(segments),
        'segment_html_files': segment_html_files,
        'all_segments_html': all_segments_html,
        'segment_details': segment_details  # Add this line
    }
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the GPX file
    result = process_gpx(file_path)

    # Return the results
    return jsonify(result)

@app.route('/maps/<filename>')
def serve_map(filename):
    return send_from_directory(app.config['MAPS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)