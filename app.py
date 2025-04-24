# --- START OF FILE app.py ---

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
import logging
import webbrowser # Added for auto-open
import threading  # Added for auto-open timer

# Determine if running as a PyInstaller bundle
# (Keep this section as is)
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

# Configure upload paths
# (Keep this section as is)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAPS_FOLDER'] = os.path.join(os.getcwd(), 'maps')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MAPS_FOLDER'], exist_ok=True)

# process_gpx function remains the same as the previous version
def process_gpx(file_path, speed_min_threshold, speed_max_threshold, avg_speed_filter_threshold):
    app.logger.info(f"Processing {file_path} with thresholds: min={speed_min_threshold}, max={speed_max_threshold}, avg_min={avg_speed_filter_threshold}")
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time:
                    points.append({
                        'latitude': point.latitude, 'longitude': point.longitude, 'time': point.time
                    })

    if not points: raise ValueError("GPX file contains no points with time information.")

    df = pd.DataFrame(points).sort_values(by='time').reset_index(drop=True)

    speeds, durations, distances = [0.0], [0.0], [0.0]
    for i in range(1, len(df)):
        coord1 = (df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude'])
        coord2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance = geodesic(coord1, coord2).meters
        time_diff = (df.iloc[i]['time'] - df.iloc[i - 1]['time']).total_seconds()
        speed = (distance / time_diff * 3.6) if time_diff > 0.01 else 0
        speeds.append(speed)
        durations.append(time_diff)
        distances.append(distance)

    df['speed'] = speeds
    df['instant_speed'] = speeds
    df['duration'] = durations
    df['distance'] = distances

    def extract_segments(dataframe, speed_min, speed_max, duration_threshold=15, min_segment_duration=15, speed_drop_tolerance=3, merge_tolerance=5):
        initial_segments = []
        current_segment = []
        below_threshold_time = 0
        started = False
        speed_drop_counter = 0

        for i, row in dataframe.iterrows():
            if i == 0: continue
            is_within_speed_band = speed_min <= row['speed'] <= speed_max

            if not started and row['speed'] > speed_min: started = True

            if started:
                if is_within_speed_band or below_threshold_time < 3:
                    current_segment.append(row)
                else:
                    if len(current_segment) > 0:
                         segment_df = pd.DataFrame(current_segment)
                         valid_segment_df = segment_df[(segment_df['speed'] >= speed_min) & (segment_df['speed'] <= speed_max)]
                         if not valid_segment_df.empty and valid_segment_df['duration'].sum() >= min_segment_duration:
                             initial_segments.append(valid_segment_df)
                    current_segment = []
                    below_threshold_time = 0
                    started = False

                if is_within_speed_band:
                    below_threshold_time = 0
                    speed_drop_counter = 0
                else:
                    speed_drop_counter += row['duration']
                    if speed_drop_counter >= speed_drop_tolerance:
                        below_threshold_time += speed_drop_counter
                        speed_drop_counter = 0

            if started and below_threshold_time >= 3 and len(current_segment) > 0:
                segment_df = pd.DataFrame(current_segment)
                valid_segment_df = segment_df[(segment_df['speed'] >= speed_min) & (segment_df['speed'] <= speed_max)]
                if not valid_segment_df.empty and valid_segment_df['duration'].sum() >= min_segment_duration:
                    initial_segments.append(valid_segment_df)
                current_segment = []
                below_threshold_time = 0
                started = False

        if len(current_segment) > 0:
            segment_df = pd.DataFrame(current_segment)
            valid_segment_df = segment_df[(segment_df['speed'] >= speed_min) & (segment_df['speed'] <= speed_max)]
            if not valid_segment_df.empty and valid_segment_df['duration'].sum() >= min_segment_duration:
                 initial_segments.append(valid_segment_df)

        merged_segments = []
        if initial_segments:
            current_merged_segment = initial_segments[0].copy()
            for i in range(1, len(initial_segments)):
                prev_segment = current_merged_segment
                next_segment = initial_segments[i]
                time_gap = (next_segment['time'].iloc[0] - prev_segment['time'].iloc[-1]).total_seconds()
                if 0 <= time_gap <= merge_tolerance:
                    last_point_time_prev = prev_segment['time'].iloc[-1]
                    first_point_time_next = next_segment['time'].iloc[0]
                    bridging_points = dataframe[(dataframe['time'] > last_point_time_prev) & (dataframe['time'] < first_point_time_next)].copy()
                    current_merged_segment = pd.concat([current_merged_segment, bridging_points, next_segment], ignore_index=True)
                else:
                    merged_segments.append(current_merged_segment)
                    current_merged_segment = next_segment.copy()
            merged_segments.append(current_merged_segment)
        else:
            merged_segments = initial_segments

        return merged_segments

    segments_before_avg_filter = extract_segments(df, speed_min=speed_min_threshold, speed_max=speed_max_threshold)
    app.logger.info(f"Found {len(segments_before_avg_filter)} segments before average speed filter.")

    final_segments = []
    for segment in segments_before_avg_filter:
        if not segment.empty:
            points_in_band = segment[(segment['speed'] >= speed_min_threshold) & (segment['speed'] <= speed_max_threshold)]
            if not points_in_band.empty:
                 avg_speed = points_in_band['speed'].mean()
                 if avg_speed >= avg_speed_filter_threshold:
                     final_segments.append(segment)
                 else:
                     app.logger.info(f"Filtering out segment with avg speed {avg_speed:.2f} (below {avg_speed_filter_threshold})")
            else:
                 app.logger.info("Segment has no points within the min/max speed band after potential merging, filtering out.")

    app.logger.info(f"Found {len(final_segments)} segments after average speed filter.")

    if not final_segments:
         return {
            'total_distance': 0, 'total_duration': str(timedelta(seconds=0)), 'total_segments': 0,
            'segment_html_files': [], 'all_segments_html': None, 'segment_details': []
         }

    all_segments_df = pd.concat(final_segments, ignore_index=True)
    segment_html_files = []
    colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
    colormap.caption = f'Speed (km/h) | Point: {speed_min_threshold}-{speed_max_threshold} | Avg Filter: >{avg_speed_filter_threshold}'

    for i, segment in enumerate(final_segments):
        if segment.empty: continue
        segment_map = folium.Map(control_scale=True)
        bounds = [(segment['latitude'].min(), segment['longitude'].min()), (segment['latitude'].max(), segment['longitude'].max())]
        segment_map.location = [segment['latitude'].mean(), segment['longitude'].mean()]

        for j in range(len(segment) - 1):
            point_speed = segment.iloc[j]['speed']
            color_speed = max(speed_min_threshold, min(point_speed, speed_max_threshold))
            folium.PolyLine(
                locations=[(segment.iloc[j]['latitude'], segment.iloc[j]['longitude']), (segment.iloc[j + 1]['latitude'], segment.iloc[j + 1]['longitude'])],
                color=colormap(color_speed), weight=4, opacity=0.7, smooth_factor=2.0, tooltip=f"Speed: {point_speed:.1f} km/h"
            ).add_to(segment_map)
        folium.Marker(location=[segment.iloc[0]['latitude'], segment.iloc[0]['longitude']], popup=f'Start Segment {i+1}', icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(segment_map)
        folium.Marker(location=[segment.iloc[-1]['latitude'], segment.iloc[-1]['longitude']], popup=f'End Segment {i+1}', icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(segment_map)
        colormap.add_to(segment_map)
        segment_map.fit_bounds(bounds)
        html_file = os.path.join(app.config['MAPS_FOLDER'], f'segment_map_{i}.html')
        segment_map.save(html_file)
        segment_html_files.append(f'segment_map_{i}.html')

    all_segments_bounds = [(all_segments_df['latitude'].min(), all_segments_df['longitude'].min()), (all_segments_df['latitude'].max(), all_segments_df['longitude'].max())]
    all_segments_map = folium.Map(control_scale=True)
    all_segments_map.location = [all_segments_df['latitude'].mean(), all_segments_df['longitude'].mean()]
    overview_colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
    overview_colormap.caption = colormap.caption

    for i, segment in enumerate(final_segments):
         if segment.empty: continue
         avg_segment_speed = segment['speed'].mean()
         folium.PolyLine(locations=list(zip(segment['latitude'], segment['longitude'])), color='purple', weight=4, opacity=0.7, smooth_factor=2.0, popup=f'Segment {i+1} (Avg Speed: {avg_segment_speed:.2f} km/h)').add_to(all_segments_map)
         folium.Marker(location=[segment.iloc[0]['latitude'], segment.iloc[0]['longitude']], icon=folium.Icon(color='green', icon='play', prefix='fa'), tooltip=f'Start Segment {i+1}').add_to(all_segments_map)
         folium.Marker(location=[segment.iloc[-1]['latitude'], segment.iloc[-1]['longitude']], icon=folium.Icon(color='red', icon='stop', prefix='fa'), tooltip=f'End Segment {i+1}').add_to(all_segments_map)

    overview_colormap.add_to(all_segments_map)
    all_segments_map.fit_bounds(all_segments_bounds)
    all_segments_html = 'all_segments_map.html'
    all_segments_map.save(os.path.join(app.config['MAPS_FOLDER'], all_segments_html))

    segment_details = []
    total_distance_meters = 0
    total_duration_seconds = 0
    for i, segment in enumerate(final_segments):
        if segment.empty: continue
        segment_dist = segment['distance'].sum()
        segment_dur = segment['duration'].sum()
        segment_avg_speed = (segment_dist / segment_dur * 3.6) if segment_dur > 0 else 0
        segment_details.append({'distance': segment_dist, 'duration': str(timedelta(seconds=int(segment_dur))), 'average_speed': segment_avg_speed})
        total_distance_meters += segment_dist
        total_duration_seconds += segment_dur
    total_duration_str = str(timedelta(seconds=int(total_duration_seconds)))

    return {
        'total_distance': total_distance_meters, 'total_duration': total_duration_str, 'total_segments': len(final_segments),
        'segment_html_files': segment_html_files, 'all_segments_html': all_segments_html, 'segment_details': segment_details
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400

    try:
        # --- UPDATED Default Values ---
        speed_min = float(request.form.get('speed_min', 5.0))  # Default 5.0
        speed_max = float(request.form.get('speed_max', 25.0)) # Default 25.0
        avg_speed_min = float(request.form.get('avg_speed_min', 9.0)) # Default 9.0
    except (ValueError, TypeError):
         return jsonify({'error': 'Invalid speed threshold value provided. Must be a number.'}), 400

    if speed_min < 0 or speed_max < 0 or avg_speed_min < 0: return jsonify({'error': 'Speed thresholds cannot be negative.'}), 400
    if speed_min >= speed_max: return jsonify({'error': 'Minimum point speed must be less than maximum point speed.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        app.logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500

    try:
        result = process_gpx(file_path, speed_min, speed_max, avg_speed_min)
        return jsonify(result)
    except FileNotFoundError:
         app.logger.error(f"Uploaded file not found after saving: {file_path}")
         return jsonify({'error': 'Uploaded file not found after saving.'}), 500
    except gpxpy.gpx.GPXException as e:
         app.logger.warning(f"Error parsing GPX file {file.filename}: {e}", exc_info=True)
         return jsonify({'error': f'Error parsing GPX file: {str(e)}'}), 400
    except ValueError as e:
         app.logger.warning(f"Value error processing {file.filename}: {e}", exc_info=True)
         return jsonify({'error': str(e)}), 400
    except Exception as e:
         app.logger.error(f"Error processing GPX file {file.filename}: {e}", exc_info=True)
         return jsonify({'error': f'An unexpected error occurred during processing.'}), 500


@app.route('/maps/<filename>')
def serve_map(filename):
    if '..' in filename or filename.startswith('/'): return jsonify({'error': 'Invalid filename'}), 400
    return send_from_directory(app.config['MAPS_FOLDER'], filename)

# --- NEW Function to open browser ---
def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # --- Add timer to open browser ---
    # Run browser 1.5 seconds after Flask server starts
    threading.Timer(1.5, open_browser).start()
    # --- Run app on 127.0.0.1 for consistency with open_browser URL ---
    app.run(host='127.0.0.1', port=5000)

# --- END OF FILE app.py ---