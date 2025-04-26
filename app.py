# --- START OF FILE app.py ---

import sys
import os
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import gpxpy
import gpxpy.gpx
from geopy.distance import geodesic
from datetime import timedelta
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from concurrent.futures import ThreadPoolExecutor
import logging
import webbrowser
import threading
import io
import time
import copy # Needed for deep copying original segments

# --- Flask App Setup ---
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAPS_FOLDER'] = os.path.join(os.getcwd(), 'maps')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MAPS_FOLDER'], exist_ok=True)

# --- Global State ---
last_processed_data = {
    'segments': None,           # Current state (potentially trimmed)
    'original_segments': None, # Segments as originally processed
    'original_filename': None,
    'thresholds': None
}

# --- Helper Functions ---
def parse_duration_to_seconds(duration_str):
    if not duration_str or isinstance(duration_str, (int, float)): return float(duration_str) if isinstance(duration_str, (int, float)) else 0
    parts = duration_str.split(':'); seconds = 0
    try:
        if len(parts) == 3: seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2: seconds = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1: seconds = float(parts[0])
    except ValueError: return 0
    return seconds

def calculate_segment_stats(segment_df):
    if segment_df is None or segment_df.empty or len(segment_df) < 2:
        return {'distance': 0, 'duration': '0:00:00', 'average_speed': 0, 'duration_seconds': 0}
    segment_df = segment_df.copy() # Ensure we don't modify the original
    segment_df['time'] = pd.to_datetime(segment_df['time'])
    distances = [0.0]
    durations = [0.0]
    for i in range(1, len(segment_df)):
        coord1 = (segment_df.iloc[i - 1]['latitude'], segment_df.iloc[i - 1]['longitude'])
        coord2 = (segment_df.iloc[i]['latitude'], segment_df.iloc[i]['longitude'])
        try:
            distance = geodesic(coord1, coord2).meters
        except ValueError: # Handle potential invalid coordinates
             distance = 0
        time_diff = (segment_df.iloc[i]['time'] - segment_df.iloc[i - 1]['time']).total_seconds()
        distances.append(distance)
        durations.append(time_diff if time_diff >= 0 else 0) # Prevent negative durations

    segment_df['distance'] = distances
    segment_df['point_duration'] = durations # Duration between points

    total_dist = segment_df['distance'].sum()
    total_dur_seconds = 0
    if len(segment_df) > 1:
        time_diff = (segment_df['time'].iloc[-1] - segment_df['time'].iloc[0]).total_seconds()
        total_dur_seconds = time_diff if time_diff >= 0 else 0

    avg_speed = (total_dist / total_dur_seconds * 3.6) if total_dur_seconds > 0 else 0
    duration_str = str(timedelta(seconds=int(total_dur_seconds)))
    return {'distance': total_dist, 'duration': duration_str, 'average_speed': avg_speed, 'duration_seconds': total_dur_seconds}


# --- Map Generation Helpers ---
def generate_map_html(map_object, output_filepath):
    """Adds highlight script and saves the map."""
    # Corrected loop condition: i <= endIndex - 1
    map_highlight_script = """
        <script>
            var polylines = []; // Store references to the line segments
            var originalStyles = []; // Store original styles to reset

            // Function called by parent window to highlight a range
            function highlightTrimRange(startIndex, endIndex) {
                console.log('Map highlightTrimRange called with:', startIndex, endIndex);
                if (polylines.length === 0) {
                    console.log('Polylines not populated yet.');
                    return; // Should not happen if map loaded
                }
                resetHighlight(); // Reset previous highlight first

                for (let i = 0; i < polylines.length; i++) {
                    // Highlight line segment 'i' (connecting point i to i+1)
                    // if its start index 'i' is within the range [startIndex, endIndex-1].
                    if (i >= startIndex && i <= endIndex - 1) { // Corrected loop condition
                         // Store original style if not already stored
                         if (!originalStyles[i]) {
                             originalStyles[i] = {
                                 color: polylines[i].options.color,
                                 weight: polylines[i].options.weight,
                                 opacity: polylines[i].options.opacity
                             };
                         }
                        // Apply highlight style (e.g., brighter color, thicker line)
                        polylines[i].setStyle({ color: '#FF00FF', weight: 8, opacity: 1.0 }); // Bright Magenta, thicker
                    }
                }
                 console.log('Highlight applied for range:', startIndex, '-', endIndex);
            }

            // Function to reset all lines to original style
            function resetHighlight() {
                console.log('Resetting highlight');
                for (let i = 0; i < polylines.length; i++) {
                    if (originalStyles[i]) {
                        polylines[i].setStyle(originalStyles[i]);
                    }
                    // If original style wasn't stored (shouldn't happen), maybe default?
                    // else { polylines[i].setStyle({ weight: 5, opacity: 0.8 }); }
                }
                 // Do not clear originalStyles here, keep them for subsequent highlights
                 console.log('Highlight reset complete');
            }

            // Ensure this runs after the map object (e.g., 'map_xxxxx') is defined
            setTimeout(() => {
                 try {
                     const mapVarName = Object.keys(window).find(k => k.startsWith('map_'));
                     if (!mapVarName) { console.error('Map variable not found'); return; }
                     const mapInstance = window[mapVarName];
                     if (!mapInstance) { console.error('Map instance not found'); return; }

                     mapInstance.eachLayer(function(layer) {
                         // Check if it's a Polyline and not part of the colormap scale
                         if (layer instanceof L.Polyline && !layer.options.isColorScale) {
                             polylines.push(layer);
                         }
                     });
                     // Polylines should be added in the order they appear in the GPX segment
                     console.log('Found', polylines.length, 'polylines for highlighting.');
                 } catch (e) {
                     console.error("Error accessing map layers for highlighting:", e);
                 }
            }, 500); // Delay to allow map layers to potentially load

        </script>
    """
    map_object.get_root().html.add_child(folium.Element(map_highlight_script))
    map_object.save(output_filepath)
    app.logger.info(f"Generated map file: {output_filepath}")


def generate_raw_track_map(full_df, thresholds, output_folder):
    """Generates map of the full track, color-coded by speed."""
    if full_df is None or full_df.empty or len(full_df) < 2:
        return None
    try:
        map_raw = folium.Map(control_scale=True)
        bounds = [(full_df['latitude'].min(), full_df['longitude'].min()), (full_df['latitude'].max(), full_df['longitude'].max())]
        map_raw.location = [full_df['latitude'].mean(), full_df['longitude'].mean()]

        speed_min_threshold = thresholds.get('speed_min', 0)
        speed_max_threshold = thresholds.get('speed_max', 100)
        colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
        colormap.caption = f'Speed (km/h)' # Simple caption for raw map

        for j in range(len(full_df) - 1):
            point_speed = full_df.iloc[j + 1].get('speed', 0)
            color_speed = max(speed_min_threshold, min(point_speed, speed_max_threshold))
            line = folium.PolyLine(
                locations=[(full_df.iloc[j]['latitude'], full_df.iloc[j]['longitude']), (full_df.iloc[j + 1]['latitude'], full_df.iloc[j + 1]['longitude'])],
                color=colormap(color_speed), weight=4, opacity=0.7,
                tooltip=f"Idx: {j}, Speed: {point_speed:.1f} km/h" # Index of point J
            )
            line.add_to(map_raw)

        folium.Marker(location=[full_df.iloc[0]['latitude'], full_df.iloc[0]['longitude']], popup='Track Start', icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(map_raw)
        folium.Marker(location=[full_df.iloc[-1]['latitude'], full_df.iloc[-1]['longitude']], popup='Track End', icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(map_raw)

        colormap.add_to(map_raw)
        map_raw.fit_bounds(bounds, padding=(0.01, 0.01))

        timestamp = int(time.time())
        raw_filename = f'raw_track_map_{timestamp}.html'
        raw_filepath = os.path.join(output_folder, raw_filename)
        # Save raw map WITHOUT the highlight script, as it's not needed/applicable here
        map_raw.save(raw_filepath)
        app.logger.info(f"Generated raw map file: {raw_filepath}")
        return raw_filename
    except Exception as e:
        app.logger.error(f"Error generating raw track map: {e}", exc_info=True)
        return None


def generate_segment_map(segment_df, segment_index, thresholds, output_folder):
    """Generates an HTML map file for a single segment."""
    if segment_df is None or segment_df.empty: return None
    try:
        map_segment = folium.Map(location=[segment_df['latitude'].mean(), segment_df['longitude'].mean()], zoom_start=15, control_scale=True)
        bounds = [(segment_df['latitude'].min(), segment_df['longitude'].min()), (segment_df['latitude'].max(), segment_df['longitude'].max())]
        speed_min_threshold = thresholds.get('speed_min', 0)
        speed_max_threshold = thresholds.get('speed_max', 100)
        colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
        colormap.caption = f'Speed (km/h) | Thresholds: {speed_min_threshold}-{speed_max_threshold}'

        for j in range(len(segment_df) - 1):
            point_speed = segment_df.iloc[j + 1].get('speed', 0)
            color_speed = max(speed_min_threshold, min(point_speed, speed_max_threshold))
            line = folium.PolyLine(
                locations=[(segment_df.iloc[j]['latitude'], segment_df.iloc[j]['longitude']), (segment_df.iloc[j + 1]['latitude'], segment_df.iloc[j + 1]['longitude'])],
                color=colormap(color_speed), weight=5, opacity=0.8,
                tooltip=f"Idx: {j} -> {j+1}, Speed: {point_speed:.1f} km/h"
            )
            line.add_to(map_segment)

        folium.Marker(location=[segment_df.iloc[0]['latitude'], segment_df.iloc[0]['longitude']], popup=f'Start Segment {segment_index + 1} (Idx 0)', icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(map_segment)
        folium.Marker(location=[segment_df.iloc[-1]['latitude'], segment_df.iloc[-1]['longitude']], popup=f'End Segment {segment_index + 1} (Idx {len(segment_df) - 1})', icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(map_segment)

        colormap.add_to(map_segment)
        map_segment.fit_bounds(bounds, padding=(0.01, 0.01))

        segment_filename = f'segment_map_{segment_index}.html'
        segment_filepath = os.path.join(output_folder, segment_filename)
        # Use helper to save segment map WITH highlight script
        generate_map_html(map_segment, segment_filepath)
        return segment_filename
    except Exception as e:
        app.logger.error(f"Error generating map for segment {segment_index}: {e}", exc_info=True)
        return None


# --- GPX Processing Function ---
def process_gpx(file_path, speed_min_threshold, speed_max_threshold, avg_speed_filter_threshold):
    app.logger.info(f"Processing {file_path} with thresholds: min={speed_min_threshold}, max={speed_max_threshold}, avg_min={avg_speed_filter_threshold}")
    maps_output_folder = app.config['MAPS_FOLDER']
    thresholds = {'speed_min': speed_min_threshold, 'speed_max': speed_max_threshold, 'avg_speed_min': avg_speed_filter_threshold}
    raw_track_map_filename = None
    final_segments = []
    segment_details = []
    segment_html_files = []
    all_segments_html = None
    total_distance_meters = 0
    total_duration_str = '0:00:00'

    try:
        # 1. Parse GPX and Calculate Initial Speeds
        with open(file_path, 'r') as gpx_file: gpx = gpxpy.parse(gpx_file)
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.time: points.append({'latitude': point.latitude, 'longitude': point.longitude, 'time': point.time})
        if not points: raise ValueError("GPX file contains no points with time information.")
        df = pd.DataFrame(points).sort_values(by='time').reset_index(drop=True); df['time'] = pd.to_datetime(df['time'])
        speeds, durations, distances = [0.0], [0.0], [0.0]
        for i in range(1, len(df)):
            coord1 = (df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude']); coord2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
            try: distance = geodesic(coord1, coord2).meters
            except ValueError: distance = 0
            time_diff = (df.iloc[i]['time'] - df.iloc[i - 1]['time']).total_seconds(); speed = (distance / time_diff * 3.6) if time_diff > 0.01 else 0
            speeds.append(speed); durations.append(time_diff if time_diff >= 0 else 0); distances.append(distance)
        df['speed'] = speeds; df['instant_speed'] = speeds; df['duration'] = durations; df['distance'] = distances
        app.logger.info("Initial GPX parsing and speed calculation complete.")

        # 2. Generate Raw Track Map
        raw_track_map_filename = generate_raw_track_map(df, thresholds, maps_output_folder)
        if not raw_track_map_filename: app.logger.warning("Failed to generate raw track map.")

        # 3. Perform Segmentation and Filtering
        def extract_segments(dataframe, speed_min, speed_max, min_segment_duration=15, speed_drop_tolerance=3, merge_tolerance=5):
            initial_segments = []; current_segment = []; below_threshold_time = 0; started = False; speed_drop_counter = 0
            for i, row in dataframe.iterrows():
                if i == 0: continue
                is_within_speed_band = speed_min <= row['speed'] <= speed_max
                if not started and row['speed'] > speed_min: started = True
                if started:
                    if is_within_speed_band or below_threshold_time < 3: current_segment.append(row)
                    else:
                        if len(current_segment) > 0:
                            segment_df = pd.DataFrame(current_segment); valid_points_df = segment_df[(segment_df['speed'] >= speed_min) & (segment_df['speed'] <= speed_max)]
                            if not valid_points_df.empty and valid_points_df['duration'].sum() >= min_segment_duration: initial_segments.append(segment_df)
                        current_segment = []; below_threshold_time = 0; started = False
                    if is_within_speed_band: below_threshold_time = 0; speed_drop_counter = 0
                    else:
                        speed_drop_counter += row['duration'];
                        if speed_drop_counter >= speed_drop_tolerance: below_threshold_time += speed_drop_counter; speed_drop_counter = 0
                if started and below_threshold_time >= 3 and len(current_segment) > 0:
                    segment_df = pd.DataFrame(current_segment); valid_points_df = segment_df[(segment_df['speed'] >= speed_min) & (segment_df['speed'] <= speed_max)]
                    if not valid_points_df.empty and valid_points_df['duration'].sum() >= min_segment_duration: initial_segments.append(segment_df)
                    current_segment = []; below_threshold_time = 0; started = False
            if len(current_segment) > 0:
                segment_df = pd.DataFrame(current_segment); valid_points_df = segment_df[(segment_df['speed'] >= speed_min) & (segment_df['speed'] <= speed_max)]
                if not valid_points_df.empty and valid_points_df['duration'].sum() >= min_segment_duration: initial_segments.append(segment_df)
            merged_segments = []
            if initial_segments:
                current_merged_segment = initial_segments[0].copy()
                for i in range(1, len(initial_segments)):
                    prev_segment = current_merged_segment; next_segment = initial_segments[i]; prev_segment['time'] = pd.to_datetime(prev_segment['time']); next_segment['time'] = pd.to_datetime(next_segment['time'])
                    time_gap = (next_segment['time'].iloc[0] - prev_segment['time'].iloc[-1]).total_seconds()
                    if 0 <= time_gap <= merge_tolerance:
                        last_point_time_prev = prev_segment['time'].iloc[-1]; first_point_time_next = next_segment['time'].iloc[0]
                        bridging_points = dataframe[(dataframe['time'] > last_point_time_prev) & (dataframe['time'] < first_point_time_next)].copy()
                        current_merged_segment = pd.concat([current_merged_segment, bridging_points, next_segment], ignore_index=True).sort_values(by='time').reset_index(drop=True)
                    else: merged_segments.append(current_merged_segment); current_merged_segment = next_segment.copy()
                merged_segments.append(current_merged_segment)
            else: merged_segments = initial_segments
            return merged_segments

        segments_before_avg_filter = extract_segments(df, speed_min=speed_min_threshold, speed_max=speed_max_threshold)
        app.logger.info(f"Found {len(segments_before_avg_filter)} segments before average speed filter.")
        temp_final_segments = []
        for segment_df in segments_before_avg_filter:
            if segment_df is None or segment_df.empty: continue
            stats = calculate_segment_stats(segment_df.copy())
            if stats['average_speed'] >= avg_speed_filter_threshold: temp_final_segments.append(segment_df)
            else: app.logger.info(f"Filtering out segment with avg speed {stats['average_speed']:.2f} km/h")
        final_segments = temp_final_segments
        app.logger.info(f"Found {len(final_segments)} segments after average speed filter.")

        # 4. Store Final Segments & Calculate Details
        global last_processed_data
        last_processed_data['original_segments'] = [seg.copy() for seg in final_segments]
        last_processed_data['segments'] = [seg.copy() for seg in final_segments]
        last_processed_data['original_filename'] = os.path.basename(file_path)
        last_processed_data['thresholds'] = thresholds

        total_duration_seconds = 0
        if final_segments:
            for i, segment in enumerate(final_segments):
                stats = calculate_segment_stats(segment.copy())
                stats['point_count'] = len(segment); stats['original_index'] = i
                segment_details.append(stats)
                total_distance_meters += stats['distance']; total_duration_seconds += stats['duration_seconds']
        total_duration_str = str(timedelta(seconds=int(total_duration_seconds)))

        # 5. Generate Final Segment Maps
        if final_segments:
            for i, segment in enumerate(final_segments):
                html_filename = generate_segment_map(segment, i, thresholds, maps_output_folder)
                if html_filename: segment_html_files.append(html_filename)

        # 6. Generate All Segments Map (Optional Overview)
        if final_segments:
            try:
                all_segments_df = pd.concat(final_segments, ignore_index=True)
                all_segments_bounds = [(all_segments_df['latitude'].min(), all_segments_df['longitude'].min()), (all_segments_df['latitude'].max(), all_segments_df['longitude'].max())]
                all_segments_map = folium.Map(control_scale=True, location=[all_segments_df['latitude'].mean(), all_segments_df['longitude'].mean()])
                overview_colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
                overview_colormap.caption = f'Speed (km/h) | Point: {speed_min_threshold}-{speed_max_threshold} | Avg Filter: >{avg_speed_filter_threshold}'
                for i, segment in enumerate(final_segments):
                    if segment.empty or i >= len(segment_details): continue
                    avg_segment_speed = segment_details[i]['average_speed']
                    folium.PolyLine(locations=list(zip(segment['latitude'], segment['longitude'])), color='purple', weight=4, opacity=0.7, smooth_factor=2.0, popup=f'Segment {i+1} (Avg Speed: {avg_segment_speed:.2f} km/h)').add_to(all_segments_map)
                    folium.Marker(location=[segment.iloc[0]['latitude'], segment.iloc[0]['longitude']], icon=folium.Icon(color='green', icon='play', prefix='fa'), tooltip=f'Start Segment {i+1}').add_to(all_segments_map)
                    folium.Marker(location=[segment.iloc[-1]['latitude'], segment.iloc[-1]['longitude']], icon=folium.Icon(color='red', icon='stop', prefix='fa'), tooltip=f'End Segment {i+1}').add_to(all_segments_map)
                overview_colormap.add_to(all_segments_map); all_segments_map.fit_bounds(all_segments_bounds, padding=(0.01, 0.01))
                all_segments_html = 'all_segments_map.html'; all_segments_map.save(os.path.join(maps_output_folder, all_segments_html))
            except Exception as e:
                app.logger.error(f"Error generating all segments overview map: {e}", exc_info=True)
                all_segments_html = None

    except Exception as process_error:
        app.logger.error(f"Error during GPX segmentation/processing: {process_error}", exc_info=True)
        return {'raw_track_map_filename': raw_track_map_filename, 'error': f'Error during segment processing: {str(process_error)}', 'total_segments': 0, 'segment_details': [], 'segment_html_files': [], 'all_segments_html': None, 'total_distance': 0, 'total_duration': '0:00:00'}

    # 7. Return All Successful Results
    return {'raw_track_map_filename': raw_track_map_filename, 'total_distance': total_distance_meters, 'total_duration': total_duration_str, 'total_segments': len(final_segments), 'segment_html_files': segment_html_files, 'all_segments_html': all_segments_html, 'segment_details': segment_details}

# --- Routes ---
@app.route('/upload', methods=['POST'])
def upload():
    global last_processed_data; last_processed_data = {'segments': None, 'original_segments': None, 'original_filename': None, 'thresholds': None}
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file'];
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    try: speed_min = float(request.form.get('speed_min', 5.0)); speed_max = float(request.form.get('speed_max', 25.0)); avg_speed_min = float(request.form.get('avg_speed_min', 9.0))
    except (ValueError, TypeError): return jsonify({'error': 'Invalid speed threshold value provided.'}), 400
    if speed_min < 0 or speed_max < 0 or avg_speed_min < 0: return jsonify({'error': 'Speed thresholds cannot be negative.'}), 400
    if speed_min >= speed_max: return jsonify({'error': 'Min point speed must be < max point speed.'}), 400
    filename = file.filename; file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try: file.save(file_path); app.logger.info(f"Successfully saved {filename} to {file_path}")
    except Exception as e: app.logger.error(f"Failed to save uploaded file '{filename}': {e}", exc_info=True); return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500
    try:
        app.logger.info(f"Processing GPX file: {file_path}")
        result = process_gpx(file_path, speed_min, speed_max, avg_speed_min)
        app.logger.info(f"GPX processing finished for {filename}. Segments found: {result.get('total_segments', 0)}")
        if 'error' in result: return jsonify(result), 200 # Return partial results + error
        else: return jsonify(result) # Return full results
    except Exception as e: app.logger.error(f"Unexpected error calling process_gpx for {filename}: {e}", exc_info=True); last_processed_data = {'segments': None, 'original_segments': None, 'original_filename': None, 'thresholds': None}; return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/reset_segment/<int:segment_index>', methods=['POST'])
def reset_segment(segment_index):
    global last_processed_data
    if (not last_processed_data or not last_processed_data.get('segments') or not last_processed_data.get('original_segments') or not last_processed_data.get('thresholds') or segment_index < 0 or segment_index >= len(last_processed_data['segments']) or segment_index >= len(last_processed_data['original_segments'])):
        app.logger.warning(f"Invalid request to reset segment index: {segment_index}")
        return jsonify({'error': 'Segment data not found or invalid index for reset.'}), 400
    try:
        original_segment_df = copy.deepcopy(last_processed_data['original_segments'][segment_index])
        last_processed_data['segments'][segment_index] = original_segment_df # Reset working copy
        app.logger.info(f"Segment {segment_index + 1} reset to original state.")
        reset_stats = calculate_segment_stats(original_segment_df); reset_stats['point_count'] = len(original_segment_df); reset_stats['original_index'] = segment_index
        generate_segment_map(original_segment_df, segment_index, last_processed_data['thresholds'], app.config['MAPS_FOLDER']) # Regenerate map
        return jsonify({'message': f'Segment {segment_index + 1} reset successfully.', 'reset_stats': reset_stats, 'original_segment_index': segment_index})
    except Exception as e: app.logger.error(f"Error resetting segment {segment_index}: {e}", exc_info=True); return jsonify({'error': f'Failed to reset segment {segment_index + 1}.'}), 500

@app.route('/trim', methods=['POST'])
def trim_segment():
    global last_processed_data
    if not last_processed_data or not last_processed_data.get('segments') or not last_processed_data.get('thresholds'): return jsonify({'error': 'No processed segment data or thresholds available.'}), 400
    try: data = request.get_json(); segment_index = int(data['segment_index']); start_point_index = int(data['start_index']); end_point_index = int(data['end_index'])
    except (TypeError, KeyError, ValueError): return jsonify({'error': 'Invalid trim parameters.'}), 400
    if segment_index < 0 or segment_index >= len(last_processed_data['segments']): return jsonify({'error': f'Invalid segment index {segment_index}.'}), 400
    current_segment_df = last_processed_data['segments'][segment_index]; max_index = len(current_segment_df) - 1
    if start_point_index < 0 or end_point_index < 0 or start_point_index > max_index or end_point_index > max_index or start_point_index > end_point_index: return jsonify({'error': f'Invalid trim indices [{start_point_index}-{end_point_index}]. Must be between 0 and {max_index} for current points in segment {segment_index+1}, and start <= end.'}), 400
    trimmed_df = current_segment_df.iloc[start_point_index : end_point_index + 1].copy().reset_index(drop=True)
    if len(trimmed_df) < 2: return jsonify({'error': 'Trimmed segment must have at least 2 points.'}), 400
    trimmed_stats = calculate_segment_stats(trimmed_df); trimmed_stats['point_count'] = len(trimmed_df); trimmed_stats['original_index'] = segment_index
    last_processed_data['segments'][segment_index] = trimmed_df # Update state
    app.logger.info(f"Segment {segment_index + 1} trimmed and server state updated.")
    try: generate_segment_map(trimmed_df, segment_index, last_processed_data['thresholds'], app.config['MAPS_FOLDER']) # Regenerate map
    except Exception as e: app.logger.error(f"Failed to regenerate map for trimmed segment {segment_index}: {e}", exc_info=True)
    return jsonify({'message': 'Trim successful, server state updated, map regenerated.', 'trimmed_stats': trimmed_stats, 'original_segment_index': segment_index })

@app.route('/export/<int:segment_index>')
def export_segment_gpx(segment_index):
    global last_processed_data
    if not last_processed_data or not last_processed_data.get('segments') or segment_index < 0 or segment_index >= len(last_processed_data['segments']): return jsonify({'error': 'Segment data not found or invalid index.'}), 404
    segment_df = last_processed_data['segments'][segment_index] # Export current state
    original_filename_base = os.path.splitext(last_processed_data.get('original_filename', 'export'))[0]; export_filename = f"{original_filename_base}_segment_{segment_index + 1}.gpx"
    if segment_df.empty: return jsonify({'error': 'Cannot export an empty segment.'}), 400
    gpx_export = gpxpy.gpx.GPX(); gpx_track = gpxpy.gpx.GPXTrack(); gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment); gpx_export.tracks.append(gpx_track)
    for _, row in segment_df.iterrows(): point = gpxpy.gpx.GPXTrackPoint(latitude=row['latitude'], longitude=row['longitude'], time=pd.to_datetime(row['time'])); gpx_segment.points.append(point)
    gpx_xml = gpx_export.to_xml(); response = make_response(gpx_xml)
    response.headers['Content-Type'] = 'application/gpx+xml'; response.headers['Content-Disposition'] = f'attachment; filename="{export_filename}"'
    return response

@app.route('/maps/<filename>')
def serve_map(filename):
    if '..' in filename or filename.startswith('/'): return jsonify({'error': 'Invalid filename'}), 400
    response = make_response(send_from_directory(app.config['MAPS_FOLDER'], filename))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'; response.headers['Pragma'] = 'no-cache'; response.headers['Expires'] = '0'
    return response

@app.route('/favicon.ico')
def favicon(): return '', 204

@app.route('/')
def index():
    global last_processed_data; last_processed_data = {'segments': None, 'original_segments': None, 'original_filename': None, 'thresholds': None}
    return render_template('index.html')

def open_browser(): webbrowser.open_new('http://127.0.0.1:5000')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    threading.Timer(1.5, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=False)

# --- END OF FILE app.py ---
