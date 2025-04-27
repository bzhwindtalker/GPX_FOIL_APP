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
import logging
import webbrowser
import threading
import io
import time
import copy # Needed for deep copying original segments
import uuid # For unique preview filenames

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
    'segments': None,
    'original_segments': None, # Stores the initial state of segments after first processing
    'segment_details': None,
    'original_filename': None,
    'original_file_path': None,
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
    segment_df = segment_df.copy()
    segment_df['time'] = pd.to_datetime(segment_df['time'])
    distances = [0.0]; durations = [0.0]
    for i in range(1, len(segment_df)):
        coord1 = (segment_df.iloc[i - 1]['latitude'], segment_df.iloc[i - 1]['longitude'])
        coord2 = (segment_df.iloc[i]['latitude'], segment_df.iloc[i]['longitude'])
        try: distance = geodesic(coord1, coord2).meters
        except ValueError: distance = 0
        time_diff = (segment_df.iloc[i]['time'] - segment_df.iloc[i - 1]['time']).total_seconds()
        distances.append(distance); durations.append(time_diff if time_diff >= 0 else 0)
    segment_df['distance'] = distances; segment_df['point_duration'] = durations
    total_dist = segment_df['distance'].sum(); total_dur_seconds = 0
    if len(segment_df) > 1:
        start_time = segment_df['time'].iloc[0]; end_time = segment_df['time'].iloc[-1]
        if pd.notna(start_time) and pd.notna(end_time):
            time_diff = (end_time - start_time).total_seconds()
            total_dur_seconds = time_diff if time_diff >= 0 else 0
        else: app.logger.warning("NaT time encountered when calculating total segment duration.")
    avg_speed = (total_dist / total_dur_seconds * 3.6) if total_dur_seconds > 0 else 0
    duration_str = str(timedelta(seconds=int(total_dur_seconds)))
    return {'distance': total_dist, 'duration': duration_str, 'average_speed': avg_speed, 'duration_seconds': total_dur_seconds}


def _calculate_dataframe_speeds(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    speeds = [0.0] * len(df)
    if len(df) > 1:
        for i in range(1, len(df)):
            p1_lat, p1_lon = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
            p2_lat, p2_lon = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            t1 = df.iloc[i-1]['time']
            t2 = df.iloc[i]['time']
            if pd.notna(p1_lat) and pd.notna(p1_lon) and pd.notna(p2_lat) and pd.notna(p2_lon) and pd.notna(t1) and pd.notna(t2):
                try: dist = geodesic((p1_lat, p1_lon), (p2_lat, p2_lon)).meters
                except ValueError: dist = 0
                time_diff = (t2 - t1).total_seconds()
                if time_diff > 0.01: speeds[i] = (dist / time_diff * 3.6)
    df['calculated_speed'] = speeds
    return df


# --- Map Generation Helpers ---

# MODIFIED: Added original_segment_df parameter and logic to draw it
# MODIFIED: Draw background even for preview if original_segment_df is provided
def _build_segment_map_object(segment_df, original_index, thresholds, is_preview=False, original_segment_df=None):
    """Helper to create the Folium Map object for a segment (preview or final).
       Can optionally draw the original segment track in gray if original_segment_df is provided.
    """
    if segment_df is None or segment_df.empty: return None
    segment_df = segment_df.copy()
    segment_df['time'] = pd.to_datetime(segment_df['time'])
    segment_df_valid = segment_df.dropna(subset=['latitude', 'longitude'])
    if segment_df_valid.empty or len(segment_df_valid) < 2:
         app.logger.warning(f"Segment original_index={original_index} {'(preview)' if is_preview else ''} has insufficient valid points for map object creation.");
         return None # Return None if not enough points

    try:
        # --- Determine bounds based on ORIGINAL data if available, else current ---
        bounds_df = segment_df_valid # Default to current
        if original_segment_df is not None and not original_segment_df.empty:
            original_df_valid = original_segment_df.dropna(subset=['latitude', 'longitude'])
            if not original_df_valid.empty:
                bounds_df = original_df_valid # Use original for bounds calculation
        # Calculate map center based on the chosen bounds_df
        map_center_lat = bounds_df['latitude'].mean()
        map_center_lon = bounds_df['longitude'].mean()
        map_segment = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=15, control_scale=True)
        bounds = [(bounds_df['latitude'].min(), bounds_df['longitude'].min()), (bounds_df['latitude'].max(), bounds_df['longitude'].max())]
        # --- End Bounds ---

        speed_min_threshold = thresholds.get('speed_min', 0); speed_max_threshold = thresholds.get('speed_max', 100)
        colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
        colormap.caption = f'Speed (km/h) | Thresh: {speed_min_threshold}-{speed_max_threshold}'

        # +++ NEW: Draw Original Segment Background if provided +++
        # MODIFICATION: Removed 'and not is_preview' condition - background drawn whenever original_segment_df is provided
        is_trimmed_view = False # Flag to indicate if trimming has likely happened for title adjustment
        if original_segment_df is not None and not original_segment_df.empty:
             # A simple check if trimming has occurred (compare lengths)
             if len(original_segment_df) != len(segment_df):
                  is_trimmed_view = True # Set flag if lengths differ

             original_df_valid_bg = original_segment_df.dropna(subset=['latitude', 'longitude'])
             if not original_df_valid_bg.empty and len(original_df_valid_bg) >= 2:
                  app.logger.info(f"Drawing gray background for original segment {original_index} {'(preview)' if is_preview else ''}")
                  locations_orig = list(zip(original_df_valid_bg['latitude'], original_df_valid_bg['longitude']))
                  folium.PolyLine(
                      locations=locations_orig,
                      color='#AAAAAA',       # Gray color
                      weight=3,             # Thinner than active segment
                      opacity=0.6,          # Slightly transparent
                      dash_array='5, 5',    # Dashed line
                      tooltip='Original path (trimmed portions)'
                  ).add_to(map_segment)
        # +++ END NEW +++

        # --- Draw Active Segment (colored by speed) ---
        segment_df_with_speeds = _calculate_dataframe_speeds(segment_df.copy()) # Calculate speeds for the active segment

        for j in range(len(segment_df_with_speeds) - 1):
             p1_lat, p1_lon = segment_df_with_speeds.iloc[j]['latitude'], segment_df_with_speeds.iloc[j]['longitude']
             p2_lat, p2_lon = segment_df_with_speeds.iloc[j + 1]['latitude'], segment_df_with_speeds.iloc[j + 1]['longitude']
             if pd.isna(p1_lat) or pd.isna(p1_lon) or pd.isna(p2_lat) or pd.isna(p2_lon): continue
             point_speed = segment_df_with_speeds.iloc[j + 1].get('calculated_speed', 0)
             if pd.isna(point_speed): point_speed = 0
             color_speed = max(speed_min_threshold, min(point_speed, speed_max_threshold))
             line = folium.PolyLine(
                 locations=[(p1_lat, p1_lon), (p2_lat, p2_lon)],
                 color=colormap(color_speed),
                 weight=5,             # Keep thicker weight for active segment
                 opacity=0.9,          # Keep higher opacity
                 tooltip=f"Idx: {j} -> {j+1}, Speed: {point_speed:.1f} km/h",
             )
             line.add_to(map_segment)
        # --- End Active Segment Drawing ---

        seg_num_display = original_index + 1
        # MODIFIED: Adjust title suffix based on view type (preview or if trimming detected)
        title_suffix = " (Preview)" if is_preview else (" (Trimmed View)" if is_trimmed_view else "")
        # Markers should reflect the START and END of the ACTIVE (potentially trimmed) segment
        if not segment_df_valid.empty:
             folium.Marker(
                 location=[segment_df_valid.iloc[0]['latitude'], segment_df_valid.iloc[0]['longitude']],
                 popup=f'Start Segment {seg_num_display}{title_suffix} (Active Idx 0)',
                 icon=folium.Icon(color='green', icon='play', prefix='fa')
             ).add_to(map_segment)
             folium.Marker(
                 location=[segment_df_valid.iloc[-1]['latitude'], segment_df_valid.iloc[-1]['longitude']],
                 popup=f'End Segment {seg_num_display}{title_suffix} (Active Idx {len(segment_df_with_speeds) - 1})',
                 icon=folium.Icon(color='red', icon='stop', prefix='fa')
             ).add_to(map_segment)

        colormap.add_to(map_segment); map_segment.fit_bounds(bounds, padding=(0.01, 0.01))
        return map_segment
    except Exception as e:
        app.logger.error(f"Error building map object for segment original_index={original_index} {'(preview)' if is_preview else ''}: {e}", exc_info=True)
        return None

# MODIFIED: Added optional original_segment_df parameter
def generate_segment_map(segment_df, segment_original_index, thresholds, output_folder, original_segment_df=None):
    """Generates the standard segment map HTML file. Can include original track background."""
    # Pass the original_segment_df to the builder function
    map_object = _build_segment_map_object(segment_df, segment_original_index, thresholds, is_preview=False, original_segment_df=original_segment_df)
    if map_object is None:
        app.logger.error(f"Failed to build map object for segment {segment_original_index}. Cannot generate map file.")
        return None
    segment_filename = f'segment_map_{segment_original_index}.html'
    segment_filepath = os.path.join(output_folder, segment_filename)
    try:
        map_object.save(segment_filepath) # Save directly
        app.logger.info(f"Generated segment map file: {segment_filepath}")
        return segment_filename
    except Exception as e:
        app.logger.error(f"Error saving map file for segment original_index={segment_original_index}: {e}", exc_info=True)
        return None


def generate_raw_track_map(full_df, thresholds, output_folder):
    if full_df is None or full_df.empty or len(full_df) < 2: return None
    try:
        map_raw = folium.Map(control_scale=True)
        full_df_valid = full_df.dropna(subset=['latitude', 'longitude'])
        if full_df_valid.empty: app.logger.warning("No valid coords for raw map."); return None
        bounds = [(full_df_valid['latitude'].min(), full_df_valid['longitude'].min()), (full_df_valid['latitude'].max(), full_df_valid['longitude'].max())]
        map_raw.location = [full_df_valid['latitude'].mean(), full_df_valid['longitude'].mean()]
        speed_min_threshold = thresholds.get('speed_min', 0); speed_max_threshold = thresholds.get('speed_max', 100)
        colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
        colormap.caption = f'Speed (km/h)'
        if 'speed' not in full_df.columns:
            app.logger.warning("Speed column missing in full_df for raw map generation. Recalculating.")
            full_df = _calculate_dataframe_speeds(full_df)
            full_df.rename(columns={'calculated_speed': 'speed'}, inplace=True)

        for j in range(len(full_df) - 1):
             p1_lat, p1_lon = full_df.iloc[j]['latitude'], full_df.iloc[j]['longitude']
             p2_lat, p2_lon = full_df.iloc[j + 1]['latitude'], full_df.iloc[j + 1]['longitude']
             if pd.isna(p1_lat) or pd.isna(p1_lon) or pd.isna(p2_lat) or pd.isna(p2_lon): continue
             point_speed = full_df.iloc[j + 1].get('speed', 0);
             if pd.isna(point_speed): point_speed = 0
             color_speed = max(speed_min_threshold, min(point_speed, speed_max_threshold))
             line = folium.PolyLine( locations=[(p1_lat, p1_lon), (p2_lat, p2_lon)], color=colormap(color_speed), weight=4, opacity=0.7, tooltip=f"Idx: {j} -> {j+1}, Speed: {point_speed:.1f} km/h" )
             line.add_to(map_raw)
        if not full_df_valid.empty:
            folium.Marker(location=[full_df_valid.iloc[0]['latitude'], full_df_valid.iloc[0]['longitude']], popup='Track Start', icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(map_raw)
            folium.Marker(location=[full_df_valid.iloc[-1]['latitude'], full_df_valid.iloc[-1]['longitude']], popup='Track End', icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(map_raw)
        colormap.add_to(map_raw); map_raw.fit_bounds(bounds, padding=(0.01, 0.01))
        timestamp = int(time.time()); raw_filename = f'raw_track_map_{timestamp}.html'
        raw_filepath = os.path.join(output_folder, raw_filename); map_raw.save(raw_filepath)
        app.logger.info(f"Generated raw map file: {raw_filepath}"); return raw_filename
    except Exception as e: app.logger.error(f"Error generating raw track map: {e}", exc_info=True); return None

# --- GPX Processing Function ---
def process_gpx(file_path, speed_min_threshold, speed_max_threshold, avg_speed_filter_threshold):
    app.logger.info(f"Processing {file_path} with thresholds: min={speed_min_threshold}, max={speed_max_threshold}, avg_min={avg_speed_filter_threshold}")
    maps_output_folder = app.config['MAPS_FOLDER']; thresholds = {'speed_min': speed_min_threshold, 'speed_max': speed_max_threshold, 'avg_speed_min': avg_speed_filter_threshold}
    raw_track_map_filename = None; final_segments = []; segment_details = []; segment_html_files = []
    total_distance_meters = 0; total_duration_str = '0:00:00'
    try:
        # --- Parsing ---
        with open(file_path, 'r') as gpx_file: gpx = gpxpy.parse(gpx_file)
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.time and isinstance(point.latitude, (int, float)) and isinstance(point.longitude, (int, float)): points.append({'latitude': point.latitude, 'longitude': point.longitude, 'time': point.time})
                    elif not point.time: app.logger.warning(f"Skipping point without time: Lat={point.latitude}, Lon={point.longitude}")
                    else: app.logger.warning(f"Skipping point with invalid coordinate: Lat={point.latitude}, Lon={point.longitude}, Time={point.time}")
        if not points: raise ValueError("GPX file contains no usable points with time and valid coordinates.")
        df = pd.DataFrame(points).sort_values(by='time').reset_index(drop=True); df['time'] = pd.to_datetime(df['time'])
        df = _calculate_dataframe_speeds(df)
        df.rename(columns={'calculated_speed': 'speed'}, inplace=True)
        app.logger.info(f"Initial GPX parsing and speed calculation complete for {len(df)} points.")
        raw_track_map_filename = generate_raw_track_map(df.copy(), thresholds, maps_output_folder) # Pass copy for safety
        if not raw_track_map_filename: app.logger.warning("Failed to generate raw track map.")

        # --- Segmentation Logic ---
        def extract_segments(dataframe, speed_min, speed_max, min_segment_duration=15, speed_drop_tolerance=3, merge_tolerance=5):
            initial_segments = []; current_segment_points = []; below_threshold_time = 0; started = False
            dataframe['duration'] = dataframe['time'].diff().dt.total_seconds().fillna(0)
            for i, row in dataframe.iterrows():
                if i == 0: continue
                point_data = row.to_dict(); point_duration = row.get('duration', 0); current_speed = row.get('speed', 0)
                is_within_speed_band = speed_min <= current_speed <= speed_max
                if not started and current_speed > speed_min:
                    started = True;
                    if i > 0 and not current_segment_points: current_segment_points.append(dataframe.iloc[i-1].to_dict())
                    current_segment_points.append(point_data); below_threshold_time = 0
                elif started:
                    if is_within_speed_band: current_segment_points.append(point_data); below_threshold_time = 0
                    else:
                        below_threshold_time += point_duration
                        if below_threshold_time < speed_drop_tolerance: current_segment_points.append(point_data)
                        else:
                            if len(current_segment_points) > 1:
                                segment_df = pd.DataFrame(current_segment_points); segment_df['time'] = pd.to_datetime(segment_df['time'])
                                if pd.notna(segment_df['time'].iloc[-1]) and pd.notna(segment_df['time'].iloc[0]):
                                    segment_duration = (segment_df['time'].iloc[-1] - segment_df['time'].iloc[0]).total_seconds()
                                    if segment_duration >= min_segment_duration: initial_segments.append(segment_df)
                                else: app.logger.warning("Segment discarded during creation: NaT time in duration check.")
                            current_segment_points = []; below_threshold_time = 0; started = False
            if len(current_segment_points) > 1:
                segment_df = pd.DataFrame(current_segment_points); segment_df['time'] = pd.to_datetime(segment_df['time'])
                if pd.notna(segment_df['time'].iloc[-1]) and pd.notna(segment_df['time'].iloc[0]):
                    segment_duration = (segment_df['time'].iloc[-1] - segment_df['time'].iloc[0]).total_seconds()
                    if segment_duration >= min_segment_duration: initial_segments.append(segment_df)
                else: app.logger.warning("Last potential segment discarded: NaT time in duration check.")

            merged_segments = []
            if not initial_segments: return []
            current_merged_segment = initial_segments[0].copy(); current_merged_segment['time'] = pd.to_datetime(current_merged_segment['time'])
            for i in range(1, len(initial_segments)):
                next_segment = initial_segments[i].copy(); next_segment['time'] = pd.to_datetime(next_segment['time'])
                last_time_prev = current_merged_segment['time'].iloc[-1]; first_time_next = next_segment['time'].iloc[0]
                if pd.isna(last_time_prev) or pd.isna(first_time_next):
                    app.logger.warning(f"Skipping merge check due to NaT time."); merged_segments.append(current_merged_segment); current_merged_segment = next_segment; continue
                time_gap = (first_time_next - last_time_prev).total_seconds()
                if 0 <= time_gap <= merge_tolerance:
                    bridging_points = dataframe[(dataframe['time'] > last_time_prev) & (dataframe['time'] < first_time_next)].copy()
                    current_merged_segment = pd.concat([current_merged_segment, bridging_points, next_segment], ignore_index=True).sort_values(by='time').reset_index(drop=True)
                    app.logger.info(f"Merged segment gap of {time_gap:.1f}s, adding {len(bridging_points)} bridging points.")
                else: merged_segments.append(current_merged_segment); current_merged_segment = next_segment
            merged_segments.append(current_merged_segment)
            return merged_segments
        # --- End Segmentation Logic ---

        segments_before_avg_filter = extract_segments(df, speed_min=speed_min_threshold, speed_max=speed_max_threshold)
        app.logger.info(f"Found {len(segments_before_avg_filter)} segments before average speed filter.")
        temp_final_segments = []
        for segment_df in segments_before_avg_filter:
            if segment_df is None or segment_df.empty or len(segment_df) < 2: continue
            stats = calculate_segment_stats(segment_df.copy())
            if stats['average_speed'] >= avg_speed_filter_threshold: temp_final_segments.append(segment_df)
            else: app.logger.info(f"Filtering out segment with avg speed {stats['average_speed']:.2f} km/h (Threshold: {avg_speed_filter_threshold} km/h)")
        final_segments = temp_final_segments
        app.logger.info(f"Found {len(final_segments)} segments after average speed filter.")

        # --- Store results and generate standard maps ---
        global last_processed_data
        # Store DEEP COPIES of the initial segments
        last_processed_data['original_segments'] = [copy.deepcopy(seg) for seg in final_segments]
        # Store potentially mutable copies for current state
        last_processed_data['segments'] = [seg.copy() for seg in final_segments]
        last_processed_data['original_filename'] = os.path.basename(file_path); last_processed_data['thresholds'] = thresholds
        last_processed_data['original_file_path'] = file_path

        total_duration_seconds = 0
        if final_segments:
            segment_details_temp = []
            initial_total_dist = 0; initial_total_dur_sec = 0
            for i, segment in enumerate(final_segments):
                stats = calculate_segment_stats(segment.copy()); stats['point_count'] = len(segment); stats['original_index'] = i
                segment_details_temp.append(stats);
                initial_total_dist += stats['distance']; initial_total_dur_sec += stats['duration_seconds']
            segment_details = segment_details_temp
            last_processed_data['segment_details'] = segment_details # Store mutable list
            total_distance_meters = initial_total_dist
            total_duration_str = str(timedelta(seconds=int(initial_total_dur_sec)))

            for i, segment in enumerate(final_segments):
                # Initial map generation: Only pass the segment itself (no original background needed yet)
                html_filename = generate_segment_map(segment, i, thresholds, maps_output_folder)
                if html_filename: segment_html_files.append(html_filename)
                else: app.logger.warning(f"Failed to generate standard map for segment original_index={i}")

    except ValueError as ve: app.logger.error(f"Value error during GPX processing: {ve}", exc_info=True); return {'raw_track_map_filename': raw_track_map_filename, 'error': f'{str(ve)}', 'total_segments': 0, 'segment_details': [], 'segment_html_files': [], 'total_distance': 0, 'total_duration': '0:00:00'}
    except Exception as process_error: app.logger.error(f"Error during GPX segmentation/processing: {process_error}", exc_info=True); return {'raw_track_map_filename': raw_track_map_filename, 'error': f'Error during segment processing: {str(process_error)}', 'total_segments': 0, 'segment_details': [], 'segment_html_files': [], 'total_distance': 0, 'total_duration': '0:00:00'}

    return {
        'raw_track_map_filename': raw_track_map_filename,
        'total_distance': total_distance_meters,
        'total_duration': total_duration_str,
        'total_segments': len(final_segments),
        'segment_html_files': segment_html_files,
        'segment_details': segment_details # Return the mutable details list
    }


# --- Routes ---
@app.route('/upload', methods=['POST'])
def upload():
    global last_processed_data; last_processed_data = { 'segments': None, 'original_segments': None, 'segment_details': None, 'original_filename': None, 'original_file_path': None, 'thresholds': None }
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
        if result.get('error'): return jsonify(result), 200
        else: return jsonify(result)
    except Exception as e:
        app.logger.error(f"Unexpected error calling process_gpx for {filename}: {e}", exc_info=True);
        last_processed_data = {'segments': None, 'original_segments': None, 'segment_details': None, 'original_filename': None, 'original_file_path': None, 'thresholds': None}
        return jsonify({'error': f'An unexpected error occurred during processing: {str(e)}'}), 500

@app.route('/generate_all_segments_map', methods=['POST'])
def generate_all_segments_map():
    global last_processed_data
    app.logger.info("====== ROUTE: /generate_all_segments_map HIT ======")
    if not last_processed_data or last_processed_data.get('segments') is None or last_processed_data.get('thresholds') is None or last_processed_data.get('segment_details') is None or last_processed_data.get('original_file_path') is None:
        return jsonify({'error': 'Processed data or original file path not available.'}), 400
    try:
        data = request.get_json(); included_original_indices_req = data.get('included_indices')
        if included_original_indices_req is None or not isinstance(included_original_indices_req, list): return jsonify({'error': 'Missing or invalid included_indices list.'}), 400
        included_original_indices_set = set(map(int, included_original_indices_req))
    except (ValueError, TypeError): return jsonify({'error': 'Invalid value in included_indices.'}), 400
    app.logger.info(f"Request to generate 'All Segments' map for original_indices: {included_original_indices_set}")

    original_file_path = last_processed_data['original_file_path']
    try:
        with open(original_file_path, 'r') as gpx_file_orig: gpx_orig = gpxpy.parse(gpx_file_orig)
        points_orig = [{'latitude': p.latitude, 'longitude': p.longitude} for t in gpx_orig.tracks for s in t.segments for p in s.points if isinstance(p.latitude, (int, float)) and isinstance(p.longitude, (int, float))]
        if not points_orig: raise ValueError("Original GPX file contains no usable coordinate points.")
        full_df = pd.DataFrame(points_orig)
    except Exception as e: app.logger.error(f"Error parsing original GPX file {original_file_path}: {e}", exc_info=True); return jsonify({'error': f'Failed to parse original GPX: {str(e)}'}), 500

    segments_to_plot = []; segment_details_to_plot = []
    # Iterate through the *current* segments and details to find matches
    for i, segment_df_current in enumerate(last_processed_data.get('segments', [])):
        if i < len(last_processed_data.get('segment_details', [])):
            detail = last_processed_data['segment_details'][i]; original_index = detail.get('original_index')
            if original_index is not None and original_index in included_original_indices_set:
                segments_to_plot.append(segment_df_current.copy()); segment_details_to_plot.append(detail)
    app.logger.info(f"Found {len(segments_to_plot)} segments matching included indices for overlay.")

    try:
        full_df_valid = full_df.dropna(subset=['latitude', 'longitude'])
        if full_df_valid.empty: return jsonify({'error': 'No valid coordinates found in original track.'}), 400
        bounds = [(full_df_valid['latitude'].min(), full_df_valid['longitude'].min()), (full_df_valid['latitude'].max(), full_df_valid['longitude'].max())]
        map_center = [full_df_valid['latitude'].mean(), full_df_valid['longitude'].mean()]
        combined_map = folium.Map(location=map_center, control_scale=True)

        thresholds = last_processed_data['thresholds']
        speed_min_threshold = thresholds.get('speed_min', 0); speed_max_threshold = thresholds.get('speed_max', 100)
        overview_colormap = cm.LinearColormap(['blue', 'cyan', 'green', 'yellow', 'red'], vmin=speed_min_threshold, vmax=speed_max_threshold).to_step(n=200)
        overview_colormap.caption = f'Segment Speed (km/h) | Thresh: {speed_min_threshold}-{speed_max_threshold}'

        # Draw background full track
        locations_bg = list(zip(full_df_valid['latitude'], full_df_valid['longitude'])) # Use valid points
        if len(locations_bg) >= 2: folium.PolyLine(locations=locations_bg, color='#AAAAAA', weight=2, opacity=0.6).add_to(combined_map) # Removed options={'isBackground': True} as it might not be needed

        # Draw selected segments on top
        for i, segment_df in enumerate(segments_to_plot):
            if segment_df is None or segment_df.empty or len(segment_df) < 2: continue
            segment_df['time'] = pd.to_datetime(segment_df['time'])
            original_index = segment_details_to_plot[i]['original_index']; seg_num_display = original_index + 1
            segment_df_overlay = _calculate_dataframe_speeds(segment_df.copy())
            for j in range(len(segment_df_overlay) - 1):
                  p1_lat, p1_lon = segment_df_overlay.iloc[j]['latitude'], segment_df_overlay.iloc[j]['longitude']
                  p2_lat, p2_lon = segment_df_overlay.iloc[j + 1]['latitude'], segment_df_overlay.iloc[j + 1]['longitude']
                  if pd.isna(p1_lat) or pd.isna(p1_lon) or pd.isna(p2_lat) or pd.isna(p2_lon): continue
                  point_speed = segment_df_overlay.iloc[j + 1].get('calculated_speed', 0); point_speed = 0 if pd.isna(point_speed) else point_speed
                  color_speed = max(speed_min_threshold, min(point_speed, speed_max_threshold))
                  folium.PolyLine(locations=[(p1_lat, p1_lon), (p2_lat, p2_lon)], color=overview_colormap(color_speed), weight=5, opacity=0.9, tooltip=f"Seg {seg_num_display}, Idx: {j}->{j+1}, Speed: {point_speed:.1f} km/h").add_to(combined_map)
            segment_df_valid_plot = segment_df_overlay.dropna(subset=['latitude', 'longitude'])
            if not segment_df_valid_plot.empty:
                 if len(segment_df_valid_plot) > 0: folium.Marker( location=[segment_df_valid_plot.iloc[0]['latitude'], segment_df_valid_plot.iloc[0]['longitude']], icon=folium.Icon(color='green', icon='play', prefix='fa'), tooltip=f'Start Segment {seg_num_display}').add_to(combined_map)
                 if len(segment_df_valid_plot) > 1: folium.Marker( location=[segment_df_valid_plot.iloc[-1]['latitude'], segment_df_valid_plot.iloc[-1]['longitude']], icon=folium.Icon(color='red', icon='stop', prefix='fa'), tooltip=f'End Segment {seg_num_display}').add_to(combined_map)

        overview_colormap.add_to(combined_map); combined_map.fit_bounds(bounds, padding=(0.01, 0.01))
        output_filename = f'all_segments_map_{int(time.time())}.html'; output_filepath = os.path.join(app.config['MAPS_FOLDER'], output_filename)
        combined_map.save(output_filepath)
        app.logger.info(f"Generated 'All Segments' map: {output_filename}")
        return jsonify({'map_filename': output_filename})
    except Exception as e: app.logger.error(f"Error generating 'All Segments' map: {e}", exc_info=True); return jsonify({'error': f'Failed to generate combined map: {str(e)}'}), 500


# +++ NEW ROUTE for Trim Preview +++
# MODIFIED: Fetch and pass original_segment_df to the map builder
@app.route('/generate_trim_preview_map', methods=['POST'])
def generate_trim_preview_map():
    global last_processed_data
    app.logger.info("====== ROUTE: /generate_trim_preview_map HIT ======")
    # MODIFIED: Check for original_segments as well
    if (not last_processed_data
        or last_processed_data.get('segments') is None
        or last_processed_data.get('original_segments') is None # Check for original data
        or not last_processed_data.get('thresholds')
        or not last_processed_data.get('segment_details')):
         app.logger.error("Preview failed: Global data incomplete.")
         return jsonify({'error': 'No processed segment data available for preview.'}), 400

    try:
        data = request.get_json()
        segment_original_index = int(data['original_segment_index'])
        start_point_index = int(data['start_index'])
        end_point_index = int(data['end_index'])
        app.logger.info(f"Preview request for original_index={segment_original_index}, range=[{start_point_index}-{end_point_index}]")
    except (TypeError, KeyError, ValueError, AttributeError):
        app.logger.error(f"Invalid preview parameters: {request.data}")
        return jsonify({'error': 'Invalid or missing preview parameters.'}), 400

    # Find the current position of the segment in the list using original_index
    current_segment_position = -1
    for i, detail in enumerate(last_processed_data.get('segment_details', [])):
         if detail.get('original_index') == segment_original_index:
              if i < len(last_processed_data.get('segments', [])):
                    current_segment_position = i
                    break
              else:
                    app.logger.error(f"Segment state mismatch: Found detail for orig_idx {segment_original_index} at pos {i}, but only {len(last_processed_data.get('segments',[]))} segments exist.")
                    return jsonify({'error': 'Internal server error: Segment state mismatch.'}), 500
    if current_segment_position == -1:
        app.logger.error(f"Segment original index {segment_original_index} not found in current details list for preview.")
        return jsonify({'error': f'Segment original index {segment_original_index+1} not found for preview.'}), 404

    # Get the *current* segment data
    current_segment_df = last_processed_data['segments'][current_segment_position]
    if current_segment_df is None or current_segment_df.empty:
        return jsonify({'error': f'Segment {segment_original_index + 1} (current state) is empty.'}), 400

    # +++ NEW: Fetch the corresponding ORIGINAL segment data +++
    original_segment_df = None
    if 0 <= segment_original_index < len(last_processed_data['original_segments']):
        original_segment_df = last_processed_data['original_segments'][segment_original_index]
    else:
        app.logger.error(f"Preview failed: Original segment index {segment_original_index} out of bounds for original_segments list.")
        return jsonify({'error': 'Internal error: Cannot find original segment data for preview background.'}), 500
    if original_segment_df is None or original_segment_df.empty:
        app.logger.warning(f"Original segment data for index {segment_original_index} is empty, preview map will not have background.")
        # Proceed without original_segment_df, map builder handles None.

    # --- Validate indices against the *current* data length ---
    max_index = len(current_segment_df) - 1
    if start_point_index < 0 or end_point_index < 0 or start_point_index > max_index or end_point_index > max_index or start_point_index > end_point_index:
        app.logger.error(f"Invalid preview indices [{start_point_index}-{end_point_index}] for segment {segment_original_index}, max index {max_index}.")
        return jsonify({'error': f'Invalid preview indices [{start_point_index}-{end_point_index}]. Max index: {max_index}.'}), 400
    if (end_point_index - start_point_index + 1) < 2:
         return jsonify({'error': 'Preview segment must have at least 2 points.'}), 400 # Need >= 2 points for a map

    # Slice the current data for preview
    preview_df = current_segment_df.iloc[start_point_index : end_point_index + 1].copy().reset_index(drop=True)
    app.logger.info(f"Generating preview map with {len(preview_df)} points.")

    # --- Build the map object for the preview data ---
    # MODIFIED: Pass the fetched original_segment_df
    map_object = _build_segment_map_object(
        preview_df,
        segment_original_index,
        last_processed_data['thresholds'],
        is_preview=True,
        original_segment_df=original_segment_df # Pass original data here
    )
    if map_object is None:
        app.logger.error(f"Failed to build preview map object for segment {segment_original_index}, range {start_point_index}-{end_point_index}.")
        return jsonify({'error': 'Failed to generate preview map object.'}), 500

    # --- Generate a unique filename and save the preview map ---
    preview_filename = f'preview_map_{segment_original_index}_{start_point_index}_{end_point_index}_{uuid.uuid4().hex[:8]}.html'
    preview_filepath = os.path.join(app.config['MAPS_FOLDER'], preview_filename)

    try:
        map_object.save(preview_filepath) # Save directly
        app.logger.info(f"Generated preview map file: {preview_filepath}")
        # TODO: Consider adding cleanup logic for old preview files later
        return jsonify({'preview_map_filename': preview_filename})
    except Exception as e:
        app.logger.error(f"Error saving preview map file {preview_filename}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save preview map file.'}), 500


@app.route('/reset_segment/<int:segment_index>', methods=['POST'])
def reset_segment(segment_index):
    global last_processed_data
    app.logger.info(f"Attempting reset for original_index: {segment_index}")

    if (not last_processed_data
        or last_processed_data.get('segments') is None
        or last_processed_data.get('original_segments') is None
        or last_processed_data.get('thresholds') is None
        or last_processed_data.get('segment_details') is None):
        app.logger.error("Reset failed: Global data incomplete.")
        return jsonify({'error': 'Segment data not found for reset.'}), 400

    if not (0 <= segment_index < len(last_processed_data['original_segments'])):
        app.logger.error(f"Reset failed: Invalid original segment index {segment_index}. Max original index: {len(last_processed_data['original_segments'])-1}")
        return jsonify({'error': f'Invalid original segment index: {segment_index}.'}), 400

    # Find the current position of the segment in the 'segments' and 'segment_details' lists
    current_segment_position = -1
    for i, detail in enumerate(last_processed_data.get('segment_details', [])):
        if detail.get('original_index') == segment_index:
            # Check if this position is valid for the current 'segments' list
            if i < len(last_processed_data.get('segments', [])):
                current_segment_position = i
                break
            else:
                app.logger.error(f"Reset failed: Segment state mismatch. Found detail for orig_idx {segment_index} at pos {i}, but only {len(last_processed_data.get('segments',[]))} segments exist.")
                return jsonify({'error': 'Internal server error: Segment state mismatch.'}), 500

    if current_segment_position == -1:
        app.logger.error(f"Reset failed: Segment original index {segment_index} not found in current details list.")
        return jsonify({'error': 'Segment to reset not found in current state.'}), 404

    try:
        # --- Perform Reset ---
        # Get a DEEP COPY of the original segment data
        original_segment_df = copy.deepcopy(last_processed_data['original_segments'][segment_index])
        # Replace the segment at the current position with the original data copy
        last_processed_data['segments'][current_segment_position] = original_segment_df.copy() # Use another copy for safety
        app.logger.info(f"Segment original_index={segment_index} (current pos {current_segment_position}) reset in memory.")

        # --- Update Stats ---
        reset_stats = calculate_segment_stats(original_segment_df.copy()); # Calculate on a copy
        reset_stats['point_count'] = len(original_segment_df);
        reset_stats['original_index'] = segment_index # Ensure original index is retained

        # --- Regenerate standard map (showing only the full, reset segment) ---
        # Pass only the reset segment data, NO 'original_segment_df' argument
        map_gen_success = generate_segment_map(
            original_segment_df, # The data to display IS the original data now
            segment_index,
            last_processed_data['thresholds'],
            app.config['MAPS_FOLDER']
            # NO original_segment_df=... argument here
        )
        if map_gen_success:
            app.logger.info(f"Regenerated map for reset segment {segment_index}.")
        else:
            app.logger.error(f"Failed to regenerate map for reset segment {segment_index}")
            # Continue but inform user if possible? For now, just log.

        # --- Update Stored Details ---
        if last_processed_data.get('segment_details') and current_segment_position < len(last_processed_data['segment_details']):
             stored_detail = last_processed_data['segment_details'][current_segment_position]
             # Update all relevant stats in the stored detail dictionary
             for key, value in reset_stats.items():
                 # Ensure we don't overwrite the original_index if it's already correct
                 if key != 'original_index' or stored_detail.get('original_index') != segment_index:
                     stored_detail[key] = value
             app.logger.info(f"Updated stored details for original_index {segment_index} at pos {current_segment_position} after reset.")
        else:
             app.logger.warning(f"Could not find stored details for original_index {segment_index} at pos {current_segment_position} to update after reset.")

        # Return the updated stats for the specific segment
        return jsonify({'message': f'Segment {segment_index + 1} reset successfully.', 'reset_stats': reset_stats, 'original_segment_index': segment_index})

    except IndexError:
        app.logger.error(f"IndexError during reset for original_index {segment_index}.", exc_info=True);
        return jsonify({'error': f'Failed to reset segment {segment_index + 1}. Internal index error.'}), 500
    except Exception as e:
        app.logger.error(f"Error resetting segment original_index={segment_index}: {e}", exc_info=True);
        return jsonify({'error': f'Failed to reset segment {segment_index + 1}. Error: {str(e)}'}), 500


@app.route('/trim', methods=['POST'])
def trim_segment():
    global last_processed_data
    app.logger.info("====== ROUTE: /trim HIT ======")
    # Ensure original_segments list is also available
    if (not last_processed_data
        or last_processed_data.get('segments') is None
        or last_processed_data.get('original_segments') is None # Check for original data
        or not last_processed_data.get('thresholds')
        or not last_processed_data.get('segment_details')):
        app.logger.error("Trim failed: Global data incomplete.")
        return jsonify({'error': 'No processed segment data available.'}), 400

    try:
        data = request.get_json();
        segment_original_index = int(data['segment_index']);
        start_point_index = int(data['start_index']);
        end_point_index = int(data['end_index']);
        app.logger.info(f"Trim request for original_index={segment_original_index}, range=[{start_point_index}-{end_point_index}]")
    except (TypeError, KeyError, ValueError, AttributeError) as e:
        app.logger.error(f"Trim failed: Invalid parameters: {request.data}, Error: {e}")
        return jsonify({'error': 'Invalid or missing trim parameters.'}), 400

    # --- Find current segment position ---
    current_segment_position = -1
    for i, detail in enumerate(last_processed_data.get('segment_details', [])):
         if detail.get('original_index') == segment_original_index:
              if i < len(last_processed_data.get('segments', [])): current_segment_position = i; break
              else:
                  app.logger.error(f"Trim failed: Segment state mismatch. Found detail for orig_idx {segment_original_index} at pos {i}, but only {len(last_processed_data.get('segments',[]))} segments exist.")
                  return jsonify({'error': 'Internal server error: Segment state mismatch.'}), 500
    if current_segment_position == -1:
        app.logger.error(f"Trim failed: Segment original index {segment_original_index} not found in current details list.")
        return jsonify({'error': f'Segment original index {segment_original_index+1} not found for trimming.'}), 404

    # --- Get current and ORIGINAL segment data ---
    current_segment_df = last_processed_data['segments'][current_segment_position]
    # Validate original segment index before accessing
    if not (0 <= segment_original_index < len(last_processed_data['original_segments'])):
        app.logger.error(f"Trim failed: Original segment index {segment_original_index} out of bounds for original_segments list (len {len(last_processed_data['original_segments'])}).")
        return jsonify({'error': 'Internal error: Original segment index out of bounds.'}), 500
    # Fetch the immutable original segment data (use deepcopy for safety if needed, though should be immutable)
    original_segment_df = last_processed_data['original_segments'][segment_original_index]

    # --- Validate indices against current segment ---
    if current_segment_df is None or current_segment_df.empty:
        app.logger.error(f"Trim failed: Segment {segment_original_index + 1} (current state at pos {current_segment_position}) is empty.")
        return jsonify({'error': f'Segment {segment_original_index + 1} (current state) is empty.'}), 400
    max_index = len(current_segment_df) - 1
    if start_point_index < 0 or end_point_index < 0 or start_point_index > max_index or end_point_index > max_index or start_point_index > end_point_index:
        app.logger.error(f"Trim failed: Invalid indices [{start_point_index}-{end_point_index}] for segment {segment_original_index}, current max index {max_index}.")
        return jsonify({'error': f'Invalid trim indices [{start_point_index}-{end_point_index}]. Current max index: {max_index}. Start must be <= End.'}), 400
    if (end_point_index - start_point_index + 1) < 2:
        app.logger.error(f"Trim failed: Resulting segment would have < 2 points.")
        return jsonify({'error': 'Trimmed segment must have at least 2 points.'}), 400

    # --- Perform Trim ---
    trimmed_df = current_segment_df.iloc[start_point_index : end_point_index + 1].copy().reset_index(drop=True)
    trimmed_stats = calculate_segment_stats(trimmed_df.copy()); # Calculate on a copy
    trimmed_stats['point_count'] = len(trimmed_df);
    trimmed_stats['original_index'] = segment_original_index # Ensure original index is retained
    last_processed_data['segments'][current_segment_position] = trimmed_df # Update the current segment state
    app.logger.info(f"Segment original_index={segment_original_index} (pos {current_segment_position}) trimmed ({start_point_index}-{end_point_index}) in memory. New point count: {len(trimmed_df)}")

    # --- Update Stored Details ---
    if last_processed_data.get('segment_details') and current_segment_position < len(last_processed_data['segment_details']):
        stored_detail = last_processed_data['segment_details'][current_segment_position]
        # Update all relevant stats in the stored detail dictionary
        for key, value in trimmed_stats.items():
             # Ensure we don't overwrite the original_index if it's already correct
             if key != 'original_index' or stored_detail.get('original_index') != segment_original_index:
                stored_detail[key] = value
        app.logger.info(f"Updated stored details for original_index {segment_original_index} at pos {current_segment_position} after trim.")
    else:
        app.logger.warning(f"Could not find stored details for original_index {segment_original_index} at pos {current_segment_position} to update after trim.")

    # --- Regenerate standard map WITH original background ---
    try:
        map_gen_success = generate_segment_map(
            trimmed_df,                     # The current (trimmed) data
            segment_original_index,         # The original index
            last_processed_data['thresholds'],
            app.config['MAPS_FOLDER'],
            original_segment_df=original_segment_df # Pass the original data here for background
        )
        if map_gen_success:
             app.logger.info(f"Regenerated map for trimmed segment {segment_original_index} with original background.")
        else:
             app.logger.error(f"Failed to regenerate map for trimmed segment {segment_original_index} after trim.")
             # Continue, but the map might be stale

    except Exception as e:
        app.logger.error(f"Error during map regeneration for trimmed segment original_index={segment_original_index}: {e}", exc_info=True)
        # Continue, but map generation failed

    return jsonify({'message': 'Trim successful, server state updated, map regenerated.', 'trimmed_stats': trimmed_stats, 'original_segment_index': segment_original_index })


@app.route('/export/<int:segment_index>')
def export_segment_gpx(segment_index):
    # Exports the CURRENT state of the segment
    global last_processed_data
    app.logger.info(f"Export request for original_index: {segment_index}")

    if (not last_processed_data
        or last_processed_data.get('segments') is None
        or last_processed_data.get('segment_details') is None
        or last_processed_data.get('original_filename') is None):
         app.logger.error("Export failed: Global data incomplete.")
         return jsonify({'error': 'Segment data or original filename not found for export.'}), 400

    # Find the current position based on original_index
    current_segment_position = -1
    for i, detail in enumerate(last_processed_data.get('segment_details', [])):
         if detail.get('original_index') == segment_index:
              if i < len(last_processed_data.get('segments', [])): current_segment_position = i; break
              else:
                  app.logger.error(f"Export failed: Segment state mismatch. Found detail for orig_idx {segment_index} at pos {i}, but only {len(last_processed_data.get('segments',[]))} segments exist.")
                  return jsonify({'error': 'Internal server error: Segment state mismatch.'}), 500

    if current_segment_position == -1:
        app.logger.error(f"Export failed: Segment original index {segment_index} not found in current details list.")
        return jsonify({'error': 'Segment data not found or invalid original index.'}), 404

    # Get the segment DataFrame from the current state
    segment_df = last_processed_data['segments'][current_segment_position]
    original_filename_base = os.path.splitext(last_processed_data.get('original_filename', 'export'))[0];
    export_filename = f"{original_filename_base}_segment_{segment_index + 1}.gpx"

    if segment_df is None or segment_df.empty:
        app.logger.error(f"Export failed: Segment {segment_index+1} (current state at pos {current_segment_position}) is empty.")
        return jsonify({'error': 'Cannot export an empty segment.'}), 400

    gpx_export = gpxpy.gpx.GPX(); gpx_track = gpxpy.gpx.GPXTrack(); gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment); gpx_export.tracks.append(gpx_track)
    exported_points_count = 0
    for _, row in segment_df.iterrows():
        try:
             # Ensure time is timezone-naive UTC before export if needed, or handle timezone appropriately
             point_time = pd.to_datetime(row['time'])
             # Make timezone naive if it exists, gpxpy usually expects naive UTC
             if point_time.tzinfo is not None:
                 point_time = point_time.tz_convert(None)

             lat = float(row['latitude']); lon = float(row['longitude'])
             if pd.isna(point_time) or pd.isna(lat) or pd.isna(lon):
                 app.logger.warning(f"Skipping point with NaT/NaN for GPX export (orig_idx {segment_index}): Time={row.get('time')}, Lat={row.get('latitude')}, Lon={row.get('longitude')}")
                 continue
             point = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, time=point_time); gpx_segment.points.append(point); exported_points_count += 1
        except Exception as point_err:
            app.logger.error(f"Error processing point for GPX export (orig_idx {segment_index}): {point_err} - Data: {row}", exc_info=False); # Keep log cleaner
            continue # Skip this point

    if not gpx_segment.points:
        app.logger.error(f"Export failed: No valid points found in the segment {segment_index+1} (current state) to export.")
        return jsonify({'error': 'No valid points found in the segment to export.'}), 400

    app.logger.info(f"Exporting segment original_index={segment_index} (current state, {exported_points_count} points) to {export_filename}")
    try:
        gpx_xml = gpx_export.to_xml(prettyprint=True);
        response = make_response(gpx_xml);
        response.headers['Content-Type'] = 'application/gpx+xml';
        response.headers['Content-Disposition'] = f'attachment; filename="{export_filename}"';
        return response
    except Exception as e:
        app.logger.error(f"Error converting segment {segment_index} to GPX XML: {e}", exc_info=True);
        return jsonify({'error': 'Failed to generate GPX file.'}), 500


# --- Serve Maps and Static ---
@app.route('/maps/<path:filename>') # Use path converter for flexibility
def serve_map(filename):
    # Basic security check
    if '..' in filename or filename.startswith('/'):
        app.logger.warning(f"Attempt to access invalid map path: {filename}")
        return jsonify({'error': 'Invalid filename'}), 400
    # Allow only HTML files
    if not filename.lower().endswith('.html'):
        app.logger.warning(f"Attempt to access non-HTML file in maps folder: {filename}")
        return jsonify({'error': 'Invalid file type'}), 400

    maps_dir = app.config['MAPS_FOLDER']
    # Securely join paths and check if file exists
    try:
        # Check if the requested file exists within the designated maps folder
        if not os.path.isfile(os.path.join(maps_dir, filename)):
            raise FileNotFoundError

        response = make_response(send_from_directory(maps_dir, filename, as_attachment=False)) # Serve inline
        # Prevent caching of maps, especially previews
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate';
        response.headers['Pragma'] = 'no-cache';
        response.headers['Expires'] = '0'
        return response
    except FileNotFoundError:
        app.logger.error(f"Map file not found: {filename} in {maps_dir}")
        return jsonify({'error': 'Map file not found.'}), 404
    except Exception as e:
        app.logger.error(f"Error serving map file {filename}: {e}", exc_info=True)
        return jsonify({'error': 'Server error serving map file.'}), 500


@app.route('/favicon.ico')
def favicon():
    # Serve from static folder if you have one, otherwise return no content
    # return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    return '', 204 # No content response

@app.route('/')
def index():
    global last_processed_data
    # Reset global state whenever the main page is loaded/reloaded
    last_processed_data = { 'segments': None, 'original_segments': None, 'segment_details': None, 'original_filename': None, 'original_file_path': None, 'thresholds': None }
    app.logger.info("====== ROUTE: / HIT - Global state reset ======")
    return render_template('index.html')

def open_browser():
    # Opens the browser to the application URL
    try:
        webbrowser.open_new('http://127.0.0.1:5000')
    except Exception as e:
        app.logger.warning(f"Could not open browser automatically: {e}")

if __name__ == '__main__':
    # Setup logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout);
    console_handler.setFormatter(log_formatter)

    # Configure Flask's default logger
    app.logger.handlers.clear();
    app.logger.addHandler(console_handler);
    app.logger.setLevel(logging.INFO) # Set desired level (INFO, DEBUG, etc.)

    # Configure Werkzeug logger (handles request logs)
    werkzeug_logger = logging.getLogger('werkzeug');
    werkzeug_logger.handlers.clear()
    werkzeug_logger.addHandler(console_handler)
    werkzeug_logger.setLevel(logging.WARNING) # Keep request logs less verbose

    app.logger.info("---------------------------------------")
    app.logger.info("Starting GPX Segment Analyzer...")
    app.logger.info(f"Template folder: {app.template_folder}")
    app.logger.info(f"Static folder: {app.static_folder}")
    app.logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Maps folder: {app.config['MAPS_FOLDER']}")
    app.logger.info("---------------------------------------")

    # Open browser after a short delay
    threading.Timer(1.5, open_browser).start()

    # Run the Flask app
    # Use debug=False and use_reloader=False for packaged app or production
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

# --- END OF FILE app.py ---
