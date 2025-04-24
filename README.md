
## Requirements

*   Python 3.x (tested : 3.10)
*   Required Python libraries:
    *   `Flask`
    *   `gpxpy`
    *   `geopy`
    *   `pandas`
    *   `numpy`
    *   `folium`
    *   `branca`

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Or download the ZIP file
    cd gpx-segment-analyzer        # Navigate into the project directory
    ```

2.  **Create a Virtual Environment (Recommended):**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install Flask gpxpy geopy pandas numpy folium branca
    ```
    *Alternatively, if you create a `requirements.txt` file (recommended):*
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Make sure you are in the project's root directory (`gpx-segment-analyzer/`).
2.  If you created a virtual environment, ensure it is activated.
3.  Run the Flask application:
    ```bash
    python app.py
    ```
4.  The script will start the Flask development server. It should automatically open `http://127.0.0.1:5000` in your default web browser after a short delay. If not, manually navigate to that URL.
5.  The application will create `uploads/` and `maps/` directories if they don't exist.

## Usage

1.  **Choose File:** Click the "Choose GPX File" button and select a `.gpx` file containing track data (latitude, longitude, and time information for points are required).
2.  **Set Thresholds:** Adjust the speed thresholds in the right-hand panel:
    *   **Min Point Speed (km/h):** Points slower than this are generally excluded.
    *   **Max Point Speed (km/h):** Points faster than this are generally excluded.
    *   **Min Avg Segment Speed (km/h):** Segments identified using the point speeds will be discarded entirely if their average speed is below this value.
3.  **Process GPX:** Click the green "Process GPX" button. A status message will appear below the button.
4.  **View Results:**
    *   **Summary:** Once processing is complete, the "Summary" section (below the navigation buttons) updates with the total distance, duration, and count for segments that passed *all* filters.
    *   **Segment Details Table:** If segments are found, a table appears below the map showing details for each. Click on a row in this table to view the map for that specific segment.
    *   **Map:** The map area below the table displays the results. Initially, it shows the "All Segments" view (all filtered segments plotted together). Use the "Prev", "Next", and "All Segments" buttons or click table rows to navigate between the combined map and individual segment maps. The track colors correspond to point speed based on the Min/Max Point Speed thresholds set.

## File Structure
