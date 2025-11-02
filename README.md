# 🚨 Emergency Vehicle Routing with Live Traffic Prediction

This is a full-stack web application that demonstrates a real-world solution to the emergency vehicle routing problem. The system uses a machine learning model to predict traffic congestion and then calculates the **fastest** route for an emergency vehicle, rather than just the _shortest_ one.

The final product is a dynamic web app built with Flask and Leaflet.js. A user can click anywhere on the map to set an "Accident" location and a "Hospital" location. The app then calculates and displays two routes:

1.  **The Naive (Shortest) Path:** The route with the shortest physical distance (blue).
2.  **Our Smart (Fastest) Path:** The route that uses our ML model's congestion predictions to find the fastest path, which is often different from the shortest.

The app also displays the ETA for both routes, proving the value of the model by showing the **total time saved**.

---

## 🛠️ Project Pipeline

This project was a complete data science pipeline, from raw data engineering to a final, deployed application.

### Step 1: Data Engineering & Graph Creation

The core of this project was turning raw, complex geospatial data into a clean, model-ready format.

1.  **Data Sourcing:** We used the **New Delhi traffic dataset**, which consists of 20 raw `probe_counts` GeoJSON files.
2.  **Graph Creation:** We parsed **24,938 road segments** from the raw files to identify **18,916 unique intersections** (nodes) that would form our graph.
3.  **Connectivity (Adjacency) Matrix:** An `adj_matrix.npy` (18916x18916) was built to represent the road network. A `1` indicates that two nodes are directly connected by a road.
4.  **Distance (Length) Matrix:** A `length_matrix.npy` (18916x18916) was built by iterating through every road segment, calculating its real-world length in meters using the `geopy` library, and storing this distance in the matrix.
5.  **Feature Engineering:**
    - The raw traffic data (nested in `segmentProbeCounts`) was extracted.
    - A critical pivot was made: we aggregated this **edge-level** data (traffic on roads) to **node-level** data (congestion at intersections) by averaging the traffic of all connecting roads.
    - We added temporal features like `hour_of_day` and `day_of_week`.
6.  **The 1-File Pivot:** Initial tests with 3+ files (over 60GB of raw data) were too large for training on commodity hardware. To create a working end-to-end prototype, a practical engineering decision was made to re-process the entire pipeline using just **1 file**. This created a smaller, manageable, but complete dataset.

### Step 2: Model Development & Evaluation

The goal was to determine if a complex, space-aware model (GCN-GRU) was truly better than a simpler, time-aware model (GRU) for this task.

- **Model A (Baseline): GRU**

  - A standard Gated Recurrent Unit (GRU) model was built.
  - This model is purely **temporal**. It predicts congestion for a node based _only_ on that single node's past traffic data. It has no "spatial" awareness of its neighbors.

- **Model B (Spatio-Temporal): GCN-GRU**
  - A Graph Convolutional Network (GCN) + GRU model was built.
  - This model is **spatio-temporal**. The GCN layer uses the `adj_matrix` to learn how congestion flows between neighboring intersections, and the GRU layer learns the time-based patterns.

Both models were trained on the same 1-file dataset and evaluated.

### Step 3: Application & Smart Routing

The best-performing model was then integrated into a Flask web application.

1.  **Backend (Flask):** An `app.py` server was built to handle API requests. It loads the ML model, the `length_matrix`, and all node maps on startup.
2.  **Frontend (Leaflet.js):** An `index.html` file provides a dynamic map. When a user clicks, the browser sends the `[lat, lon]` to the backend. The backend uses a `scipy.KDTree` to instantly find the nearest real road intersection (node) to the click.
3.  **The Smart Routing Logic:** This is the core of the app. When a user requests a path, the server:
    - **Runs Dijkstra's algorithm TWICE.**
    - **Path 1 (Naive):** Runs `dijkstra(length_matrix)` to find the _shortest_ physical path.
    - **Path 2 (Smart):**
      1.  Uses the ML model to predict congestion (a value from 0.0 to 1.0) for all 18,916 nodes.
      2.  Creates a new dynamic `cost_graph` where `cost = length * (1 + (congestion * 5.0))`. This heavily penalizes congested roads.
      3.  Runs `dijkstra(cost_graph)` to find the _fastest_ (congestion-aware) path.
    - **Calculates ETAs:** It then calculates the realistic travel time for both paths to show the user the exact time saved.

---

## 🤖 Model Comparison

Both models were trained on the 1-file dataset. The results were as follows:

| Model                 | Mean Absolute Error (MAE) | Notes                                                                      |
| :-------------------- | :-----------------------: | :------------------------------------------------------------------------- |
| **Model A (GRU)**     |       **`0.1361`**        | **Best Performer.** Simple, fast, and most accurate on this small dataset. |
| **Model B (GCN-GRU)** |         `0.1876`          | More complex and slower. Likely _underfit_ due to the small dataset.       |

**Conclusion:** For this prototype, the simpler **Model A (GRU)** was chosen as the prediction engine. It was more accurate on the limited data and significantly faster, which is ideal for a real-time web application.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Amanraj316/PathSafe
cd [YOUR_REPO_FOLDER_NAME]
```
