import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree # Fast nearest-neighbor search

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. NEW: Define Speed Constants ---
# Speeds in meters per second for calculations
MAX_SPEED_KMH = 60
MIN_SPEED_KMH = 5
AVG_SPEED_KMH = 40 # For the "naive" baseline

def kmh_to_mps(kmh):
    return (kmh * 1000) / 3600

MAX_SPEED_MPS = kmh_to_mps(MAX_SPEED_KMH)
MIN_SPEED_MPS = kmh_to_mps(MIN_SPEED_KMH)
AVG_SPEED_MPS = kmh_to_mps(AVG_SPEED_KMH)

# --- 3. Load All Models and Data (GLOBAL) ---
print("Loading models and data... Please wait.")

# Global variables
MODEL = None
LENGTH_MATRIX = None
MODEL_INPUT_DATA = None
N_NODES = 0
ID_TO_NODE_MAP = {}
NODE_KD_TREE = None
NODE_LIST_FOR_KD_TREE = []

try:
    # Load Model
    MODEL_PATH = os.path.join('models', 'model_A_gru_baseline.keras')
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    # Load Data Files
    DATA_PATH = 'data'
    LENGTH_MATRIX = np.load(os.path.join(DATA_PATH, 'length_matrix.npy'))
    X_TEST = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
    
    with open(os.path.join(DATA_PATH, 'id_to_node_map.pkl'), 'rb') as f:
        ID_TO_NODE_MAP = pickle.load(f)
    
    X_TEST_SAMPLE = X_TEST[0]
    MODEL_INPUT_DATA = np.transpose(X_TEST_SAMPLE, (1, 0, 2))
    N_NODES = MODEL_INPUT_DATA.shape[0]
    
    # Pre-compute nodes for fast search
    print("Building fast search tree for nodes...")
    temp_coord_list = []
    for node_id, coord_str in ID_TO_NODE_MAP.items():
        lon, lat = map(float, coord_str.split(','))
        temp_coord_list.append([lat, lon])
        NODE_LIST_FOR_KD_TREE.append(node_id)
        
    NODE_KD_TREE = KDTree(temp_coord_list)
    
    print(f"Data loaded. {N_NODES} nodes found and indexed.")
    print("Server is ready.")

except Exception as e:
    print(f"--- FATAL ERROR ON STARTUP ---: {e}")

# --- 4. Helper Function: Predict Congestion ---
def predict_all_congestion():
    """Uses the GRU model to predict congestion for ALL nodes at once."""
    print("Predicting congestion for all nodes...")
    predictions = MODEL.predict(MODEL_INPUT_DATA, verbose=0) # verbose=0 to silence logs
    
    congestion_vector = predictions.flatten()
    congestion_vector[congestion_vector < 0] = 0
    congestion_vector[congestion_vector > 1] = 1
    
    print("Prediction complete.")
    return congestion_vector

# --- 5. NEW: Helper Function: Calculate Path ETA ---
def calculate_path_eta(path_node_ids, congestion_vector):
    """Calculates the total travel time (in seconds) for a given path and traffic."""
    total_time_seconds = 0
    
    for i in range(len(path_node_ids) - 1):
        node_a_id = path_node_ids[i]
        node_b_id = path_node_ids[i+1]
        
        # 1. Get distance of this road segment
        distance_meters = LENGTH_MATRIX[node_a_id, node_b_id]
        
        if np.isinf(distance_meters) or distance_meters == 0:
            continue
            
        # 2. Get congestion at the *destination* node
        congestion = congestion_vector[node_b_id]
        
        # 3. Calculate effective speed for this segment
        # Linear interpolation: 0.0 congestion = MAX_SPEED, 1.0 congestion = MIN_SPEED
        speed_range = MAX_SPEED_MPS - MIN_SPEED_MPS
        effective_speed_mps = MAX_SPEED_MPS - (congestion * speed_range)
        
        # 4. Calculate time for this segment and add to total
        time_seconds = distance_meters / effective_speed_mps
        total_time_seconds += time_seconds
        
    return total_time_seconds

# --- 6. Helper Function: Reconstruct Path ---
def get_path_from_predecessors(start_node_id, end_node_id, predecessors):
    """Traces the Dijkstra predecessors to get the path list."""
    path_node_ids = []
    current_node = end_node_id
    while current_node != start_node_id:
        if current_node < 0: # -9999 indicates no path
            return None # No path found
        path_node_ids.append(int(current_node))
        current_node = predecessors[current_node]
        
    path_node_ids.append(int(start_node_id))
    path_node_ids.reverse()
    return path_node_ids

# --- 7. Main Web Page Route ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- 8. API: Get Nearest Node (Unchanged) ---
@app.route('/api/get_nearest_node', methods=['POST'])
def get_nearest_node():
    data = request.json
    lat, lon = data.get('lat'), data.get('lon')
    if lat is None or lon is None:
        return jsonify({"error": "Missing lat or lon"}), 400
        
    distance, index = NODE_KD_TREE.query([lat, lon])
    found_node_id = NODE_LIST_FOR_KD_TREE[index]
    coord_str = ID_TO_NODE_MAP[found_node_id]
    found_lon, found_lat = map(float, coord_str.split(','))
    
    return jsonify({"id": found_node_id, "lat": found_lat, "lon": found_lon})

# --- 9. API: Find the Fastest Path (HEAVILY UPDATED) ---
# --- 9. API: Find the Fastest Path (HEAVILY UPDATED) ---
@app.route('/api/find_path', methods=['POST'])
def find_path():
    data = request.json
    start_node_id = data.get('start_node_id')
    end_node_id = data.get('end_node_id')
    
    print(f"Received path request from {start_node_id} to {end_node_id}")

    # --- Step A: Get Congestion Predictions ---
    congestion_vector = predict_all_congestion()
    
    # --- Step B: Find "Naive" (Shortest) Path ---
    print("1. Finding NAIVE (shortest) path...")
    naive_distances, naive_predecessors = dijkstra(
        csgraph=LENGTH_MATRIX, # Use pure distance
        directed=False,
        indices=start_node_id,
        return_predecessors=True
    )
    naive_path_ids = get_path_from_predecessors(start_node_id, end_node_id, naive_predecessors)
    
    if naive_path_ids is None:
        print("No path found at all.")
        return jsonify({"error": "No path found between these points"}), 404

    # Calculate "Naive ETA" (actual time it would take to drive this shortest path in traffic)
    naive_eta_seconds = calculate_path_eta(naive_path_ids, congestion_vector)
    
    # --- Step C: Find "Smart" (Fastest) Path ---
    print("2. Finding SMART (fastest) path...")
    cost_graph = np.copy(LENGTH_MATRIX)
    congestion_penalty = 5.0 
    
    for node_id in range(N_NODES):
        congestion = congestion_vector[node_id]
        penalty = 1.0 + (congestion * congestion_penalty)
        cost_graph[:, node_id] *= penalty
        
    smart_distances, smart_predecessors = dijkstra(
        csgraph=cost_graph, # Use congestion-weighted cost
        directed=False,
        indices=start_node_id,
        return_predecessors=True
    )
    smart_path_ids = get_path_from_predecessors(start_node_id, end_node_id, smart_predecessors)
    
    if smart_path_ids is None:
        print("Smart path search failed (this shouldn't happen).")
        return jsonify({"error": "Smart path search failed"}), 500

    # Calculate "Smart ETA" (actual time it takes to drive our smart path)
    smart_eta_seconds = calculate_path_eta(smart_path_ids, congestion_vector)

    # --- Step D: Convert to Coordinates ---
    def get_coords_from_ids(id_list):
        coords = []
        for node_id in id_list:
            coord_str = ID_TO_NODE_MAP.get(node_id)
            if coord_str:
                lon, lat = map(float, coord_str.split(','))
                coords.append([lat, lon]) # Leaflet needs [lat, lon]
        return coords
        
    naive_path_coords = get_coords_from_ids(naive_path_ids)
    smart_path_coords = get_coords_from_ids(smart_path_ids)
            
    # --- Step E: Send Full Response ---
    time_saved_seconds = naive_eta_seconds - smart_eta_seconds
    
    print("Paths found and ETAs calculated.")
    
    # --- THIS IS THE FIX ---
    # We wrap the numbers in float() to convert them from numpy.float32
    # to standard Python floats, which jsonify can handle.
    return jsonify({
        # Smart Path (Our Model)
        "smart_path_coords": smart_path_coords,
        "smart_eta_minutes": float(smart_eta_seconds / 60.0),
        
        # Naive Path (Baseline)
        "naive_path_coords": naive_path_coords,
        "naive_eta_minutes": float(naive_eta_seconds / 60.0),
        
        # The "Diff"
        "time_saved_minutes": float(time_saved_seconds / 60.0)
    })
    # --- END OF FIX ---

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)