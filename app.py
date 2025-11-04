import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GRU, Dense, Layer, TimeDistributed, Reshape, Permute, Lambda
from flask import Flask, render_template, jsonify, request
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
import scipy.sparse as sp

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Define Speed Constants ---
MAX_SPEED_KMH = 60
MIN_SPEED_KMH = 5
def kmh_to_mps(kmh):
    return (kmh * 1000) / 3600
MAX_SPEED_MPS = kmh_to_mps(MAX_SPEED_KMH)
MIN_SPEED_MPS = kmh_to_mps(MIN_SPEED_KMH)

# --- 3. Load All Models and Data (GLOBAL) ---
print("Loading all models and data... This may take a moment.")

# Global variables
MODEL_A = None
MODEL_B = None
LENGTH_MATRIX = None
X_TEST_SAMPLE_A = None # Input for Model A
X_TEST_SAMPLE_B = None # Input for Model B
N_NODES = 0
ID_TO_NODE_MAP = {}
NODE_KD_TREE = None
NODE_LIST_FOR_KD_TREE = []
ADJ_TENSOR = None # For Model B

# --- Helper: Model B GCN Layer ---
class GCNLayer(Layer):
    def __init__(self, units, adj_matrix_tensor):
        super(GCNLayer, self).__init__()
        self.units = units
        self.adj_matrix = adj_matrix_tensor
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=False)
    def call(self, inputs):
        support = tf.matmul(inputs, self.kernel)
        output = tf.matmul(self.adj_matrix, support)
        return tf.nn.relu(output)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

# --- Helper: Model B Normalization ---
def normalize_adj_sparse(adj):
    adj = sp.csr_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32)

# --- Helper: Model B Build Function ---
def build_gcn_gru_model(timesteps, n_nodes, features):
    gcn_units = 32
    gru_units = 32
    input_shape_model = (timesteps, n_nodes, features)
    inputs = Input(shape=input_shape_model)
    gcn_out = TimeDistributed(GCNLayer(gcn_units, ADJ_TENSOR))(inputs)
    x = Permute((2, 1, 3))(gcn_out)
    x = Lambda(lambda x: tf.reshape(x, (-1, timesteps, gcn_units)))(x)
    x = GRU(gru_units, activation='tanh', trainable=False)(x)
    x = Dense(1)(x)
    outputs = Lambda(lambda x: tf.reshape(x, (-1, n_nodes)))(x)
    model = Model(inputs, outputs)
    return model

# --- Startup Data Loading ---
try:
    DATA_PATH = 'data'
    MODEL_PATH = 'models'

    # --- Load Data Files ---
    print("Loading data files...")
    LENGTH_MATRIX = np.load(os.path.join(DATA_PATH, 'length_matrix.npy'))
    X_TEST = np.load(os.path.join(DATA_PATH, 'X_test.npy')) # Shape (Samples, 12, N_Nodes, Features)
    ADJ_MATRIX = np.load(os.path.join(DATA_PATH, 'adj_matrix.npy')) # Or adj_matrix_checkpoint.npy
    
    with open(os.path.join(DATA_PATH, 'id_to_node_map.pkl'), 'rb') as f:
        ID_TO_NODE_MAP = pickle.load(f)
    
    N_NODES = ADJ_MATRIX.shape[0]
    TIMESTEPS = X_TEST.shape[1]
    FEATURES = X_TEST.shape[3]
    
    # --- Load Model A (GRU) ---
    print("Loading Model A (GRU)...")
    MODEL_A_PATH = os.path.join(MODEL_PATH, 'model_A_gru_baseline.keras')
    MODEL_A = tf.keras.models.load_model(MODEL_A_PATH)
    # Prepare input for Model A: (N_Nodes, 12, Features)
    X_TEST_SAMPLE_A = np.transpose(X_TEST[0], (1, 0, 2))
    print("Model A loaded.")

    # --- Load Model B (GCN-GRU) ---
    print("Loading Model B (GCN-GRU)...")
    print("Normalizing Adjacency Matrix...")
    adj_normalized = normalize_adj_sparse(ADJ_MATRIX)
    ADJ_TENSOR = tf.convert_to_tensor(adj_normalized.toarray(), dtype=tf.float32)
    
    print("Building GCN-GRU architecture...")
    MODEL_B = build_gcn_gru_model(TIMESTEPS, N_NODES, FEATURES)
    MODEL_B_WEIGHTS_PATH = os.path.join(MODEL_PATH, 'model_B_gcn_gru_final.weights.h5')
    MODEL_B.load_weights(MODEL_B_WEIGHTS_PATH)
    # Prepare input for Model B: (1, 12, N_Nodes, Features)
    X_TEST_SAMPLE_B = np.expand_dims(X_TEST[0], axis=0)
    print("Model B loaded.")
    
    # --- Pre-compute nodes for fast search ---
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
    print("Please check all file paths in /data and /models folders.")

# --- 4. Helper Functions: Predictions ---
def predict_congestion_model_A():
    print("Predicting with Model A...")
    predictions = MODEL_A.predict(X_TEST_SAMPLE_A, verbose=0)
    congestion_vector = predictions.flatten()
    congestion_vector[congestion_vector < 0] = 0
    congestion_vector[congestion_vector > 1] = 1
    return congestion_vector

def predict_congestion_model_B():
    print("Predicting with Model B...")
    predictions = MODEL_B.predict(X_TEST_SAMPLE_B, verbose=0)
    congestion_vector = predictions.flatten() # or predictions[0]
    congestion_vector[congestion_vector < 0] = 0
    congestion_vector[congestion_vector > 1] = 1
    return congestion_vector

# --- 5. Helper Function: Calculate Path ETA ---
def calculate_path_eta(path_node_ids, congestion_vector):
    """Calculates the total travel time (in seconds) for a given path and traffic."""
    total_time_seconds = 0
    total_distance_meters = 0
    
    for i in range(len(path_node_ids) - 1):
        node_a_id = path_node_ids[i]
        node_b_id = path_node_ids[i+1]
        
        distance_meters = LENGTH_MATRIX[node_a_id, node_b_id]
        if np.isinf(distance_meters) or distance_meters == 0:
            continue
            
        total_distance_meters += distance_meters
        congestion = congestion_vector[node_b_id]
        
        speed_range = MAX_SPEED_MPS - MIN_SPEED_MPS
        effective_speed_mps = MAX_SPEED_MPS - (congestion * speed_range)
        
        time_seconds = distance_meters / effective_speed_mps
        total_time_seconds += time_seconds
        
    return total_time_seconds, total_distance_meters

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

# --- 9. API: Find the Fastest Path (Calculates ONE path) ---
@app.route('/api/find_path', methods=['POST'])
def find_path():
    data = request.json
    start_node_id = data.get('start_node_id')
    end_node_id = data.get('end_node_id')
    method = data.get('method') # 'naive', 'model_a', or 'model_b'
    
    print(f"Received path request from {start_node_id} to {end_node_id} using method: {method}")

    # --- Step A: Get "Ground Truth" Congestion ---
    # We use Model A's predictions as the "ground truth" for calculating
    # all final ETAs, since it was our most accurate model.
    congestion_truth = predict_congestion_model_A()
    
    path_ids = None
    method_name = ""
    cost_graph = None

    # --- Step B: Find the Path based on user's choice ---
    if method == 'naive':
        print("1. Calculating NAIVE (shortest) path...")
        method_name = "Naive (Shortest) Path"
        cost_graph = LENGTH_MATRIX # Use pure distance
        
    elif method == 'model_a':
        print("1. Calculating SMART (Model A) path...")
        method_name = "Model A (GRU) Path"
        cost_graph = np.copy(LENGTH_MATRIX)
        congestion_penalty = 5.0
        for node_id in range(N_NODES):
            cost_graph[:, node_id] *= (1.0 + (congestion_truth[node_id] * congestion_penalty))

    elif method == 'model_b':
        print("1. Calculating SMART (Model B) path...")
        method_name = "Model B (GCN-GRU) Path"
        # We get Model B's predictions just to find its *proposed* path.
        congestion_model_B = predict_congestion_model_B()
        cost_graph = np.copy(LENGTH_MATRIX)
        congestion_penalty = 5.0
        for node_id in range(N_NODES):
            cost_graph[:, node_id] *= (1.0 + (congestion_model_B[node_id] * congestion_penalty))
    else:
        return jsonify({"error": "Invalid method"}), 400

    # --- Step C: Run Dijkstra on the chosen cost graph ---
    distances, predecessors = dijkstra(
        csgraph=cost_graph,
        directed=False,
        indices=start_node_id,
        return_predecessors=True
    )
    path_ids = get_path_from_predecessors(start_node_id, end_node_id, predecessors)
    
    if path_ids is None:
        print("No path found.")
        return jsonify({"error": "No path found between these points"}), 404

    # --- Step D: Calculate "Real" ETA ---
    # We calculate the ETA of the *chosen path* using the *ground truth congestion*
    print(f"2. Calculating true ETA for {method_name}...")
    eta_seconds, distance_meters = calculate_path_eta(path_ids, congestion_truth)

    # --- Step E: Convert path to Coordinates ---
    def get_coords_from_ids(id_list):
        coords = []
        for node_id in id_list:
            coord_str = ID_TO_NODE_MAP.get(node_id)
            if coord_str:
                lon, lat = map(float, coord_str.split(','))
                coords.append([lat, lon]) # Leaflet needs [lat, lon]
        return coords
        
    path_coords = get_coords_from_ids(path_ids)
            
    # --- Step F: Send Full Response ---
    print("Path found and ETA calculated.")
    return jsonify({
        "method_name": method_name,
        "path_coords": path_coords,
        "eta_minutes": float(eta_seconds / 60.0),
        "distance_km": float(distance_meters / 1000.0)
    })

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)