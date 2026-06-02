#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys

# =====================================================================
# 1. GNN Model Definition (Inductive Dense GCN)
# =====================================================================

class DenseMSTSPGNN(nn.Module):
    def __init__(self, d_model=128):
        super(DenseMSTSPGNN, self).__init__()
        self.node_embed = nn.Linear(2, d_model)
        self.gcn1 = nn.Linear(d_model, d_model)
        self.gcn2 = nn.Linear(d_model, d_model)
        self.edge_proj = nn.Linear(d_model, d_model)
        self.gamma = nn.Parameter(torch.tensor([2.0]))
        self.dist_decay = nn.Parameter(torch.tensor([1.5]))
        
    def forward(self, coords, dist_matrix):
        """
        coords: (Batch, N, 2)
        dist_matrix: (Batch, N, N)
        """
        h = F.relu(self.node_embed(coords))
        A = torch.exp(-self.gamma * dist_matrix)
        row_sum = A.sum(dim=-1, keepdim=True) + 1e-6
        A_norm = A / row_sum
        h = F.relu(self.gcn1(torch.matmul(A_norm, h)))
        h = F.relu(self.gcn2(torch.matmul(A_norm, h)))
        h_proj = self.edge_proj(h)
        similarity = torch.matmul(h_proj, h.transpose(1, 2))
        edge_logits = similarity - self.dist_decay * dist_matrix
        edge_probs = torch.sigmoid(edge_logits)
        edge_probs = (edge_probs + edge_probs.transpose(1, 2)) / 2.0
        return edge_probs

# =====================================================================
# 2. 2-Opt Local Search for Clean Labels
# =====================================================================

def two_opt(tour, dist_matrix):
    num_nodes = len(tour) - 1
    best_tour = list(tour)
    improved = True
    max_iters = 5
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, num_nodes - 1):
            for j in range(i + 1, num_nodes):
                u1, v1 = best_tour[i-1], best_tour[i]
                u2, v2 = best_tour[j], best_tour[j+1]
                old_d = dist_matrix[u1, v1] + dist_matrix[u2, v2]
                new_d = dist_matrix[u1, u2] + dist_matrix[v1, v2]
                if new_d < old_d - 1e-4:
                    best_tour[i:j+1] = list(reversed(best_tour[i:j+1]))
                    improved = True
                    break
            if improved:
                break
    return best_tour

# =====================================================================
# 3. Synthetic Data Generator
# =====================================================================

def generate_synthetic_data(num_samples=100, num_nodes=150):
    coords_list = []
    dist_list = []
    labels_list = []
    for _ in range(num_samples):
        coords = np.random.rand(num_nodes, 2).astype(np.float32)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
        max_d = dist_matrix.max()
        if max_d > 0:
            dist_matrix = dist_matrix / max_d
            
        target_edges = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for _ in range(5):
            start = np.random.randint(num_nodes)
            unvisited = set(range(num_nodes))
            unvisited.remove(start)
            curr = start
            tour = [curr]
            while unvisited:
                candidates = list(unvisited)
                dists = [dist_matrix[curr, c] for c in candidates]
                probs = np.exp(-np.array(dists) * 10)
                probs /= np.sum(probs)
                next_node = np.random.choice(candidates, p=probs)
                unvisited.remove(next_node)
                tour.append(next_node)
                curr = next_node
            tour.append(start)
            tour = two_opt(tour, dist_matrix)
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i+1]
                target_edges[u, v] = 1.0
                target_edges[v, u] = 1.0
        coords_list.append(coords)
        dist_list.append(dist_matrix)
        labels_list.append(target_edges)
    return (torch.tensor(np.array(coords_list)), 
            torch.tensor(np.array(dist_list)), 
            torch.tensor(np.array(labels_list)))

# =====================================================================
# 4. Helper: Parse Cities JSON/JS
# =====================================================================

def parse_cities_file(filepath):
    """
    Parses cities coordinates from standard JSON or raw coordinates lists.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # If it's a javascript file (like convenience stores), extract JSON-like structures
        if filepath.endswith('.js'):
            # Strip variable definitions e.g., "var okmart = "
            if '=' in content:
                content = content.split('=', 1)[1].strip()
            if content.endswith(';'):
                content = content[:-1].strip()
                
        data = json.loads(content)
        
        # Extract cities array (handles {"cities": [...]}, {"OK超商": [...]}, or raw list [...])
        if isinstance(data, dict):
            # Check for common array keys
            for key in data.keys():
                if isinstance(data[key], list):
                    return data[key]
            raise ValueError("Could not find list property in JSON object.")
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("JSON file root must be a list or dictionary.")
    except Exception as e:
        print(f"Error parsing coordinates file: {e}")
        print("Please ensure the file is valid JSON (e.g., [{'x': 120.0, 'y': 25.0}, ...])")
        sys.exit(1)

# =====================================================================
# 5. CLI Execution Loop
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Dense MSTSP GNN Optimization CLI Tool")
    
    # Task modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train GNN model from scratch and export weights')
    group.add_argument('--cities_file', type=str, help='Run high-speed GNN inference on a coordinates JSON file and output probability matrix')
    
    # Training Parameters
    parser.add_argument('--d_model', type=int, default=128, help='GNN layer hidden dimension (default: 128)')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training (default: 40)')
    parser.add_argument('--train_samples', type=int, default=300, help='Number of synthetic training graphs (default: 300)')
    parser.add_argument('--val_samples', type=int, default=50, help='Number of synthetic validation graphs (default: 50)')
    parser.add_argument('--nodes', type=int, default=150, help='Number of cities in synthetic graphs (default: 150)')
    
    # Inference / Weights Loading Parameters
    parser.add_argument('--weights', type=str, default='mstsp_gnn_weights.json', help='Path to pre-trained GNN weights JSON (default: mstsp_gnn_weights.json)')
    
    # Output file
    parser.add_argument('--out', type=str, help='Output filepath (defaults: mstsp_gnn_weights.json for training, mstsp_gnn_probs.json for inference)')
    parser.add_argument('--export_onnx', action='store_true', help='Also export a standard self-contained ONNX model')

    args = parser.parse_args()
    
    # Determine default outputs
    if not args.out:
        args.out = 'mstsp_gnn_weights.json' if args.train else 'mstsp_gnn_probs.json'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using compute device: {device.type.upper()}")

    # ==========================================
    # MODE 1: GNN Model Training
    # ==========================================
    if args.train:
        print("\n--- Starting GNN Model Training Pipeline ---")
        print(f"1. Generating synthetic data ({args.train_samples} train, {args.val_samples} val, N={args.nodes} nodes)...")
        train_coords, train_dists, train_labels = generate_synthetic_data(args.train_samples, args.nodes)
        val_coords, val_dists, val_labels = generate_synthetic_data(args.val_samples, args.nodes)
        
        print(f"2. Initializing {args.d_model}-channel Dense GCN model...")
        model = DenseMSTSPGNN(d_model=args.d_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        print(f"3. Executing GNN training for {args.epochs} epochs...")
        batch_size = 10
        indices = torch.arange(train_coords.shape[0])
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            perm = torch.randperm(train_coords.shape[0])
            
            for i in range(0, train_coords.shape[0], batch_size):
                batch_idx = perm[i:i+batch_size]
                coords = train_coords[batch_idx].to(device)
                dists = train_dists[batch_idx].to(device)
                targets = train_labels[batch_idx].to(device)
                
                optimizer.zero_grad()
                preds = model(coords, dists)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * coords.shape[0]
                
            epoch_loss /= train_coords.shape[0]
            
            # Evaluate Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(val_coords.to(device), val_dists.to(device))
                val_loss = criterion(val_preds, val_labels.to(device)).item()
                
            if epoch % 5 == 0 or epoch == 1:
                print(f"   Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")
                
        # Export Model weights dictionary JSON
        print(f"\n4. Exporting GNN model weights to JSON: '{args.out}'...")
        model.eval()
        weights = {
            "node_embed_w": model.node_embed.weight.detach().cpu().numpy().tolist(),
            "node_embed_b": model.node_embed.bias.detach().cpu().numpy().tolist(),
            "gcn1_w": model.gcn1.weight.detach().cpu().numpy().tolist(),
            "gcn1_b": model.gcn1.bias.detach().cpu().numpy().tolist(),
            "gcn2_w": model.gcn2.weight.detach().cpu().numpy().tolist(),
            "gcn2_b": model.gcn2.bias.detach().cpu().numpy().tolist(),
            "edge_proj_w": model.edge_proj.weight.detach().cpu().numpy().tolist(),
            "edge_proj_b": model.edge_proj.bias.detach().cpu().numpy().tolist(),
            "gamma": float(model.gamma.detach().cpu().numpy()[0]),
            "dist_decay": float(model.dist_decay.detach().cpu().numpy()[0])
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(weights, f, indent=2)
        print("   Weights JSON exported successfully!")
        
        # Optional: ONNX Export
        if args.export_onnx:
            onnx_path = args.out.replace('.json', '.onnx')
            print(f"5. Exporting standalone ONNX model to: '{onnx_path}'...")
            try:
                dummy_coords = torch.rand(1, args.nodes, 2).to(device)
                dummy_dists = torch.rand(1, args.nodes, args.nodes).to(device)
                torch.onnx.export(
                    model,
                    (dummy_coords, dummy_dists),
                    onnx_path,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=['coords', 'dist_matrix'],
                    output_names=['edge_probabilities'],
                    external_data=False
                )
                print("   ONNX model exported successfully!")
            except Exception as e:
                print(f"   ONNX export failed: {e}")
                
        print("\n🎉 Model training and exports are fully completed!")

    # ==========================================
    # MODE 2: High-Speed GNN Inference on Custom Cities
    # ==========================================
    elif args.cities_file:
        print(f"\n--- Running High-Speed GNN Inference on: '{args.cities_file}' ---")
        
        print("1. Parsing city coordinates...")
        cities_list = parse_cities_file(args.cities_file)
        
        # Append depot as the last node, perfectly matching index.html logic!
        # Depot: x = 121.434686, y = 25.033276
        depot = {"x": 121.434686, "y": 25.033276}
        cities_list.append(depot)
        
        total_nodes = len(cities_list)
        print(f"   Successfully parsed {total_nodes - 1} cities + 1 depot (Total: {total_nodes} nodes).")
        
        # Extract x and y coordinates
        coords_raw = np.array([[c['x'], c['y']] for c in cities_list], dtype=np.float32)
        
        print("2. Preprocessing & Normalizing spatial parameters...")
        # Coordinate normalization to [0, 1]
        min_coords = coords_raw.min(axis=0)
        max_coords = coords_raw.max(axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1.0
        normalized_coords = (coords_raw - min_coords) / range_coords
        
        # Build normalized distance matrix
        diff = normalized_coords[:, np.newaxis, :] - normalized_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
        max_d = dist_matrix.max()
        if max_d > 0:
            dist_matrix = dist_matrix / max_d
            
        # Convert to PyTorch Tensors
        coords_tensor = torch.tensor(normalized_coords).unsqueeze(0).to(device) # Shape: (1, N, 2)
        dists_tensor = torch.tensor(dist_matrix).unsqueeze(0).to(device)       # Shape: (1, N, N)
        
        print(f"3. Loading pre-trained weights from: '{args.weights}'...")
        if not os.path.exists(args.weights):
            print(f"   Error: Weights file '{args.weights}' not found. Please run training first via '--train'.")
            sys.exit(1)
            
        with open(args.weights, 'r', encoding='utf-8') as f:
            weights = json.load(f)
            
        # Inspect weights dimension dynamically
        d_model = len(weights["node_embed_w"])
        print(f"   Detected weights dimension size: {d_model}")
        
        # Initialize GNN structure and load weights
        model = DenseMSTSPGNN(d_model=d_model).to(device)
        model.node_embed.weight.data = torch.tensor(weights["node_embed_w"]).to(device)
        model.node_embed.bias.data = torch.tensor(weights["node_embed_b"]).to(device)
        model.gcn1.weight.data = torch.tensor(weights["gcn1_w"]).to(device)
        model.gcn1.bias.data = torch.tensor(weights["gcn1_b"]).to(device)
        model.gcn2.weight.data = torch.tensor(weights["gcn2_w"]).to(device)
        model.gcn2.bias.data = torch.tensor(weights["gcn2_b"]).to(device)
        model.edge_proj.weight.data = torch.tensor(weights["edge_proj_w"]).to(device)
        model.edge_proj.bias.data = torch.tensor(weights["edge_proj_b"]).to(device)
        model.gamma.data = torch.tensor([weights["gamma"]]).to(device)
        model.dist_decay.data = torch.tensor([weights["dist_decay"]]).to(device)
        model.eval()
        
        print("4. Executing GNN forward pass natively...")
        with torch.no_grad():
            # Executed in parallel on GPU/CPU!
            edge_probs_tensor = model(coords_tensor, dists_tensor)
            
        # Convert back to NumPy array
        edge_probs = edge_probs_tensor.squeeze(0).cpu().numpy()
        
        # Optimize output format based on size
        # If N > 300, export as a highly compressed sparse JSON to save 99% disk space and bypass browser string limits!
        if total_nodes > 300:
            print("   Large graph detected. Compressing to sparse GNN matrix to avoid browser string size limits...")
            sparse_data = {}
            # Keep top 30 highest probability edges per node
            K = min(30, total_nodes)
            for i in range(total_nodes):
                row = edge_probs[i]
                # Get indices of top K probabilities
                top_indices = np.argsort(row)[-K:]
                node_dict = {}
                for idx in top_indices:
                    prob = float(row[idx])
                    if prob > 0.01:  # Only save meaningful probabilities
                        node_dict[str(idx)] = round(prob, 4) # Round to 4 decimals to save massive text space
                if node_dict:
                    sparse_data[str(i)] = node_dict
                    
            output_payload = {
                "sparse": True,
                "N": total_nodes,
                "data": sparse_data
            }
            print(f"   Compression complete. Sparse edges saved: {sum(len(v) for v in sparse_data.values())}")
        else:
            output_payload = edge_probs.tolist()
        
        print(f"5. Exporting pre-computed edge probability matrix to: '{args.out}'...")
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(output_payload, f)
            
        print("\n🎉 GNN Inference completed successfully!")
        print(f"👉 File size: {os.path.getsize(args.out) / (1024*1024):.2f} MB")
        print(f"💡 Success: Drag-and-drop '{args.out}' directly into the web frontend GNN uploader under 'Custom' mode!")
        print("   This bypasses browser calculations entirely, rendering instantly even for 6,000+ nodes!")

if __name__ == '__main__':
    main()
