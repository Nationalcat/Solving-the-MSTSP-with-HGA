import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

# =====================================================================
# 1. Define a Dense GCN (Graph Convolutional Network) for TSP
# =====================================================================

class DenseMSTSPGNN(nn.Module):
    def __init__(self, d_model=64):
        super(DenseMSTSPGNN, self).__init__()
        # Node input: 2D coordinates (x, y)
        self.node_embed = nn.Linear(2, d_model)
        
        # GCN layers
        self.gcn1 = nn.Linear(d_model, d_model)
        self.gcn2 = nn.Linear(d_model, d_model)
        
        # Bilinear Edge projection
        self.edge_proj = nn.Linear(d_model, d_model)
        
        # Distance decay scale (learnable parameters)
        self.gamma = nn.Parameter(torch.tensor([2.0]))
        self.dist_decay = nn.Parameter(torch.tensor([1.5]))
        
    def forward(self, coords, dist_matrix):
        """
        coords: (Batch, N, 2) - 2D coordinates in [0, 1]
        dist_matrix: (Batch, N, N) - Pre-normalized distance matrix in [0, 1]
        """
        # 1. Embed node features: (Batch, N, d_model)
        h = F.relu(self.node_embed(coords))
        
        # 2. Compute Adjacency using distance-decay: (Batch, N, N)
        # Closer nodes have higher adjacency weights
        # dist_matrix is already pre-normalized by the caller to [0, 1]
        A = torch.exp(-self.gamma * dist_matrix)
        
        # Row-normalize Adjacency
        row_sum = A.sum(dim=-1, keepdim=True) + 1e-6
        A_norm = A / row_sum
        
        # 3. Dense GCN Propagation
        h = F.relu(self.gcn1(torch.matmul(A_norm, h)))
        h = F.relu(self.gcn2(torch.matmul(A_norm, h)))
        
        # 4. Bilinear Edge Predictor: (Batch, N, N)
        h_proj = self.edge_proj(h) # (Batch, N, d_model)
        similarity = torch.matmul(h_proj, h.transpose(1, 2)) # (Batch, N, N)
        
        # Edge logits combine node representation similarity and absolute distance decay
        edge_logits = similarity - self.dist_decay * dist_matrix
        edge_probs = torch.sigmoid(edge_logits)
        
        # Make the output matrix symmetric: P = (P + P^T) / 2
        edge_probs = (edge_probs + edge_probs.transpose(1, 2)) / 2.0
        return edge_probs


# =====================================================================
# 2. Synthetic Data Generation (Nearest Neighbor + Multimodal Simulation)
# =====================================================================

def two_opt(tour, dist_matrix):
    """
    Optimizes a TSP tour using a fast 2-opt local search.
    """
    num_nodes = len(tour) - 1
    best_tour = list(tour)
    
    improved = True
    max_iters = 5  # Quick optimization iteration cap for fast data generation
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


def generate_synthetic_data(num_samples=100, num_nodes=150):
    """
    Generates synthetic TSP coordinates, distance matrices, and labels.
    Labels are predicted edge probabilities from simulation of multimodal tours.
    """
    coords_list = []
    dist_list = []
    labels_list = []
    
    for _ in range(num_samples):
        # Generate random 2D coordinates in [0, 1]
        coords = np.random.rand(num_nodes, 2).astype(np.float32)
        
        # Compute pairwise Euclidean distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
        
        # Normalize distance matrix to [0, 1] pre-training to keep ONNX model clean
        max_d = dist_matrix.max()
        if max_d > 0:
            dist_matrix = dist_matrix / max_d
        
        # Simulate multiple optimal/suboptimal tours using Nearest Neighbor with random starting points
        # to construct a "union of good edges" for the MSTSP target labels.
        target_edges = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Run 5 different randomized constructive runs to get multiple diverse solutions
        for _ in range(5):
            start = np.random.randint(num_nodes)
            unvisited = set(range(num_nodes))
            unvisited.remove(start)
            curr = start
            tour = [curr]
            
            while unvisited:
                # Greedy choice with noise (softmax-like greedy to mimic multimodal exploration)
                candidates = list(unvisited)
                dists = [dist_matrix[curr, c] for c in candidates]
                # Softmax probability with low temperature
                probs = np.exp(-np.array(dists) * 10)
                probs /= np.sum(probs)
                next_node = np.random.choice(candidates, p=probs)
                
                unvisited.remove(next_node)
                tour.append(next_node)
                curr = next_node
            tour.append(start)
            
            # Run fast 2-opt to make the synthetic tours highly optimal and clean
            tour = two_opt(tour, dist_matrix)
            
            # Record edges in target matrix
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
# 3. Training Loop
# =====================================================================

def main():
    print("Initializing Dense MSTSP GCN Training Pipeline...")
    num_nodes = 150
    d_model = 128
    epochs = 40
    batch_size = 10
    
    # Generate Training and Validation Data
    print("Generating synthetic multimodal training data...")
    train_coords, train_dists, train_labels = generate_synthetic_data(300, num_nodes)
    val_coords, val_dists, val_labels = generate_synthetic_data(50, num_nodes)
    
    # Initialize GNN model, optimizer, loss
    model = DenseMSTSPGNN(d_model=d_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        # Shuffle dataset manually for simplicity
        indices = torch.randperm(train_coords.shape[0])
        
        for i in range(0, train_coords.shape[0], batch_size):
            batch_indices = indices[i:i + batch_size]
            coords = train_coords[batch_indices]
            dists = train_dists[batch_indices]
            targets = train_labels[batch_indices]
            
            optimizer.zero_grad()
            preds = model(coords, dists)
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * coords.shape[0]
            
        epoch_loss /= train_coords.shape[0]
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_coords, val_dists)
            val_loss = criterion(val_preds, val_labels).item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")

    # =====================================================================
    # 4. Save Model Weights to JSON (100% Robust & Browser-compatible)
    # =====================================================================
    print("\nSaving trained GCN model weights to JSON...")
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
    
    json_filename = "mstsp_gnn_weights.json"
    with open(json_filename, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Successfully saved model weights to JSON: '{json_filename}'!")

    # =====================================================================
    # 5. Export Model to ONNX format (Optional Side Output)
    # =====================================================================
    print("\nExporting model to ONNX format (Static shape: 150 nodes)...")
    try:
        # Example input for tracing (Batch=1, N=150, Features)
        dummy_coords = torch.rand(1, num_nodes, 2)
        dummy_dists = torch.rand(1, num_nodes, num_nodes)
        
        onnx_filename = "mstsp_gnn.onnx"
        torch.onnx.export(
            model,
            (dummy_coords, dummy_dists),
            onnx_filename,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['coords', 'dist_matrix'],
            output_names=['edge_probabilities'],
            external_data=False
        )
        print(f"Successfully exported model to ONNX: '{onnx_filename}'!")
    except Exception as e:
        print(f"ONNX export skipped/failed: {e}")

if __name__ == '__main__':
    main()
