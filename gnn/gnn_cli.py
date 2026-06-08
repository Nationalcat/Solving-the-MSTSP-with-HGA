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
# 5. HGA Guided Optimization in Python
# =====================================================================

def calculate_distance_matrix(cities_list, is_mstsp=False):
    N = len(cities_list)
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0.0
                continue
            p1 = cities_list[i]
            p2 = cities_list[j]
            if is_mstsp:
                d = np.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
                dist_matrix[i, j] = round(d)
            else:
                R = 6371.0  # Earth radius in km
                dLat = np.radians(p2['y'] - p1['y'])
                dLon = np.radians(p2['x'] - p1['x'])
                lat1 = np.radians(p1['y'])
                lat2 = np.radians(p2['y'])
                
                a = np.sin(dLat / 2)**2 + np.sin(dLon / 2)**2 * np.cos(lat1) * np.cos(lat2)
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist_matrix[i, j] = R * c
    return dist_matrix

def calculate_distance(tour, distance_matrix):
    N = len(tour)
    indices = np.empty(N + 2, dtype=np.int32)
    indices[0] = N
    indices[1:-1] = tour
    indices[-1] = N
    return float(distance_matrix[indices[:-1], indices[1:]].sum())

def calculate_similarity(g1, g2):
    N = len(g1)
    def get_edges(g):
        edges = set()
        curr = N
        for node in g:
            u, v = curr, node
            edges.add((u, v) if u < v else (v, u))
            curr = node
        u, v = curr, N
        edges.add((u, v) if u < v else (v, u))
        return edges

    edges1 = get_edges(g1)
    edges2 = get_edges(g2)
    shared = len(edges1.intersection(edges2))
    return shared / len(edges1)

def create_valid_genome_gnn(top_gnn_neighbors, edge_probabilities, N):
    unvisited = set(range(N))
    tour = []
    curr = N
    temp = 0.3
    
    while len(unvisited) > 0:
        neighbors = top_gnn_neighbors[curr]
        candidates = [n for n in neighbors if n in unvisited]
        
        if len(candidates) > 0:
            scores = np.array([edge_probabilities[curr, c] for c in candidates])
            max_score = np.max(scores)
            exps = np.exp((scores - max_score) / temp)
            sum_exps = np.sum(exps)
            probs = exps / sum_exps
            probs = probs / np.sum(probs) # Force sum to exactly 1.0 to satisfy np.random.choice
            next_city = np.random.choice(candidates, p=probs)
        else:
            next_city = next(iter(unvisited))
            
        unvisited.remove(next_city)
        tour.append(next_city)
        curr = next_city
        
    return tour

def crossover(p1, p2):
    N = len(p1)
    start = np.random.randint(0, N)
    end = np.random.randint(start, N)
    
    child = np.full(N, -1, dtype=np.int32)
    in_child = np.zeros(N, dtype=np.bool_)
    
    child[start:end+1] = p1[start:end+1]
    in_child[p1[start:end+1]] = True
    
    p2_idx = 0
    for i in range(N):
        if start <= i <= end:
            continue
        while in_child[p2[p2_idx]]:
            p2_idx += 1
        city = p2[p2_idx]
        child[i] = city
        in_child[city] = True
        p2_idx += 1
        
    return child.tolist()

def mutate_gnn(genome_list, edge_probabilities):
    genome = np.array(genome_list, dtype=np.int32)
    N = len(genome)
    
    min_p = float('inf')
    weakest_idx = -1
    
    u = N
    for i in range(N):
        v = genome[i]
        p = edge_probabilities[u, v]
        if p < min_p:
            min_p = p
            weakest_idx = i
        u = v
        
    last_p = edge_probabilities[u, N]
    if last_p < min_p:
        min_p = last_p
        weakest_idx = N
        
    if np.random.random() < 0.7 and weakest_idx != -1:
        i = weakest_idx
        if 0 <= i < N:
            j = np.random.randint(0, N)
            if i != j:
                start = min(i, j)
                end = max(i, j)
                genome[start:end+1] = genome[start:end+1][::-1]
                return genome.tolist()
                
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    genome[i], genome[j] = genome[j], genome[i]
    
    if np.random.random() < 0.5:
        a = np.random.randint(0, N)
        b = np.random.randint(0, N)
        start = min(a, b)
        end = max(a, b)
        genome[start:end+1] = genome[start:end+1][::-1]
        
    return genome.tolist()

class Individual:
    __slots__ = ['genome', 'distance', 'fitness']
    def __init__(self, genome, distance=0.0, fitness=0.0):
        self.genome = genome
        self.distance = distance
        self.fitness = fitness

class IslandNode:
    def __init__(self, node_id, node_type, color):
        self.id = node_id
        self.type = node_type
        self.color = color
        self.population = []
        self.best_genome = None
        self.best_distance = float('inf')

class PythonHGA:
    def __init__(self, cities_list, problem_id, edge_probs, pop_size=100, islands=4, migration_interval=50):
        self.cities_list = cities_list
        self.problem_id = problem_id
        self.edge_probs = edge_probs
        self.pop_size = pop_size
        self.islands_count = islands
        self.migration_interval = migration_interval
        
        self.total_nodes = len(cities_list)
        self.N = self.total_nodes - 1
        
        is_mstsp = problem_id.startswith("MSTSP-")
        self.distance_matrix = calculate_distance_matrix(cities_list, is_mstsp)
        
        self.top_gnn_neighbors = []
        K = min(25, self.N)
        for i in range(self.total_nodes):
            row = []
            for j in range(self.total_nodes):
                if i != j:
                    row.append((j, float(edge_probs[i, j])))
            row.sort(key=lambda x: x[1], reverse=True)
            self.top_gnn_neighbors.append([x[0] for x in row[:K]])
            
        self.nodes = []
        colors = ["#a855f7", "#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#84cc16", "#06b6d4"]
        self.nodes.append(IslandNode("island_root", "trunk", "#ffffff"))
        for i in range(islands):
            color = colors[i % len(colors)]
            self.nodes.append(IslandNode(f"island_{i}", "leaf", color))
            
        for node in self.nodes:
            node.population = []
            for _ in range(self.pop_size):
                genome = create_valid_genome_gnn(self.top_gnn_neighbors, self.edge_probs, self.N)
                node.population.append(Individual(genome))
                
        self.global_best_distance = float('inf')
        self.generation = 0
        self.best_time = 0.0
        self.best_generation = 0
        
    def evaluate(self):
        global_best = float('inf')
        for node in self.nodes:
            node_best_dist = float('inf')
            node_best_genome = None
            
            for ind in node.population:
                dist = calculate_distance(ind.genome, self.distance_matrix)
                ind.distance = dist
                ind.fitness = 1.0 / (dist + 1.0)
                
                if dist < node_best_dist:
                    node_best_dist = dist
                    node_best_genome = list(ind.genome)
                    
            if node_best_genome:
                node.best_distance = node_best_dist
                node.best_genome = node_best_genome
                
            if node.best_distance < global_best:
                global_best = node.best_distance
                
        if global_best < self.global_best_distance:
            self.global_best_distance = global_best
            self.best_generation = self.generation
            
    def evolve_node(self, node):
        new_pop = []
        
        node.population.sort(key=lambda x: x.fitness, reverse=True)
        elite_count = max(1, int(self.pop_size * 0.05))
        for i in range(elite_count):
            new_pop.append(Individual(list(node.population[i].genome), node.population[i].distance, node.population[i].fitness))
            
        def tournament_select(pop):
            k = 5
            best_ind = None
            for _ in range(k):
                ind = pop[np.random.randint(0, len(pop))]
                if best_ind is None or ind.fitness > best_ind.fitness:
                    best_ind = ind
            return best_ind
            
        while len(new_pop) < self.pop_size:
            p1 = tournament_select(node.population)
            p2 = tournament_select(node.population)
            child_genome = crossover(p1.genome, p2.genome)
            if np.random.random() < 0.5:
                child_genome = mutate_gnn(child_genome, self.edge_probs)
            new_pop.append(Individual(child_genome))
            
        node.population = new_pop
        
    def migrate(self):
        leaf_bests = []
        for node in self.nodes:
            if node.type == "leaf" and node.best_genome:
                leaf_bests.append(Individual(list(node.best_genome), node.best_distance, 1.0 / (node.best_distance + 1.0)))
                
        root_node = self.nodes[0]
        root_node.population.sort(key=lambda x: x.fitness, reverse=True)
        
        replace_idx = len(root_node.population) - 1
        for migrant in leaf_bests:
            if replace_idx >= 0:
                root_node.population[replace_idx] = migrant
                replace_idx -= 1
                
        for i in range(1, len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                nodeA = self.nodes[i]
                nodeB = self.nodes[j]
                if nodeA.best_genome and nodeB.best_genome:
                    sim = calculate_similarity(nodeA.best_genome, nodeB.best_genome)
                    if sim > 0.8:
                        worse_node = nodeA if nodeA.best_distance > nodeB.best_distance else nodeB
                        for ind in worse_node.population:
                            ind.genome = mutate_gnn(ind.genome, self.edge_probs)
                            ind.distance = calculate_distance(ind.genome, self.distance_matrix)
                            ind.fitness = 1.0 / (ind.distance + 1.0)
                            
    def run_generation(self):
        self.generation += 1
        for node in self.nodes:
            self.evolve_node(node)
            
        if self.generation > 0 and self.generation % self.migration_interval == 0:
            self.migrate()
            
        self.evaluate()

    def export_state_dict(self):
        solutions = [n.best_genome for n in self.nodes if n.best_genome]
        diversity = 0.0
        if len(solutions) >= 2:
            total_sim = 0.0
            pair_count = 0
            for i in range(len(solutions)):
                for j in range(i + 1, len(solutions)):
                    total_sim += calculate_similarity(solutions[i], solutions[j])
                    pair_count += 1
            if pair_count > 0:
                diversity = 1.0 - (total_sim / pair_count)
                
        nodes_exported = []
        for n in self.nodes:
            nodes_exported.append({
                "id": str(n.id),
                "type": str(n.type),
                "bestGenome": [int(x) for x in n.best_genome] if n.best_genome is not None else None,
                "bestDistance": float(n.best_distance) if n.best_distance != float('inf') else None,
                "color": str(n.color),
                "visible": True
            })
            
        from datetime import datetime, timezone
        state = {
            "problemId": self.problem_id,
            "generation": int(self.generation),
            "globalBestDistance": float(self.global_best_distance),
            "nodes": nodes_exported,
            "fBeta": 0.0,
            "diversity": float(diversity),
            "bestTime": float(round(self.best_time, 2)),
            "bestGeneration": int(self.best_generation),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
        return state

# =====================================================================
# 6. CLI Execution Loop
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
    parser.add_argument('--out', type=str, help='Output filepath (defaults: mstsp_gnn_weights.json, mstsp_gnn_probs.json, or mstsp_hga_state.json)')
    
    # HGA Optimization Parameters
    parser.add_argument('--hga', action='store_true', help='Run GNN-guided Hybrid Genetic Algorithm (HGA)')
    parser.add_argument('--generations', type=int, default=500, help='Number of generations for HGA (default: 500)')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size for HGA (default: 100)')
    parser.add_argument('--islands', type=int, default=4, help='Number of islands (islands) in HGA (default: 4)')
    parser.add_argument('--migration_interval', type=int, default=50, help='Migration interval for HGA (default: 50)')

    args = parser.parse_args()
    
    # Determine default outputs
    if not args.out:
        if args.train:
            args.out = f'mstsp_gnn_weights_d{args.d_model}_e{args.epochs}_n{args.nodes}.json'
        elif args.hga:
            args.out = 'mstsp_hga_state.json'
        else:
            args.out = 'mstsp_gnn_probs.json'
        
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
            edge_probs_tensor = model(coords_tensor, dists_tensor)
            
        # Convert back to NumPy array
        edge_probs = edge_probs_tensor.squeeze(0).cpu().numpy()
        
        # Determine Problem ID from cities filepath to match frontend problems keys:
        # Front-end keys are: "7-11", "全家", "OK超商", "萊爾富"
        problem_id = "7-11"
        lower_path = args.cities_file.lower()
        if 'ok' in lower_path:
            problem_id = "OK超商"
        elif 'family' in lower_path or '全家' in lower_path:
            problem_id = "全家"
        elif 'hilife' in lower_path or '萊爾富' in lower_path:
            problem_id = "萊爾富"
        elif 'seven' in lower_path or '7-11' in lower_path or '711' in lower_path:
            problem_id = "7-11"
        elif 'mstsp-' in lower_path:
            import re
            match = re.search(r'mstsp-\w+', lower_path)
            if match:
                problem_id = match.group(0).upper()
            else:
                problem_id = "MSTSP-CUSTOM"
        
        # ==========================================
        # SUBMODE A: Run GNN-Guided HGA solver
        # ==========================================
        if args.hga:
            import time
            print(f"\n--- Initializing GNN-Guided HGA Solver ---")
            print(f"   Problem ID: {problem_id}")
            print(f"   Population Size: {args.pop_size} | Islands: {args.islands} | Generations: {args.generations}")
            
            hga_solver = PythonHGA(
                cities_list=cities_list,
                problem_id=problem_id,
                edge_probs=edge_probs,
                pop_size=args.pop_size,
                islands=args.islands,
                migration_interval=args.migration_interval
            )
            
            print("   Running HGA optimization...")
            start_time = time.time()
            
            # Initial evaluation
            hga_solver.evaluate()
            
            for gen in range(1, args.generations + 1):
                hga_solver.run_generation()
                
                # Print progress every 50 generations or at the end
                if gen % 50 == 0 or gen == 1 or gen == args.generations:
                    elapsed = time.time() - start_time
                    print(f"   Gen {gen:04d}/{args.generations:04d} | Best Distance: {hga_solver.global_best_distance:.3f} | Elapsed: {elapsed:.2f}s")
                    
            end_time = time.time()
            hga_solver.best_time = end_time - start_time
            
            print(f"\n🎉 HGA Optimization Completed in {hga_solver.best_time:.2f}s!")
            print(f"👉 Best Distance Found: {hga_solver.global_best_distance:.4f}")
            print(f"👉 Found at Generation: {hga_solver.best_generation}")
            
            # Export state dictionary
            state_data = hga_solver.export_state_dict()
            
            print(f"\n5. Exporting HGA state JSON to: '{args.out}'...")
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
                
            print(f"👉 State successfully exported! File size: {os.path.getsize(args.out) / 1024:.2f} KB")
            print(f"💡 Success: Open the web frontend, click 'Import', select '{args.out}', and visualize the routes instantly!")
            
        # ==========================================
        # SUBMODE B: Standard precomputed matrix output
        # ==========================================
        else:
            # Optimize output format based on size
            if total_nodes > 300:
                print("   Large graph detected. Compressing to sparse GNN matrix to avoid browser string size limits...")
                sparse_data = {}
                K = min(30, total_nodes)
                for i in range(total_nodes):
                    row = edge_probs[i]
                    top_indices = np.argsort(row)[-K:]
                    node_dict = {}
                    for idx in top_indices:
                        prob = float(row[idx])
                        if prob > 0.01:
                            node_dict[str(idx)] = round(prob, 4)
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

