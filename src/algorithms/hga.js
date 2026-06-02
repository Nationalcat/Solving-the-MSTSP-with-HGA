function getHGALogic() {
  return {
    createValidGenome() {
      let cities = Array.from({ length: this.cityCount }, (_, i) => i);
      this.shuffle(cities);
      return cities;
    },

    shuffle(array) {
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
      return array;
    },

    evaluate() {
      let globalBest = Infinity;

      // Evaluate each node independently
      this.nodes.forEach((node) => {
        let nodeBestDist = Infinity;
        let nodeBestGenome = null;

        node.population.forEach((genome) => {
          const dist = this.calculateDistance(genome);
          genome.fitness = 1 / (dist + 1);
          genome.distance = dist;

          if (dist < nodeBestDist) {
            nodeBestDist = dist;
            nodeBestGenome = [...genome];
          }
        });

        // Update node best
        if (nodeBestGenome) {
          node.bestDistance = nodeBestDist;
          node.bestGenome = nodeBestGenome;
        }

        // Track global best (from any node)
        if (node.bestDistance < globalBest) {
          globalBest = node.bestDistance;
        }
      });

      if (globalBest < Infinity) {
        this.globalBestDistance = globalBest;
      }
    },

    // Calculate similarity between two genomes based on shared edges
    calculateSimilarity(g1, g2) {
      if (!g1 || !g2 || g1.length !== g2.length) return 0;

      const getEdges = (genome) => {
        const edges = new Set();
        let currentCity = -1; // -1 is Depot

        for (let i = 0; i < genome.length; i++) {
          const nextCity = genome[i];

          const u = currentCity;
          const v = nextCity;
          edges.add(u < v ? `${u}_${v}` : `${v}_${u}`);

          currentCity = nextCity;
        }
        // Return to depot
        const u = currentCity;
        const v = -1;
        edges.add(u < v ? `${u}_${v}` : `${v}_${u}`);
        return edges;
      };

      const edges1 = getEdges(g1);
      const edges2 = getEdges(g2);

      let shared = 0;
      edges2.forEach((e) => {
        if (edges1.has(e)) shared++;
      });

      // Jaccard Similarity or just Overlap?
      // Usually for TSP edges: shared / total_edges_in_one_solution
      // Since both have same number of edges (N + m), we can divide by size.
      return shared / edges1.size;
    },

    getCanonicalSignature(genome) {
      const edges = [];
      let currentCity = -1; // -1 represents Depot

      for (let i = 0; i < genome.length; i++) {
        const nextCity = genome[i];

        const u = currentCity;
        const v = nextCity;
        edges.push(u < v ? `${u}_${v}` : `${v}_${u}`);

        currentCity = nextCity;
      }
      // Return to depot
      const u = currentCity;
      const v = -1;
      edges.push(u < v ? `${u}_${v}` : `${v}_${u}`);

      edges.sort();
      return edges.join("|");
    },

    calculateMetrics() {
      if (!this.groundTruth || !this.groundTruth[this.problemId]) {
        this.fBeta = 0;
        this.diversity = 0;
        return;
      }

      const gt = this.groundTruth[this.problemId];
      // Increased tolerance to handle float precision vs integer ground truth
      const tolerance = 1.0;

      // 1. Update Found Optima
      this.nodes.forEach((node) => {
        if (
          node.bestGenome &&
          Math.abs(node.bestDistance - gt.optLength) < tolerance
        ) {
          const sig = this.getCanonicalSignature(node.bestGenome);
          this.foundOptima.add(sig);
        }
      });

      // 2. Calculate F-Beta
      const R = this.foundOptima.size / gt.optCount;
      const P = 1.0;
      const betaSq = 0.3;

      if (P + R === 0) {
        this.fBeta = 0;
      } else {
        this.fBeta = ((1 + betaSq) * P * R) / (betaSq * P + R);
      }

      // 3. Calculate Diversity (1 - Average Similarity of best solutions)
      let totalSim = 0;
      let pairCount = 0;
      const solutions = this.nodes.map((n) => n.bestGenome).filter((g) => g);

      if (solutions.length < 2) {
        this.diversity = 0;
      } else {
        for (let i = 0; i < solutions.length; i++) {
          for (let j = i + 1; j < solutions.length; j++) {
            totalSim += this.calculateSimilarity(solutions[i], solutions[j]);
            pairCount++;
          }
        }
        const avgSim = totalSim / pairCount;
        this.diversity = 1 - avgSim;
      }
    },

    calculateDistance(genome) {
      const N = this.cityCount;
      
      // Pre-compute the distance matrix once per problem
      if (!this.distanceMatrix) {
        const useCities = this.evalCities || this.cities;
        const useDepot = this.evalDepot || this.depot;
        const isMSTSP = this.problemId && this.problemId.toString().startsWith("MSTSP-");
        
        const getDist = (p1, p2) => {
          if (isMSTSP) {
              let d = Math.hypot(p1.x - p2.x, p1.y - p2.y);
              return Math.round(d);
          } else {
              const R = 6371; // Earth radius in km
              const dLat = (p2.y - p1.y) * Math.PI / 180;
              const dLon = (p2.x - p1.x) * Math.PI / 180;
              const lat1 = p1.y * Math.PI / 180;
              const lat2 = p2.y * Math.PI / 180;

              const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                        Math.sin(dLon/2) * Math.sin(dLon/2) * Math.cos(lat1) * Math.cos(lat2);
              const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
              return R * c;
          }
        };

        const allPoints = [...useCities, useDepot]; // Depot is at index N
        const numPoints = N + 1;
        const dists = Array.from({ length: numPoints }, () => new Float32Array(numPoints));
        for (let i = 0; i < numPoints; i++) {
          for (let j = 0; j < numPoints; j++) {
            dists[i][j] = getDist(allPoints[i], allPoints[j]);
          }
        }
        this.distanceMatrix = dists;
      }

      // Fast evaluation using distance lookup matrix
      let totalDist = 0;
      const N_depot = N; // Depot is at index N
      let currentIdx = N_depot;

      for (let i = 0; i < genome.length; i++) {
        const nextIdx = genome[i];
        totalDist += this.distanceMatrix[currentIdx][nextIdx];
        currentIdx = nextIdx;
      }

      totalDist += this.distanceMatrix[currentIdx][N_depot];

      return totalDist;
    },

    evolve() {
      // 1. Parallel Evolution for each Node
      this.nodes.forEach((node) => {
        this.evolveNode(node);
      });

      // 2. Hierarchical Migration (Bottom-Up)
      if (
        this.generation > 0 &&
        this.generation % this.migrationInterval === 0
      ) {
        this.migrate();
      }

      // 3. Dynamic Population Management (Add/Delete)
      this.managePopulations();

      this.evaluate();
      this.calculateMetrics();
    },

    managePopulations() {
      // 1. Prune Duplicates (Delete)
      const nodesToRemove = new Set();
      const leaves = this.nodes.filter((n) => n.type === "leaf");

      for (let i = 0; i < leaves.length; i++) {
        for (let j = i + 1; j < leaves.length; j++) {
          const n1 = leaves[i];
          const n2 = leaves[j];
          if (nodesToRemove.has(n1.id) || nodesToRemove.has(n2.id)) continue;

          if (n1.bestGenome && n2.bestGenome) {
            const sim = this.calculateSimilarity(n1.bestGenome, n2.bestGenome);
            if (sim > 0.95) {
              // Very similar, remove the worse one
              if (n1.bestDistance < n2.bestDistance) {
                nodesToRemove.add(n2.id);
              } else {
                nodesToRemove.add(n1.id);
              }
            }
          }
        }
      }

      if (nodesToRemove.size > 0) {
        this.nodes = this.nodes.filter((n) => !nodesToRemove.has(n.id));
      }

      // 2. Spawn New (Add)
      // Add a new node every 20 generations if we have room (< 20 nodes)
      // This encourages exploration
      // Only add nodes if we have ground truth (user request: limit to 2 for unknown problems)
      if (
        this.groundTruth &&
        this.groundTruth[this.problemId] &&
        this.generation > 0 &&
        this.generation % 20 === 0 &&
        this.nodes.length < 20
      ) {
        this.addNode();
      }

      this.solutionCount = this.nodes.length - 1; // Update count for UI
    },

    addNode() {
      if (!this.solutionColors) return;
      const id =
        this.nodes.length > 0 ? Math.max(...this.nodes.map((n) => n.id)) + 1 : 1;
      const color = this.solutionColors[id % this.solutionColors.length];

      const newNode = {
        id: id,
        type: "leaf",
        population: [],
        bestGenome: null,
        bestDistance: Infinity,
        color: color,
        visible: true,
      };

      for (let i = 0; i < this.popSize; i++) {
        if (this.edgeProbabilities) {
          newNode.population.push(this.createValidGenomeGNN(this.edgeProbabilities));
        } else {
          newNode.population.push(this.createValidGenome());
        }
      }

      this.nodes.push(newNode);
    },

    evolveNode(node) {
      const newPop = [];

      // Elitism
      node.population.sort((a, b) => b.fitness - a.fitness);
      const eliteCount = Math.floor(this.popSize * 0.05);
      for (let i = 0; i < eliteCount; i++)
        newPop.push([...node.population[i]]);

      // Standard GA Loop
      while (newPop.length < this.popSize) {
        const p1 = this.tournamentSelect(node.population);
        const p2 = this.tournamentSelect(node.population);
        let child = this.crossover(p1, p2);
        if (Math.random() < 0.5) {
          if (this.edgeProbabilities) {
            this.mutateGNN(child, this.edgeProbabilities);
          } else {
            this.mutate(child);
          }
        }
        newPop.push(child);
      }

      // Apply Clearing (Niching) within the node?
      // Or is Clearing applied globally?
      // The paper usually applies Clearing to maintain diversity *within* a node
      // OR to ensure nodes are distinct.
      // Here we apply a simple clearing: if individuals are too similar to the best, penalize them?
      // Actually, standard GA within node is fine. The distinctness comes from independent evolution + migration logic.

      node.population = newPop;
    },

    migrate() {
      // Bottom-Up Migration: Leaves -> Root
      // 1. Identify best from each Leaf
      const leafBests = [];
      this.nodes.forEach((node) => {
        if (node.type === "leaf" && node.bestGenome) {
          leafBests.push({
            genome: [...node.bestGenome],
            fitness: 1 / (node.bestDistance + 1),
          });
        }
      });

      // 2. Send to Root
      const rootNode = this.nodes[0];
      // Replace worst individuals in Root with Leaf Bests
      rootNode.population.sort((a, b) => b.fitness - a.fitness); // Best first

      let replaceIdx = rootNode.population.length - 1;
      leafBests.forEach((migrant) => {
        if (replaceIdx >= 0) {
          rootNode.population[replaceIdx] = migrant.genome;
          replaceIdx--;
        }
      });

      // 3. Top-Down? (Optional, some HGA do this)
      // Root sends its best back to leaves to guide them?
      // The paper emphasizes Bottom-Up for collecting solutions.
      // But to help leaves converge, Root can also seed them.
      // Let's implement a "Clearing" check here:
      // If two leaves are too similar, force one to diverge (restart/mutate).

      for (let i = 1; i < this.nodes.length; i++) {
        for (let j = i + 1; j < this.nodes.length; j++) {
          const nodeA = this.nodes[i];
          const nodeB = this.nodes[j];

          if (nodeA.bestGenome && nodeB.bestGenome) {
            const sim = this.calculateSimilarity(
              nodeA.bestGenome,
              nodeB.bestGenome
            );
            if (sim > 0.8) {
              // Too similar
              // Penalize the worse one (reset its population to explore elsewhere)
              const worseNode =
                nodeA.bestDistance > nodeB.bestDistance ? nodeA : nodeB;
              // Heavy mutation on the worse node's population to force divergence
              worseNode.population.forEach((g) => this.mutate(g)); // Mutate everyone
            }
          }
        }
      }
    },

    tournamentSelect(pop) {
      const k = 5;
      let best = null;
      for (let i = 0; i < k; i++) {
        const ind = pop[Math.floor(Math.random() * pop.length)];
        if (!best || ind.fitness > best.fitness) best = ind;
      }
      return best;
    },

    crossover(p1, p2) {
      const N = p1.length;
      const start = Math.floor(Math.random() * N);
      const end = Math.floor(Math.random() * (N - start)) + start;
      const child = new Array(N).fill(-1);
      const inChild = new Uint8Array(N);

      for (let i = start; i <= end; i++) {
        const city = p1[i];
        child[i] = city;
        inChild[city] = 1;
      }

      let p2Idx = 0;
      for (let i = 0; i < N; i++) {
        if (i >= start && i <= end) continue;
        while (inChild[p2[p2Idx]]) {
          p2Idx++;
        }
        const city = p2[p2Idx];
        child[i] = city;
        inChild[city] = 1;
        p2Idx++;
      }
      return child;
    },

    mutate(genome) {
      const i = Math.floor(Math.random() * genome.length);
      const j = Math.floor(Math.random() * genome.length);
      [genome[i], genome[j]] = [genome[j], genome[i]];

      if (Math.random() < 0.5) {
        const a = Math.floor(Math.random() * genome.length);
        const b = Math.floor(Math.random() * genome.length);
        const start = Math.min(a, b);
        const end = Math.max(a, b);
        let left = start,
          right = end;
        while (left < right) {
          [genome[left], genome[right]] = [genome[right], genome[left]];
          left++;
          right--;
        }
      }
    },

    computeBuiltinGNNHeuristic(cities, depot) {
      const N = cities.length;
      const allNodes = [...cities, depot]; // Index 0 to N-1 are cities, N is depot
      const distMatrix = Array.from({ length: N + 1 }, () => new Float32Array(N + 1));
      
      // Calculate distances and find average
      let totalDist = 0;
      let count = 0;
      const isMSTSP = this.problemId && this.problemId.toString().startsWith("MSTSP-");

      const getDist = (p1, p2) => {
        if (isMSTSP) {
            return Math.hypot(p1.x - p2.x, p1.y - p2.y);
        } else {
            const R = 6371; // Earth radius in km
            const dLat = (p2.y - p1.y) * Math.PI / 180;
            const dLon = (p2.x - p1.x) * Math.PI / 180;
            const lat1 = p1.y * Math.PI / 180;
            const lat2 = p2.y * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                      Math.sin(dLon/2) * Math.sin(dLon/2) * Math.cos(lat1) * Math.cos(lat2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }
      };

      for (let i = 0; i <= N; i++) {
        for (let j = i + 1; j <= N; j++) {
          const d = getDist(allNodes[i], allNodes[j]);
          distMatrix[i][j] = d;
          distMatrix[j][i] = d;
          totalDist += d;
          count++;
        }
      }
      
      const avgDist = count > 0 ? totalDist / count : 1.0;
      const edgeProbabilities = Array.from({ length: N + 1 }, () => new Float32Array(N + 1));
      
      // Compute GAT-like edge attention scores
      for (let i = 0; i <= N; i++) {
        let rowSum = 0;
        const tempScores = new Float32Array(N + 1);
        for (let j = 0; j <= N; j++) {
          if (i === j) {
            tempScores[j] = 0;
          } else {
            // Distance decay: closer nodes get higher scores
            const score = Math.exp(-3.5 * (distMatrix[i][j] / avgDist));
            tempScores[j] = score;
            rowSum += score;
          }
        }
        
        // Row softmax/normalization
        if (rowSum > 0) {
          for (let j = 0; j <= N; j++) {
            edgeProbabilities[i][j] = tempScores[j] / rowSum;
          }
        }
      }
      
      // Make it symmetric: P = (P + P^T) / 2
      for (let i = 0; i <= N; i++) {
        for (let j = i + 1; j <= N; j++) {
          const symScore = (edgeProbabilities[i][j] + edgeProbabilities[j][i]) / 2;
          edgeProbabilities[i][j] = symScore;
          edgeProbabilities[j][i] = symScore;
        }
      }
      
      return edgeProbabilities;
    },

    createValidGenomeGNN(edgeProbabilities) {
      const N = this.cityCount;
      
      // Pre-compute the Top-K GNN neighbors for each node if not already done
      if (!this.topGNNNeighbors) {
        this.topGNNNeighbors = [];
        const K = Math.min(25, N); // Look at top 25 neighbors
        for (let i = 0; i <= N; i++) {
          const row = [];
          for (let j = 0; j <= N; j++) {
            if (i !== j) {
              row.push({ node: j, prob: edgeProbabilities[i][j] });
            }
          }
          // Sort descending by probability
          row.sort((a, b) => b.prob - a.prob);
          // Keep top K nodes
          this.topGNNNeighbors.push(row.slice(0, K).map(item => item.node));
        }
      }

      const unvisited = new Set(Array.from({ length: N }, (_, i) => i));
      const tour = [];
      let curr = N; // Start from Depot (index N)
      
      const temp = 0.3; // Softmax temperature
      
      while (unvisited.size > 0) {
        // Get pre-computed top neighbors for the current node
        const neighbors = this.topGNNNeighbors[curr];
        const candidates = [];
        for (let i = 0; i < neighbors.length; i++) {
          const neighbor = neighbors[i];
          if (unvisited.has(neighbor)) {
            candidates.push(neighbor);
          }
        }
        
        let nextCity = -1;
        
        if (candidates.length > 0) {
          // Perform softmax and roulette-wheel selection over these few candidates
          const scores = candidates.map(c => edgeProbabilities[curr][c]);
          const maxScore = Math.max(...scores);
          let sumExps = 0;
          const exps = new Float32Array(candidates.length);
          for (let i = 0; i < candidates.length; i++) {
            exps[i] = Math.exp((scores[i] - maxScore) / temp);
            sumExps += exps[i];
          }
          
          const r = Math.random();
          let cumulative = 0;
          nextCity = candidates[candidates.length - 1];
          for (let i = 0; i < candidates.length; i++) {
            cumulative += exps[i] / sumExps;
            if (r <= cumulative) {
              nextCity = candidates[i];
              break;
            }
          }
        } else {
          // Fallback: choose a random unvisited node quickly
          const iterator = unvisited.values();
          nextCity = iterator.next().value;
        }
        
        unvisited.delete(nextCity);
        tour.push(nextCity);
        curr = nextCity;
      }
      
      return tour;
    },

    mutateGNN(genome, edgeProbabilities) {
      const N = genome.length;
      
      // Single-pass O(N) minimum search to find weakest edge
      // Avoids object allocations and sorting overhead entirely
      let minP = Infinity;
      let weakestIdx = -1;
      
      let u = N; // Start at Depot (index N)
      for (let i = 0; i < N; i++) {
        const v = genome[i];
        const p = edgeProbabilities[u][v];
        if (p < minP) {
          minP = p;
          weakestIdx = i;
        }
        u = v;
      }
      
      // Return edge to depot
      const lastP = edgeProbabilities[u][N];
      if (lastP < minP) {
        minP = lastP;
        weakestIdx = N;
      }
      
      // With 70% probability, try to optimize around the weakest edge
      if (Math.random() < 0.7 && weakestIdx !== -1) {
        const i = weakestIdx;
        if (i < N && i >= 0) {
          const j = Math.floor(Math.random() * N);
          if (i !== j) {
            const start = Math.min(i, j);
            const end = Math.max(i, j);
            
            // Reverse segment [start, end] in place
            let left = start, right = end;
            while (left < right) {
              const tmp = genome[left];
              genome[left] = genome[right];
              genome[right] = tmp;
              left++;
              right--;
            }
            return;
          }
        }
      }
      
      // Fallback to standard fast mutation
      const i = Math.floor(Math.random() * N);
      const j = Math.floor(Math.random() * N);
      const tmp = genome[i];
      genome[i] = genome[j];
      genome[j] = tmp;
      
      if (Math.random() < 0.5) {
        const a = Math.floor(Math.random() * N);
        const b = Math.floor(Math.random() * N);
        const start = Math.min(a, b);
        const end = Math.max(a, b);
        let left = start, right = end;
        while (left < right) {
          const t = genome[left];
          genome[left] = genome[right];
          genome[right] = t;
          left++;
          right--;
        }
      }
    },
  };
}
