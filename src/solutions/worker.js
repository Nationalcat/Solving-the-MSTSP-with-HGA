// Inline hga.js content to avoid importScripts issues
function getHGALogic() {
  return {
    createValidGenome() {
      let cities = Array.from({ length: this.cityCount }, (_, i) => i);
      this.shuffle(cities);

      if (this.salesmenCount === 1) return cities;

      // mTSP Logic for > 1 salesman
      if (this.salesmenCount > this.cityCount) {
        return this.shuffle(
          Array.from(
            { length: this.cityCount + this.salesmenCount - 1 },
            (_, i) => i
          )
        );
      }

      let possibleSplits = Array.from(
        { length: this.cityCount - 1 },
        (_, i) => i + 1
      );
      this.shuffle(possibleSplits);
      let splits = possibleSplits
        .slice(0, this.salesmenCount - 1)
        .sort((a, b) => a - b);

      let genome = [];
      let currentCityIdx = 0;
      let separatorIdx = this.cityCount;

      for (let split of splits) {
        while (currentCityIdx < split) genome.push(cities[currentCityIdx++]);
        genome.push(separatorIdx++);
      }
      while (currentCityIdx < this.cityCount)
        genome.push(cities[currentCityIdx++]);

      return genome;
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
          let nextCity = genome[i];
          if (nextCity >= this.cityCount) nextCity = -1;

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

      return shared / edges1.size;
    },

    getCanonicalSignature(genome) {
      const edges = [];
      let currentCity = -1; // -1 represents Depot

      for (let i = 0; i < genome.length; i++) {
        let nextCity = genome[i];
        if (nextCity >= this.cityCount) nextCity = -1;

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
      // Worker doesn't need to calculate metrics usually, but we keep it for compatibility
      if (!this.groundTruth || !this.groundTruth[this.problemId]) {
        this.fBeta = 0;
        this.diversity = 0;
        return;
      }
      // ... simplified for worker ...
    },

    calculateDistance(genome) {
      let totalDist = 0;
      // Use evaluation coordinates (Original or Screen)
      const useCities = this.evalCities || this.cities;
      const useDepot = this.evalDepot || this.depot;
      const shouldRound = this.problemId && this.problemId.toString().startsWith("MSTSP-");

      let currentPos = useDepot;
      let citiesInRoute = 0;
      let penalty = 0;
      const PENALTY_AMOUNT = 100000;

      for (let i = 0; i < genome.length; i++) {
        const index = genome[i];
        let nextPos;

        if (index < this.cityCount) {
          nextPos = useCities[index];
          citiesInRoute++;
        } else {
          if (citiesInRoute === 0) penalty += PENALTY_AMOUNT;
          nextPos = useDepot;
          citiesInRoute = 0;
        }

        let dist = Math.hypot(
          currentPos.x - nextPos.x,
          currentPos.y - nextPos.y
        );
        if (shouldRound) dist = Math.round(dist);
        totalDist += dist;
        
        currentPos = nextPos;
      }

      if (citiesInRoute === 0 && this.salesmenCount > 1)
        penalty += PENALTY_AMOUNT;

      let returnDist = Math.hypot(
        currentPos.x - useDepot.x,
        currentPos.y - useDepot.y
      );
      if (shouldRound) returnDist = Math.round(returnDist);
      totalDist += returnDist;

      return totalDist + penalty;
    },

    evolve() {
      // 1. Parallel Evolution for each Node
      this.nodes.forEach((node) => {
        this.evolveNode(node);
      });
      // Migration and Population Management are handled differently in Worker or not needed per tick
      this.evaluate();
    },

    managePopulations() {
       // Not used in single-node worker context
    },

    addNode() {
       // Not used in worker
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
        if (Math.random() < 0.5) this.mutate(child); 
        newPop.push(child);
      }
      node.population = newPop;
    },

    migrate() {
       // Not used in worker
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
      const start = Math.floor(Math.random() * p1.length);
      const end = Math.floor(Math.random() * (p1.length - start)) + start;
      const child = new Array(p1.length).fill(-1);

      for (let i = start; i <= end; i++) child[i] = p1[i];

      let p2Idx = 0;
      for (let i = 0; i < child.length; i++) {
        if (i >= start && i <= end) continue;
        while (child.includes(p2[p2Idx])) p2Idx++;
        child[i] = p2[p2Idx];
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
  };
}

const logic = getHGALogic();

// Mock the Vue instance 'this' context
const context = {
    cityCount: 0,
    salesmenCount: 1,
    evalCities: [],
    evalDepot: null,
    problemId: '',
    nodes: [], // Will hold only ONE node (the one this worker manages)
    popSize: 100,
    groundTruth: null,
    
    // Bind methods from logic to this context
    ...logic
};

// Bind all functions in logic to context
for (const key in logic) {
    if (typeof logic[key] === 'function') {
        context[key] = logic[key].bind(context);
    }
}

let running = false;

self.onmessage = function(e) {
    const { type, data } = e.data;
    
    if (type === 'init') {
        // Initialize context
        context.cityCount = data.cityCount;
        context.salesmenCount = data.salesmenCount;
        context.evalCities = data.evalCities;
        context.evalDepot = data.evalDepot;
        context.problemId = data.problemId;
        context.popSize = data.popSize;
        context.groundTruth = data.groundTruth;
        
        // Initialize single node for this worker
        context.nodes = [{
            id: data.nodeId,
            type: 'leaf',
            population: [],
            bestGenome: null,
            bestDistance: Infinity
        }];
        
        // Create initial population
        for (let i = 0; i < context.popSize; i++) {
            context.nodes[0].population.push(context.createValidGenome());
        }
        
        context.evaluate();
        postMessage({ 
            type: 'init_done', 
            bestDistance: context.nodes[0].bestDistance,
            bestGenome: context.nodes[0].bestGenome
        });
        
    } else if (type === 'start') {
        running = true;
        runEvolution();
    } else if (type === 'stop') {
        running = false;
    } else if (type === 'migrate_in') {
        // Receive best solutions from other islands
        // data.migrants is an array of genomes
        const node = context.nodes[0];
        const migrants = data.migrants;
        
        // Replace worst individuals
        node.population.sort((a, b) => b.fitness - a.fitness); // Best first (fitness = 1/dist)
        // So worst are at the end
        
        let replaceIdx = node.population.length - 1;
        migrants.forEach(genome => {
            if (replaceIdx >= 0) {
                node.population[replaceIdx] = genome;
                replaceIdx--;
            }
        });
    }
};

function runEvolution() {
    if (!running) return;
    
    // Run a batch of generations
    // Reduced batch size to allow more frequent communication/migration
    const batchSize = 5; 
    for(let i=0; i<batchSize; i++) {
        context.evolveNode(context.nodes[0]);
    }
    context.evaluate();
    
    postMessage({ 
        type: 'update', 
        bestGenome: context.nodes[0].bestGenome,
        bestDistance: context.nodes[0].bestDistance,
        generations: batchSize
    });
    
    // Schedule next batch immediately (macro-task) to allow message handling
    setTimeout(runEvolution, 0);
}
