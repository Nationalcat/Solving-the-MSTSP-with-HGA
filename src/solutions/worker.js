importScripts('hga.js');

const logic = getHGALogic();

// Mock the Vue instance 'this' context
const context = {
    cityCount: 0,
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
