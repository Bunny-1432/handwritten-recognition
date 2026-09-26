/* ============================================================
   inference.js
   Handles ONNX Runtime Web inference for handwritten character recognition.
   Uses the global `ort` object from onnxruntime-web CDN script.
   ============================================================ */

const ModelInference = (() => {
    let session = null;
    let currentDataset = 'mnist';
    let numClasses = 10;
    let modelPath = 'data/model/mnist.onnx';

    // Dataset constants for Z-score normalization (matching train.py)
    const configs = {
        mnist: {
            numClasses: 10,
            path: 'data/model/mnist.onnx',
            mean: 0.1307,
            std: 0.3081
        },
        emnist: {
            numClasses: 62,
            path: 'data/model/emnist.onnx',
            mean: 0.1751,
            std: 0.3332
        }
    };

    async function loadModel(dataset = 'mnist') {
        currentDataset = dataset;
        const config = configs[dataset];
        numClasses = config.numClasses;
        modelPath = config.path;

        try {
            // Configure ONNX Runtime WebAssembly backend
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

            session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm']
            });
            console.log(`[Inference] ✅ Loaded ${dataset} model from ${modelPath}`);
            console.log('[Inference] Input names:', session.inputNames);
            console.log('[Inference] Output names:', session.outputNames);
            return true;
        } catch (e) {
            console.error(`[Inference] ❌ Failed to load model "${dataset}":`, e);
            throw e;
        }
    }

    function preprocess(grayscaleData) {
        const config = configs[currentDataset];

        // Apply Z-score normalization: (x - mean) / std
        // This MUST match train.py transforms.Normalize(mean, std)
        const normalized = new Float32Array(grayscaleData.length);
        for (let i = 0; i < grayscaleData.length; i++) {
            normalized[i] = (grayscaleData[i] - config.mean) / config.std;
        }

        // ONNX expects shape [batch, channel, height, width] -> [1, 1, 28, 28] (NCHW)
        // canvas.js returns flat Float32Array of 784 values (28*28), which we feed directly
        return new ort.Tensor('float32', normalized, [1, 1, 28, 28]);
    }

    async function predict(grayscaleData) {
        if (!session) {
            throw new Error('[Inference] Model not loaded. Call loadModel() first.');
        }

        const inputTensor = preprocess(grayscaleData);
        console.log('[Inference] Input tensor shape:', inputTensor.dims);
        console.log('[Inference] Input sample values (first 5):', Array.from(inputTensor.data).slice(0, 5));

        try {
            // Use the actual input name from the ONNX graph ('input')
            const inputName = session.inputNames[0];
            const feeds = {};
            feeds[inputName] = inputTensor;

            const results = await session.run(feeds);

            // Get output using the actual output name from the ONNX graph ('output')
            const outputName = session.outputNames[0];
            const outputData = results[outputName].data;
            console.log('[Inference] Raw logits (first 5):', Array.from(outputData).slice(0, 5));

            // Softmax to get probabilities
            const probs = softmax(Array.from(outputData));
            const prediction = probs.indexOf(Math.max(...probs));

            console.log(`[Inference] Prediction: ${prediction} (${getClassLabel(prediction)}) | Confidence: ${(probs[prediction] * 100).toFixed(1)}%`);

            return {
                prediction,
                confidence: probs[prediction],
                probabilities: probs
            };
        } catch (e) {
            console.error('[Inference] ❌ Prediction error:', e);
            throw e;
        }
    }

    function softmax(logits) {
        const maxLogit = Math.max(...logits);
        const scores = logits.map(l => Math.exp(l - maxLogit));
        const sum = scores.reduce((a, b) => a + b, 0);
        return scores.map(s => s / sum);
    }

    function getClassLabel(index) {
        if (currentDataset === 'mnist') {
            return index.toString();
        } else {
            const labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
            return labels[index] || 'Unknown';
        }
    }

    // Public API
    return {
        loadModel,
        predict,
        getClassLabel,
        get currentDataset() { return currentDataset; },
        get numClasses() { return numClasses; }
    };
})();

// Expose globally
window.ModelInference = ModelInference;
