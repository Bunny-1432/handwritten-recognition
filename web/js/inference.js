/* ============================================================
   TensorFlow.js Inference Logic
   ============================================================ */
const InferenceModel = (() => {
  let model = null;
  let isReady = false;
  let isLoading = false;
  
  // List of working TFJS MNIST model URLs (in priority order)
  // These are actively maintained by TensorFlow team or hosted with permissive CORS
  const MODEL_URLS = [
    'data/model/model.json',                                                                 // Locally hosted model (offline/pre-trained)
    'https://raw.githubusercontent.com/google/tfjs-mnist-workshop/master/model/model.json',  // Reliable remote fallback (supports file:// CORS)
  ];
  
  const DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

  async function loadModel() {
    if (model) return true;
    if (isLoading) return false;
    
    isLoading = true;
    
    // Try each model URL until one succeeds
    for (const modelUrl of MODEL_URLS) {
      try {
        // Check if TF is loaded
        if (typeof tf === 'undefined') {
          console.error('TensorFlow.js not loaded!');
          isLoading = false;
          return false;
        }
        
        console.log(`Loading TFJS model from: ${modelUrl}`);
        model = await tf.loadLayersModel(modelUrl);
        isReady = true;
        isLoading = false;
        console.log('✓ Model loaded successfully');
        
        // Warm up the model
        // Dynamically get the expected shape (e.g., [null, 784] or [null, 28, 28, 1])
        const inputShape = model.inputs[0].shape;
        const warmupShape = [1, ...inputShape.slice(1)];
        const dummy = tf.zeros(warmupShape);
        model.predict(dummy);
        dummy.dispose();
        
        return true;
      } catch (err) {
        console.warn(`✗ Failed to load from ${modelUrl}:`, err.message);
        // Continue to next URL
      }
    }
    
    // All URLs failed
    console.error('Failed to load model from any available source');
    console.warn('To fix this, consider exporting your trained PyTorch model to TFJS format and hosting it locally.');
    isLoading = false;
    return false;
  }
  
  async function predict(tensor) {
    if (!isReady || !model) {
      console.warn('Model not ready');
      return null;
    }
    
    try {
      // Dynamically reshape tensor to match model's expected input shape
      const inputShape = model.inputs[0].shape; // e.g. [null, 784] or [null, 28, 28, 1]
      
      // Determine total elements expected (excluding batch dim)
      const expectedElements = inputShape.slice(1).reduce((a, b) => a * b, 1);
      const tensorElements = tensor.shape.slice(1).reduce((a, b) => a * b, 1);
      
      let reshapedTensor;
      if (expectedElements === tensorElements) {
        // Shapes are compatible — reshape to exactly what the model expects
        const targetShape = [1, ...inputShape.slice(1)];
        reshapedTensor = tensor.reshape(targetShape);
      } else {
        console.error(`Shape mismatch: model expects ${expectedElements} elements, tensor has ${tensorElements}`);
        return null;
      }
      
      const prediction = model.predict(reshapedTensor);
      const probabilities = await prediction.data();
      prediction.dispose();
      reshapedTensor.dispose();
      
      // Get top 5
      const top5 = Array.from(probabilities)
        .map((prob, index) => ({
          class: DIGIT_CLASSES[index],
          probability: prob,
          index: index
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 5);
        
      return top5;
    } catch (err) {
      console.error('Prediction error:', err);
      return null;
    }
  }

  return {
    loadModel,
    predict,
    get isReady() { return isReady; }
  };
})();
