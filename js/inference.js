/* ============================================================
   TensorFlow.js Inference Logic
   ============================================================ */
const InferenceModel = (() => {
  let model = null;
  let isReady = false;
  let isLoading = false;
  
  // A pre-trained TFJS MNIST model from Google Storage
  const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_v1/model.json';
  
  const DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

  async function loadModel() {
    if (model) return true;
    if (isLoading) return false;
    
    isLoading = true;
    try {
      // Check if TF is loaded
      if (typeof tf === 'undefined') {
        console.error('TensorFlow.js not loaded!');
        return false;
      }
      
      console.log('Loading TFJS model...');
      model = await tf.loadLayersModel(MODEL_URL);
      isReady = true;
      isLoading = false;
      console.log('Model loaded successfully');
      
      // Warm up the model
      const dummy = tf.zeros([1, 28, 28, 1]);
      model.predict(dummy);
      dummy.dispose();
      
      return true;
    } catch (err) {
      console.error('Failed to load model:', err);
      isLoading = false;
      return false;
    }
  }
  
  async function predict(tensor) {
    if (!isReady || !model) {
      console.warn('Model not ready');
      return null;
    }
    
    try {
      // Model expects [1, 28, 28, 1]
      const prediction = model.predict(tensor);
      const probabilities = await prediction.data();
      prediction.dispose();
      
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
