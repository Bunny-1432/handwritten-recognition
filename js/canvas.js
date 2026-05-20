/* ============================================================
   Canvas Drawing Pad Logic
   ============================================================ */
class DrawingCanvas {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    // Config
    this.brushSize = options.brushSize || 18;
    this.brushColor = options.brushColor || '#000000';
    this.isDrawing = false;
    this.hasData = false;
    this.onInteractionEnd = options.onInteractionEnd || (() => {});
    
    // History for undo
    this.history = [];
    this.maxHistory = 10;
    
    this.init();
  }
  
  init() {
    // Setup context
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = this.brushColor;
    this.ctx.lineWidth = this.brushSize;
    
    // Fill white background (important for NN input)
    this.clear(false);
    
    // Event listeners
    this.canvas.addEventListener('mousedown', this.startPosition.bind(this));
    this.canvas.addEventListener('mouseup', this.endPosition.bind(this));
    this.canvas.addEventListener('mousemove', this.draw.bind(this));
    this.canvas.addEventListener('mouseout', this.endPosition.bind(this));
    
    // Touch support
    this.canvas.addEventListener('touchstart', this.touchStart.bind(this), {passive: false});
    this.canvas.addEventListener('touchend', this.touchEnd.bind(this), {passive: false});
    this.canvas.addEventListener('touchmove', this.touchMove.bind(this), {passive: false});
    
    this.saveState();
  }
  
  saveState() {
    if (this.history.length >= this.maxHistory) {
      this.history.shift();
    }
    this.history.push(this.canvas.toDataURL());
  }
  
  undo() {
    if (this.history.length > 1) {
      this.history.pop(); // Remove current state
      const prevDataUrl = this.history[this.history.length - 1];
      const img = new Image();
      img.onload = () => {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(img, 0, 0);
        this.hasData = this.history.length > 1; // if it's the first state, it's blank
        this.onInteractionEnd(this.hasData);
      };
      img.src = prevDataUrl;
    } else {
      this.clear();
    }
  }
  
  clear(saveToHistory = true) {
    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.hasData = false;
    if (saveToHistory) {
      this.saveState();
      this.onInteractionEnd(this.hasData);
    }
  }
  
  startPosition(e) {
    this.isDrawing = true;
    this.draw(e);
  }
  
  endPosition() {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    this.ctx.beginPath();
    this.hasData = true;
    this.saveState();
    this.onInteractionEnd(this.hasData);
  }
  
  draw(e) {
    if (!this.isDrawing) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
    
    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }
  
  touchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    this.startPosition(mouseEvent);
  }
  
  touchEnd(e) {
    e.preventDefault();
    this.endPosition();
  }
  
  touchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    this.draw(mouseEvent);
  }
  
  getImageData() {
    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }
  
  // Extract bounded box and center it on a 28x28 canvas (MNIST preprocessing)
  getPreprocessedTensor(tf) {
    if (!tf) return null;
    
    // Create a temporary canvas for scaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // We want to simulate the MNIST centering process here.
    // For simplicity in this demo, we'll draw the scaled canvas directly,
    // invert the colors (MNIST is white on black), and convert to tensor.
    tempCtx.drawImage(this.canvas, 0, 0, 28, 28);
    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;
    
    // Convert to grayscale and invert
    const grayscale = new Float32Array(28 * 28);
    for (let i = 0; i < data.length; i += 4) {
      // average RGB, then invert (255 - val), then normalize 0-1
      const avg = (data[i] + data[i+1] + data[i+2]) / 3;
      const inverted = 255 - avg;
      grayscale[i / 4] = inverted / 255.0;
    }
    
    // Create tensor: shape [1, 1, 28, 28] (batch, channel, height, width) for PyTorch
    // Or [1, 28, 28, 1] for typical TF models. Our exported model will likely expect TF format.
    // We assume [1, 28, 28, 1] here for TFJS standard, but if exported from PyTorch as ONNX->TF it might be NCHW.
    // Let's use [1, 28, 28, 1] and let inference handle reshaping if needed.
    return tf.tensor4d(grayscale, [1, 28, 28, 1]);
  }
}

window.DrawingCanvas = DrawingCanvas;
