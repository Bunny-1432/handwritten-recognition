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
    
    const originalCtx = this.ctx;
    const imgDataFull = originalCtx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const dataFull = imgDataFull.data;
    
    // 1. Find bounding box of the drawing
    let minX = this.canvas.width, minY = this.canvas.height, maxX = 0, maxY = 0;
    let hasDrawn = false;
    
    for (let y = 0; y < this.canvas.height; y++) {
      for (let x = 0; x < this.canvas.width; x++) {
        const idx = (y * this.canvas.width + x) * 4;
        const avg = (dataFull[idx] + dataFull[idx+1] + dataFull[idx+2]) / 3;
        // Background is white (255), so drawn pixels have lower values
        if (avg < 250) { 
          hasDrawn = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    
    if (!hasDrawn) return null; // Empty canvas
    
    // Add a small padding to the bounding box from the original canvas
    const padding = 15;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(this.canvas.width, maxX + padding);
    maxY = Math.min(this.canvas.height, maxY + padding);
    
    const boxWidth = maxX - minX;
    const boxHeight = Math.max(1, maxY - minY);
    
    // Create a temporary canvas to hold just the bounded digit
    const boxCanvas = document.createElement('canvas');
    boxCanvas.width = boxWidth;
    boxCanvas.height = boxHeight;
    const boxCtx = boxCanvas.getContext('2d');
    boxCtx.fillStyle = '#FFFFFF';
    boxCtx.fillRect(0, 0, boxWidth, boxHeight);
    boxCtx.drawImage(this.canvas, minX, minY, boxWidth, boxHeight, 0, 0, boxWidth, boxHeight);
    
    // 2. Scale into 20x20 preserving aspect ratio
    const scale = 20.0 / Math.max(boxWidth, boxHeight);
    const scaledWidth = boxWidth * scale;
    const scaledHeight = boxHeight * scale;
    
    // 3. Center into 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, 28, 28);
    
    const dx = (28 - scaledWidth) / 2;
    const dy = (28 - scaledHeight) / 2;
    
    // Draw the bounded image centered
    tempCtx.drawImage(boxCanvas, 0, 0, boxWidth, boxHeight, dx, dy, scaledWidth, scaledHeight);
    
    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;
    
    // Convert to grayscale and invert
    const grayscale = new Float32Array(28 * 28);
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i+1] + data[i+2]) / 3;
      const inverted = 255 - avg;
      // Normalize to 0-1
      grayscale[i / 4] = inverted / 255.0;
    }
    
    return tf.tensor4d(grayscale, [1, 28, 28, 1]);
  }
}

window.DrawingCanvas = DrawingCanvas;
