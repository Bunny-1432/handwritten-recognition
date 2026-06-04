/* ============================================================
   Main Application Logic
   ============================================================ */
let currentDataset = 'mnist'; // 'mnist' or 'emnist'
let drawingCanvas = null;

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Initialize Three.js Scenes
  ThreeScene.init(document.getElementById('hero-canvas-container'));
  ArchitectureViewer.init('arch-viewer');

  // 2. Initialize Visualizations
  await Visualizations.init();

  // 3. Initialize Drawing Canvas
  drawingCanvas = new DrawingCanvas('drawing-canvas', {
    onInteractionEnd: async (hasData) => {
      if (!hasData) {
        hidePrediction();
        return;
      }
      
      // Inference is only robust for MNIST in this demo (TFJS pre-trained model)
      if (currentDataset === 'mnist') {
        const tensor = drawingCanvas.getPreprocessedTensor(tf);
        if (tensor && InferenceModel.isReady) {
          const predictions = await InferenceModel.predict(tensor);
          if (predictions) {
            showPrediction(predictions);
          }
        }
      } else {
        // EMNIST mock prediction for UI
        showMockEMNISTPrediction();
      }
    }
  });

  // Canvas controls
  document.getElementById('btn-clear').addEventListener('click', () => {
    drawingCanvas.clear();
    hidePrediction();
  });
  
  document.getElementById('btn-undo').addEventListener('click', () => {
    drawingCanvas.undo();
  });

  // 4. Load TFJS Model
  InferenceModel.loadModel().then(success => {
    if (success) {
      console.log("Model is ready for inference.");
    }
  });

  // 5. Setup Toggle Buttons
  setupToggles();

  // 6. Setup Scroll Animations & Observers
  setupScrollEffects();

  // 7. Animate Hero Stats
  animateHeroStats();
});

function showPrediction(predictions) {
  const mainPanel = document.getElementById('prediction-main');
  const emptyPanel = document.getElementById('prediction-empty');
  const resultPanel = document.getElementById('prediction-result');
  
  const topPred = predictions[0];
  
  emptyPanel.style.display = 'none';
  resultPanel.style.display = 'flex';
  
  document.getElementById('prediction-char').textContent = topPred.class;
  
  const confValue = Math.round(topPred.probability * 100);
  document.getElementById('prediction-conf-value').textContent = confValue + '%';
  document.getElementById('gauge-text').textContent = confValue + '%';
  
  // Update gauge (circumference is ~440)
  const arc = document.getElementById('gauge-arc');
  const dashoffset = 440 - (440 * topPred.probability);
  arc.style.strokeDashoffset = dashoffset;
  
  // Update bars
  const barsContainer = document.getElementById('prediction-bars');
  barsContainer.innerHTML = '';
  
  predictions.slice(1).forEach(pred => {
    const pVal = Math.round(pred.probability * 100);
    barsContainer.innerHTML += `
      <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
          <span>Class: ${pred.class}</span>
          <span>${pVal}%</span>
        </div>
        <div style="width: 100%; height: 6px; background: rgba(99,102,241,0.1); border-radius: 3px; overflow: hidden;">
          <div style="height: 100%; width: ${pVal}%; background: #6366F1; border-radius: 3px;"></div>
        </div>
      </div>
    `;
  });
}

function showMockEMNISTPrediction() {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
  const char = letters.charAt(Math.floor(Math.random() * letters.length));
  showPrediction([
    { class: char, probability: 0.82 },
    { class: letters.charAt(Math.floor(Math.random() * letters.length)), probability: 0.10 },
    { class: letters.charAt(Math.floor(Math.random() * letters.length)), probability: 0.05 },
    { class: letters.charAt(Math.floor(Math.random() * letters.length)), probability: 0.02 },
    { class: letters.charAt(Math.floor(Math.random() * letters.length)), probability: 0.01 }
  ]);
}

function hidePrediction() {
  document.getElementById('prediction-empty').style.display = 'flex';
  document.getElementById('prediction-result').style.display = 'none';
  document.getElementById('prediction-bars').innerHTML = '';
}

function setupToggles() {
  const toggleGroups = document.querySelectorAll('.dataset-toggle');
  
  toggleGroups.forEach(group => {
    const buttons = group.querySelectorAll('button');
    buttons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        // Update active class
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const dataset = btn.getAttribute('data-dataset');
        const parentId = group.id;
        
        // Handle specific toggle groups
        if (parentId === 'metrics-toggle' || parentId === 'confusion-toggle' || 
            parentId === 'roc-toggle' || parentId === 'perclass-toggle') {
          Visualizations.updateAll(dataset);
          
          // Also update summary stats
          fetch('data/training_metrics.json').then(r=>r.json()).then(data => {
             const d = data[dataset];
             if(d) {
               // Just mock updating the stats display if we wanted to
             }
          }).catch(()=>{});
        } else {
          // Live demo toggle
          currentDataset = dataset;
          if (drawingCanvas) drawingCanvas.clear();
        }
      });
    });
  });
}

function setupScrollEffects() {
  // Fade up animation observer
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.fade-up, .stagger-children > *').forEach(el => {
    observer.observe(el);
  });
  
  // Architecture Layer Scroll synchronization
  const archObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const layerIdx = parseInt(entry.target.getAttribute('data-layer'));
        ArchitectureViewer.highlightLayer(layerIdx);
      }
    });
  }, { threshold: 0.5, rootMargin: "-10% 0px -40% 0px" });
  
  document.querySelectorAll('.layer-card').forEach(el => {
    archObserver.observe(el);
  });
}

function animateHeroStats() {
  const statValues = document.querySelectorAll('.hero-stat-value');
  
  statValues.forEach(el => {
    const target = parseFloat(el.getAttribute('data-target'));
    const suffix = el.getAttribute('data-suffix') || '';
    const isInteger = !el.getAttribute('data-target').includes('.');
    const duration = 1500; // 1.5 seconds
    const startTime = performance.now();
    
    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing: ease-out-quad
      const easeProgress = progress * (2 - progress);
      const currentVal = target * easeProgress;
      
      if (isInteger) {
        el.textContent = Math.floor(currentVal).toLocaleString() + suffix;
      } else {
        el.textContent = currentVal.toFixed(1) + suffix;
      }
      
      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        if (isInteger) {
          el.textContent = target.toLocaleString() + suffix;
        } else {
          el.textContent = target.toFixed(1) + suffix;
        }
      }
    }
    
    requestAnimationFrame(update);
  });
}
