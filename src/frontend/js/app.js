/* ============================================================
   app.js - Main Application Logic
   Uses global ModelInference (from inference.js),
   DrawingCanvas (from canvas.js), ThreeScene, ArchitectureViewer,
   Visualizations (from their respective scripts).
   ============================================================ */

// Use the global ModelInference instance (exposed by inference.js)
const inference = ModelInference;

let currentDataset = 'mnist'; // 'mnist' or 'emnist'
window.currentDataset = currentDataset;
let drawingCanvas = null;

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Initialize Three.js Scenes
  if (typeof ThreeScene !== 'undefined') {
    ThreeScene.init(document.getElementById('hero-canvas-container'));
  }
  if (typeof ArchitectureViewer !== 'undefined') {
    ArchitectureViewer.init('arch-viewer');
  }

  // 2. Initialize Visualizations
  if (typeof Visualizations !== 'undefined') {
    await Visualizations.init();
  }

  // 3. Initialize Drawing Canvas
  drawingCanvas = new DrawingCanvas('drawing-canvas', {
    onInteractionEnd: async (hasData) => {
      if (!hasData) {
        hidePrediction();
        return;
      }

      // Get raw grayscale data (Float32Array, [0,1] range) from canvas
      const grayscaleData = drawingCanvas.getGrayscaleData();
      if (grayscaleData) {
        try {
          const result = await inference.predict(grayscaleData);
          if (result) {
            // Build top-5 predictions list
            const allPreds = result.probabilities.map((prob, idx) => ({
              class: inference.getClassLabel(idx),
              probability: prob
            }));

            // Sort by probability descending
            allPreds.sort((a, b) => b.probability - a.probability);

            // Top prediction + next 4
            const top = allPreds[0];
            const rest = allPreds.slice(1, 5);

            showPrediction([top, ...rest]);
          }
        } catch (e) {
          console.error('[App] Inference Error:', e);
          // Show user-friendly error in prediction panel
          showError('Model error. Check console.');
        }
      }
    }
  });

  // Canvas controls
  const btnClear = document.getElementById('btn-clear');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      drawingCanvas.clear();
      hidePrediction();
    });
  }

  const btnUndo = document.getElementById('btn-undo');
  if (btnUndo) {
    btnUndo.addEventListener('click', () => {
      drawingCanvas.undo();
    });
  }

  // 4. Load Initial ONNX Model
  try {
    await inference.loadModel(currentDataset);
    console.log(`[App] ✅ Initial model (${currentDataset}) loaded and ready.`);
  } catch (e) {
    console.error('[App] ❌ Initial model load failed:', e);
    // Show a warning banner if model fails to load
    const demoSection = document.getElementById('demo');
    if (demoSection) {
      const warn = document.createElement('div');
      warn.style.cssText = 'background: #fee2e2; color: #b91c1c; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem;';
      warn.textContent = `⚠️ Model failed to load: ${e.message || e}. Check the console for details.`;
      demoSection.querySelector('.container').prepend(warn);
    }
  }

  // 5. Setup Toggle Buttons
  setupToggles();

  // 6. Setup Scroll Animations & Observers
  setupScrollEffects();

  // 7. Animate Hero Stats
  animateHeroStats();
});

function showPrediction(predictions) {
  const emptyPanel = document.getElementById('prediction-empty');
  const resultPanel = document.getElementById('prediction-result');

  if (!predictions || predictions.length === 0) return;
  const topPred = predictions[0];

  if (emptyPanel) emptyPanel.style.display = 'none';
  if (resultPanel) resultPanel.style.display = 'flex';

  const predChar = document.getElementById('prediction-char');
  if (predChar) predChar.textContent = topPred.class;

  const confValue = Math.round(topPred.probability * 100);
  const confValueEl = document.getElementById('prediction-conf-value');
  if (confValueEl) confValueEl.textContent = confValue + '%';

  const gaugeText = document.getElementById('gauge-text');
  if (gaugeText) gaugeText.textContent = confValue + '%';

  // Update gauge arc (circumference ≈ 440 for r=70)
  const arc = document.getElementById('gauge-arc');
  if (arc) {
    const circumference = 2 * Math.PI * 70; // ~440
    arc.style.strokeDasharray = `${topPred.probability * circumference} ${circumference}`;
  }

  // Update top-5 bars
  const barsContainer = document.getElementById('prediction-bars');
  if (barsContainer) {
    barsContainer.innerHTML = '';
    predictions.slice(0, 5).forEach((pred, i) => {
      const pVal = Math.round(pred.probability * 100);
      const barDiv = document.createElement('div');
      barDiv.style.marginBottom = '12px';

      const labelDiv = document.createElement('div');
      labelDiv.style.cssText = 'display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;';

      const labelText = document.createElement('span');
      labelText.textContent = `Class: ${pred.class}${i === 0 ? ' ⭐' : ''}`;
      labelText.style.fontWeight = i === 0 ? '600' : '400';

      const valText = document.createElement('span');
      valText.textContent = `${pVal}%`;

      labelDiv.appendChild(labelText);
      labelDiv.appendChild(valText);

      const track = document.createElement('div');
      track.style.cssText = 'width:100%; height:6px; background:rgba(99,102,241,0.1); border-radius:3px; overflow:hidden;';

      const fill = document.createElement('div');
      fill.style.cssText = `height:100%; width:${pVal}%; background:${i === 0 ? '#6366F1' : '#8B5CF6'}; border-radius:3px; transition:width 0.3s ease;`;

      track.appendChild(fill);
      barDiv.appendChild(labelDiv);
      barDiv.appendChild(track);
      barsContainer.appendChild(barDiv);
    });
  }
}

function showError(message) {
  const emptyPanel = document.getElementById('prediction-empty');
  const resultPanel = document.getElementById('prediction-result');
  if (emptyPanel) {
    emptyPanel.style.display = 'flex';
    const msgEl = emptyPanel.querySelector('p');
    if (msgEl) msgEl.textContent = message;
  }
  if (resultPanel) resultPanel.style.display = 'none';
}

function hidePrediction() {
  const emptyPanel = document.getElementById('prediction-empty');
  const resultPanel = document.getElementById('prediction-result');
  if (emptyPanel) {
    emptyPanel.style.display = 'flex';
    const msgEl = emptyPanel.querySelector('p');
    if (msgEl) msgEl.textContent = 'Draw a character to see predictions';
  }
  if (resultPanel) resultPanel.style.display = 'none';
  const barsContainer = document.getElementById('prediction-bars');
  if (barsContainer) barsContainer.innerHTML = '';
}

function setupToggles() {
  const toggleGroups = document.querySelectorAll('.dataset-toggle');

  toggleGroups.forEach(group => {
    const buttons = group.querySelectorAll('button');
    buttons.forEach(btn => {
      btn.addEventListener('click', async (e) => {
        // Update active class
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const dataset = btn.getAttribute('data-dataset');
        const parentId = group.id;

        if (parentId === 'metrics-toggle' || parentId === 'confusion-toggle' ||
            parentId === 'roc-toggle' || parentId === 'perclass-toggle') {
          if (typeof Visualizations !== 'undefined') {
            Visualizations.updateAll(dataset);
          }
        } else {
          // Live demo toggle — switch model
          currentDataset = dataset;
          window.currentDataset = dataset;
          if (drawingCanvas) drawingCanvas.clear();
          hidePrediction();

          try {
            await inference.loadModel(dataset);
            console.log(`[App] ✅ Switched to ${dataset} model`);
          } catch (e) {
            console.error('[App] ❌ Failed to switch models:', e);
          }
        }
      });
    });
  });
}

function setupScrollEffects() {
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

  if (typeof ArchitectureViewer !== 'undefined') {
    const archObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const layerIdx = parseInt(entry.target.getAttribute('data-layer'));
          ArchitectureViewer.highlightLayer(layerIdx);
        }
      });
    }, { threshold: 0.5, rootMargin: '-10% 0px -40% 0px' });

    document.querySelectorAll('.layer-card').forEach(el => {
      archObserver.observe(el);
    });
  }
}

function animateHeroStats() {
  const statValues = document.querySelectorAll('.hero-stat-value');

  statValues.forEach(el => {
    const target = parseFloat(el.getAttribute('data-target'));
    const suffix = el.getAttribute('data-suffix') || '';
    const isInteger = !el.getAttribute('data-target').includes('.');
    const duration = 1500;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
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
        el.textContent = isInteger ? target.toLocaleString() + suffix : target.toFixed(1) + suffix;
      }
    }
    requestAnimationFrame(update);
  });
}
