/* ============================================================
   Visualizations (Chart.js)
   ============================================================ */
const Visualizations = (() => {
  let metricsData = null;
  let charts = {};

  const COLORS = {
    primary: '#6366F1',
    primaryLight: 'rgba(99,102,241,0.2)',
    secondary: '#8B5CF6',
    secondaryLight: 'rgba(139,92,246,0.2)',
    accent: '#06B6D4',
    background: '#FFFFFF',
    text: '#1F2937',
    grid: '#E5E7EB'
  };

  async function loadData() {
    try {
      const response = await fetch('data/training_metrics.json');
      metricsData = await response.json();
      return true;
    } catch (err) {
      console.error('Failed to load training metrics:', err);
      // Fallback dummy data if fetch fails (e.g., viewing directly from file://)
      metricsData = getFallbackData();
      return true;
    }
  }

  function renderLossAccuracy(dataset = 'mnist') {
    const data = metricsData[dataset];
    
    // Loss Chart
    const ctxLoss = document.getElementById('loss-chart').getContext('2d');
    if (charts.loss) charts.loss.destroy();
    
    charts.loss = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: Array.from({length: data.epochs}, (_, i) => `Epoch ${i+1}`),
        datasets: [{
          label: 'Training Loss',
          data: data.loss,
          borderColor: COLORS.primary,
          backgroundColor: COLORS.primaryLight,
          borderWidth: 3,
          tension: 0.4,
          fill: true
        }]
      },
      options: getCommonOptions('Loss')
    });

    // Accuracy Chart
    const ctxAcc = document.getElementById('accuracy-chart').getContext('2d');
    if (charts.accuracy) charts.accuracy.destroy();
    
    charts.accuracy = new Chart(ctxAcc, {
      type: 'line',
      data: {
        labels: Array.from({length: data.epochs}, (_, i) => `Epoch ${i+1}`),
        datasets: [{
          label: 'Test Accuracy (%)',
          data: data.accuracy,
          borderColor: COLORS.secondary,
          backgroundColor: COLORS.secondaryLight,
          borderWidth: 3,
          tension: 0.4,
          fill: true
        }]
      },
      options: getCommonOptions('Accuracy (%)')
    });
  }

  function renderConfusionMatrix(dataset = 'mnist') {
    const data = metricsData[dataset];
    const ctx = document.getElementById('confusion-chart').getContext('2d');
    if (charts.confusion) charts.confusion.destroy();
    
    // We'll approximate a heatmap using a bubble chart or 2D bar if matrix plugin not available.
    // For simplicity with standard Chart.js, we use a scatter/bubble chart formatted as a grid.
    const matrixData = [];
    const maxVal = Math.max(...data.confusion_matrix.flat());
    
    for (let i = 0; i < data.classes.length; i++) {
      for (let j = 0; j < data.classes.length; j++) {
        const val = data.confusion_matrix[i][j];
        if (val > 0) {
          matrixData.push({
            x: j,
            y: data.classes.length - 1 - i, // invert y for visual matrix
            r: Math.max(3, (val / maxVal) * 20),
            v: val
          });
        }
      }
    }

    charts.confusion = new Chart(ctx, {
      type: 'bubble',
      data: {
        datasets: [{
          label: 'Confusion Matrix',
          data: matrixData,
          backgroundColor: COLORS.primaryLight,
          borderColor: COLORS.primary,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const i = data.classes.length - 1 - ctx.raw.y;
                const j = ctx.raw.x;
                return `True: ${data.classes[i]}, Pred: ${data.classes[j]} (Count: ${ctx.raw.v})`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Predicted Class' },
            ticks: {
              callback: function(val) { return data.classes[val] || ''; },
              stepSize: 1
            },
            grid: { display: false }
          },
          y: {
            title: { display: true, text: 'True Class' },
            ticks: {
              callback: function(val) { return data.classes[data.classes.length - 1 - val] || ''; },
              stepSize: 1
            },
            grid: { display: false }
          }
        }
      }
    });
  }

  function renderROCCurve(dataset = 'mnist') {
    const data = metricsData[dataset];
    const ctx = document.getElementById('roc-chart').getContext('2d');
    if (charts.roc) charts.roc.destroy();

    const rocPoints = data.roc.fpr.map((x, i) => ({ x: x, y: data.roc.tpr[i] }));

    charts.roc = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'ROC Curve (Macro Avg)',
            data: rocPoints,
            borderColor: COLORS.secondary,
            backgroundColor: 'transparent',
            borderWidth: 3,
            showLine: true,
            tension: 0.1
          },
          {
            label: 'Random Guess',
            data: [{x: 0, y: 0}, {x: 1, y: 1}],
            borderColor: COLORS.grid,
            borderWidth: 2,
            borderDash: [5, 5],
            showLine: true,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
          y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
        }
      }
    });

    // Populate AUC Cards
    const aucContainer = document.getElementById('auc-cards');
    if (aucContainer) {
      aucContainer.innerHTML = '';
      data.auc.slice(0, 5).forEach((val, i) => {
        aucContainer.innerHTML += `
          <div class="glass-card" style="margin-bottom: 10px; padding: 15px;">
            <div style="font-size: 0.9rem; color: #666;">Class ${data.classes[i]}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: ${COLORS.primary};">AUC: ${val.toFixed(3)}</div>
          </div>
        `;
      });
    }
  }

  function renderPerClass(dataset = 'mnist') {
    const data = metricsData[dataset];
    const ctx = document.getElementById('perclass-chart').getContext('2d');
    if (charts.perclass) charts.perclass.destroy();

    charts.perclass = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.classes.slice(0, 20), // Show max 20 for EMNIST
        datasets: [
          {
            label: 'Precision',
            data: data.per_class.precision.slice(0, 20),
            backgroundColor: COLORS.primary
          },
          {
            label: 'Recall',
            data: data.per_class.recall.slice(0, 20),
            backgroundColor: COLORS.secondary
          },
          {
            label: 'F1 Score',
            data: data.per_class.f1.slice(0, 20),
            backgroundColor: COLORS.accent
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { tooltip: { mode: 'index', intersect: false } },
        scales: {
          x: { title: { display: true, text: 'Class' } },
          y: { title: { display: true, text: 'Score' }, min: 0.8, max: 1.0 }
        }
      }
    });
  }

  function getCommonOptions(yLabel) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true }
      },
      scales: {
        x: { grid: { color: COLORS.grid } },
        y: {
          title: { display: true, text: yLabel },
          grid: { color: COLORS.grid }
        }
      }
    };
  }

  function updateAll(dataset) {
    if (!metricsData) return;
    renderLossAccuracy(dataset);
    renderConfusionMatrix(dataset);
    renderROCCurve(dataset);
    renderPerClass(dataset);
  }

  // Fallback data
  function getFallbackData() {
    return {
      mnist: {
        epochs: 5,
        loss: [1.0, 0.5, 0.2, 0.1, 0.05],
        accuracy: [80, 90, 95, 98, 99],
        classes: ['0','1','2','3','4','5','6','7','8','9'],
        confusion_matrix: Array(10).fill(0).map((_,i) => Array(10).fill(0).map((_,j) => i===j ? 100 : Math.floor(Math.random()*5))),
        roc: { fpr: [0, 0.1, 1], tpr: [0, 0.9, 1] },
        auc: Array(10).fill(0.99),
        per_class: {
          precision: Array(10).fill(0.98),
          recall: Array(10).fill(0.98),
          f1: Array(10).fill(0.98)
        }
      },
      emnist: {
        epochs: 5,
        loss: [2.0, 1.5, 1.0, 0.8, 0.6],
        accuracy: [60, 75, 80, 85, 87],
        classes: ['A','B','C','D','E'],
        confusion_matrix: Array(5).fill(0).map((_,i) => Array(5).fill(0).map((_,j) => i===j ? 100 : Math.floor(Math.random()*15))),
        roc: { fpr: [0, 0.2, 1], tpr: [0, 0.8, 1] },
        auc: Array(5).fill(0.95),
        per_class: {
          precision: Array(5).fill(0.85),
          recall: Array(5).fill(0.85),
          f1: Array(5).fill(0.85)
        }
      }
    };
  }

  return {
    init: async () => {
      await loadData();
      updateAll('mnist');
    },
    updateAll
  };
})();
