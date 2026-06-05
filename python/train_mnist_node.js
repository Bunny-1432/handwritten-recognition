/**
 * train_mnist_node.js
 * ====================
 * Train a CNN on MNIST using TF.js Node.js and save it for the web app.
 *
 * Usage:
 *   npm install @tensorflow/tfjs-node mnist
 *   node train_mnist_node.js
 *
 * This saves the trained model to ../web/data/model/ as a TF.js LayersModel,
 * compatible with tf.loadLayersModel() in the browser.
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const https = require('https');
const zlib = require('zlib');

// ─── Config ──────────────────────────────────────────────────────────────────
const EPOCHS = 10;
const BATCH_SIZE = 128;
const OUTPUT_DIR = path.join(__dirname, '..', 'web', 'data', 'model');
const METRICS_PATH = path.join(__dirname, '..', 'web', 'data', 'training_metrics.json');

// ─── MNIST Download helpers ───────────────────────────────────────────────────
const MNIST_URLS = {
  trainImages: 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
  trainLabels: 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
  testImages:  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
  testLabels:  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
};

const CACHE_DIR = path.join(__dirname, '.mnist_cache');

function downloadAndDecompress(url, destPath) {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(destPath)) {
      console.log(`  [cache] ${path.basename(destPath)}`);
      return resolve(fs.readFileSync(destPath));
    }
    console.log(`  [download] ${path.basename(url)}...`);
    https.get(url, (res) => {
      const gunzip = zlib.createGunzip();
      const chunks = [];
      res.pipe(gunzip);
      gunzip.on('data', chunk => chunks.push(chunk));
      gunzip.on('end', () => {
        const buf = Buffer.concat(chunks);
        fs.mkdirSync(CACHE_DIR, { recursive: true });
        fs.writeFileSync(destPath, buf);
        resolve(buf);
      });
      gunzip.on('error', reject);
    }).on('error', reject);
  });
}

async function loadMNIST() {
  console.log('\nLoading MNIST dataset...');
  const [trainImgBuf, trainLblBuf, testImgBuf, testLblBuf] = await Promise.all([
    downloadAndDecompress(MNIST_URLS.trainImages, path.join(CACHE_DIR, 'train-images')),
    downloadAndDecompress(MNIST_URLS.trainLabels, path.join(CACHE_DIR, 'train-labels')),
    downloadAndDecompress(MNIST_URLS.testImages,  path.join(CACHE_DIR, 'test-images')),
    downloadAndDecompress(MNIST_URLS.testLabels,  path.join(CACHE_DIR, 'test-labels')),
  ]);

  function parseImages(buf) {
    const n = buf.readUInt32BE(4);
    const rows = buf.readUInt32BE(8);
    const cols = buf.readUInt32BE(12);
    const pixels = new Float32Array(n * rows * cols);
    for (let i = 0; i < pixels.length; i++) {
      pixels[i] = buf[16 + i] / 255.0;
    }
    return tf.tensor4d(pixels, [n, rows, cols, 1]);
  }

  function parseLabels(buf) {
    const n = buf.readUInt32BE(4);
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) labels[i] = buf[8 + i];
    return tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
  }

  const xTrain = parseImages(trainImgBuf);
  const yTrain = parseLabels(trainLblBuf);
  const xTest  = parseImages(testImgBuf);
  const yTest  = parseLabels(testLblBuf);

  console.log(`  Train: ${xTrain.shape}  Test: ${xTest.shape}`);
  return { xTrain, yTrain, xTest, yTest };
}

// ─── Model Architecture ───────────────────────────────────────────────────────
function buildModel() {
  const model = tf.sequential({ name: 'HandwrittenCNN_MNIST' });

  // Conv Block 1
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', useBias: false, inputShape: [28, 28, 1], name: 'conv1' }));
  model.add(tf.layers.batchNormalization({ name: 'bn1' }));
  model.add(tf.layers.activation({ activation: 'relu', name: 'relu1' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: 'pool1' }));

  // Conv Block 2
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', useBias: false, name: 'conv2' }));
  model.add(tf.layers.batchNormalization({ name: 'bn2' }));
  model.add(tf.layers.activation({ activation: 'relu', name: 'relu2' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: 'pool2' }));

  // Conv Block 3
  model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: 'same', useBias: false, name: 'conv3' }));
  model.add(tf.layers.batchNormalization({ name: 'bn3' }));
  model.add(tf.layers.activation({ activation: 'relu', name: 'relu3' }));

  // Classifier
  model.add(tf.layers.globalAveragePooling2d({ name: 'gap' }));
  model.add(tf.layers.dense({ units: 256, activation: 'relu', name: 'fc1' }));
  model.add(tf.layers.dropout({ rate: 0.5, name: 'dropout' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax', name: 'output' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();
  console.log(`\nTotal params: ${model.countParams().toLocaleString()}`);
  return model;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log('='.repeat(60));
  console.log('  NeuroScribe — MNIST CNN Training (TF.js Node.js)');
  console.log('='.repeat(60));

  const { xTrain, yTrain, xTest, yTest } = await loadMNIST();
  const model = buildModel();

  const history = { acc: [], valAcc: [], loss: [], valLoss: [] };

  // Split 10% for validation
  const nTrain = xTrain.shape[0];
  const nVal = Math.floor(nTrain * 0.1);
  const nTrainActual = nTrain - nVal;

  const xVal = xTrain.slice([nTrainActual, 0, 0, 0], [nVal, 28, 28, 1]);
  const yVal = yTrain.slice([nTrainActual, 0], [nVal, 10]);
  const xTrainActual = xTrain.slice([0, 0, 0, 0], [nTrainActual, 28, 28, 1]);
  const yTrainActual = yTrain.slice([0, 0], [nTrainActual, 10]);

  console.log(`\nTraining for ${EPOCHS} epochs...`);
  const t0 = Date.now();

  await model.fit(xTrainActual, yTrainActual, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [xVal, yVal],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        history.acc.push(parseFloat(logs.acc.toFixed(4)));
        history.valAcc.push(parseFloat(logs.val_acc.toFixed(4)));
        history.loss.push(parseFloat(logs.loss.toFixed(4)));
        history.valLoss.push(parseFloat(logs.val_loss.toFixed(4)));
        console.log(
          `  Epoch ${String(epoch + 1).padStart(2)}/${EPOCHS}` +
          `  loss=${logs.loss.toFixed(4)}  acc=${(logs.acc * 100).toFixed(2)}%` +
          `  val_loss=${logs.val_loss.toFixed(4)}  val_acc=${(logs.val_acc * 100).toFixed(2)}%`
        );
      }
    }
  });

  const elapsed = (Date.now() - t0) / 1000;
  const mins = Math.floor(elapsed / 60);
  const secs = Math.floor(elapsed % 60);

  // Evaluate on test set
  const evalResult = model.evaluate(xTest, yTest, { batchSize: 256 });
  const testLoss = (await evalResult[0].data())[0];
  const testAcc  = (await evalResult[1].data())[0];
  evalResult.forEach(t => t.dispose());

  console.log(`\n✓ Test Accuracy: ${(testAcc * 100).toFixed(2)}%  |  Test Loss: ${testLoss.toFixed(4)}`);
  console.log(`✓ Training time: ${mins}m ${secs}s`);

  // Save TF.js model
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  await model.save(`file://${OUTPUT_DIR}`);
  console.log(`\n✓ Model saved to: ${OUTPUT_DIR}`);
  fs.readdirSync(OUTPUT_DIR).forEach(f => {
    const size = fs.statSync(path.join(OUTPUT_DIR, f)).size;
    console.log(`  ${f.padEnd(40)} ${(size / 1024).toFixed(1)} KB`);
  });

  // Update training_metrics.json
  let metrics = {};
  if (fs.existsSync(METRICS_PATH)) {
    metrics = JSON.parse(fs.readFileSync(METRICS_PATH, 'utf8'));
  }

  metrics.mnist = {
    test_accuracy:  parseFloat((testAcc * 100).toFixed(2)),
    test_loss:      parseFloat(testLoss.toFixed(4)),
    epochs:         history.acc.length,
    total_params:   model.countParams(),
    training_time:  `${mins}m ${secs}s`,
    train_accuracy: history.acc,
    val_accuracy:   history.valAcc,
    train_loss:     history.loss,
    val_loss:       history.valLoss,
  };

  fs.writeFileSync(METRICS_PATH, JSON.stringify(metrics, null, 2));
  console.log(`✓ Metrics updated: ${METRICS_PATH}`);

  // Clean up tensors
  xTrain.dispose(); yTrain.dispose();
  xTest.dispose();  yTest.dispose();
  xVal.dispose();   yVal.dispose();
  xTrainActual.dispose(); yTrainActual.dispose();

  console.log('\n' + '='.repeat(60));
  console.log(`  DONE — Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);
  console.log('='.repeat(60));
}

main().catch(err => {
  console.error('Training failed:', err);
  process.exit(1);
});
