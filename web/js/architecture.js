/* ============================================================
   3D CNN Architecture Viewer (Three.js)
   Interactive layer visualization with data-flow particles
   ============================================================ */
const ArchitectureViewer = (() => {
  let scene, camera, renderer, controls;
  let layerMeshes = [];
  let flowParticles = [];
  let animationId;
  let clock;
  let hoveredLayer = -1;
  let raycaster, mouse;
  let container;
  let tooltip;

  // Layer definitions: name, type, dimensions (w, h, d for visual), color
  const LAYERS = [
    { name: 'Input',         type: 'input',  w: 2.8, h: 2.8, d: 0.3,  color: 0x9CA3AF, x: -18 },
    { name: 'Conv1 3×3',     type: 'conv',   w: 2.8, h: 2.8, d: 1.0,  color: 0x6366F1, x: -15 },
    { name: 'BN+ReLU',       type: 'act',    w: 2.8, h: 2.8, d: 0.4,  color: 0x10B981, x: -13 },
    { name: 'MaxPool',       type: 'pool',   w: 1.8, h: 1.8, d: 1.0,  color: 0x06B6D4, x: -10.5 },
    { name: 'Conv2 3×3',     type: 'conv',   w: 1.8, h: 1.8, d: 1.6,  color: 0x6366F1, x: -7.5 },
    { name: 'BN+ReLU',       type: 'act',    w: 1.8, h: 1.8, d: 0.5,  color: 0x10B981, x: -5.2 },
    { name: 'MaxPool',       type: 'pool',   w: 1.0, h: 1.0, d: 1.6,  color: 0x06B6D4, x: -2.8 },
    { name: 'Conv3 3×3',     type: 'conv',   w: 1.0, h: 1.0, d: 2.5,  color: 0x6366F1, x: 0.5 },
    { name: 'BN+ReLU',       type: 'act',    w: 1.0, h: 1.0, d: 0.5,  color: 0x10B981, x: 3 },
    { name: 'AdaptPool',     type: 'pool',   w: 0.6, h: 0.6, d: 2.5,  color: 0x06B6D4, x: 5.5 },
    { name: 'Flatten',       type: 'flat',   w: 4.0, h: 0.3, d: 0.3,  color: 0xF59E0B, x: 8.5 },
    { name: 'FC 256',        type: 'dense',  w: 3.0, h: 0.4, d: 0.4,  color: 0xA855F7, x: 12 },
    { name: 'Dropout',       type: 'drop',   w: 3.0, h: 0.3, d: 0.3,  color: 0xF43F5E, x: 14.5 },
    { name: 'Output',        type: 'dense',  w: 1.5, h: 0.4, d: 0.4,  color: 0xA855F7, x: 17 }
  ];

  function init(containerId) {
    container = document.getElementById(containerId);
    if (!container || typeof THREE === 'undefined') return false;

    clock = new THREE.Clock();
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2(-999, -999);

    // Create tooltip element
    tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.style.position = 'absolute';
    container.style.position = 'relative';
    container.appendChild(tooltip);

    // Scene
    scene = new THREE.Scene();

    const w = container.clientWidth;
    const h = Math.max(container.clientHeight, 500);

    // Camera
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(0, 8, 28);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);
    container.insertBefore(renderer.domElement, container.firstChild);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.minDistance = 10;
    controls.maxDistance = 60;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(10, 15, 10);
    scene.add(directional);

    const directional2 = new THREE.DirectionalLight(0x818CF8, 0.3);
    directional2.position.set(-10, 5, -10);
    scene.add(directional2);

    // Build layers
    buildLayers();

    // Build connections
    buildConnections();

    // Build flow particles
    buildFlowParticles();

    // Ground grid (subtle)
    const gridHelper = new THREE.GridHelper(40, 20, 0xE0E0E0, 0xF0F0F0);
    gridHelper.position.y = -4;
    scene.add(gridHelper);

    // Mouse events
    container.addEventListener('mousemove', onMouseMove);

    // Resize
    window.addEventListener('resize', () => {
      const w = container.clientWidth;
      const h = Math.max(container.clientHeight, 500);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

    animate();
    return true;
  }

  function buildLayers() {
    LAYERS.forEach((layer, i) => {
      const geometry = new THREE.BoxGeometry(layer.d, layer.h, layer.w);

      const material = new THREE.MeshPhongMaterial({
        color: layer.color,
        transparent: true,
        opacity: 0.75,
        shininess: 60
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(layer.x, 0, 0);
      mesh.userData = { layerIndex: i, layerData: layer };

      // Edges
      const edgeGeo = new THREE.EdgesGeometry(geometry);
      const edgeMat = new THREE.LineBasicMaterial({
        color: layer.color,
        transparent: true,
        opacity: 0.4
      });
      const edges = new THREE.LineSegments(edgeGeo, edgeMat);
      mesh.add(edges);

      // Intro animation: layers start below and fly up
      mesh.position.y = -10;
      mesh.scale.set(0.01, 0.01, 0.01);

      scene.add(mesh);
      layerMeshes.push(mesh);

      // Animate in with stagger
      setTimeout(() => {
        animateLayerIn(mesh);
      }, i * 120);
    });
  }

  function animateLayerIn(mesh) {
    const targetY = 0;
    const duration = 800;
    const startTime = performance.now();
    const startY = mesh.position.y;

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const t = Math.min(elapsed / duration, 1);
      // Ease out back
      const c1 = 1.70158;
      const c3 = c1 + 1;
      const eased = 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);

      mesh.position.y = startY + (targetY - startY) * eased;
      mesh.scale.setScalar(eased);

      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function buildConnections() {
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x6366F1,
      transparent: true,
      opacity: 0.15
    });

    for (let i = 0; i < LAYERS.length - 1; i++) {
      const points = [
        new THREE.Vector3(LAYERS[i].x + LAYERS[i].d / 2, 0, 0),
        new THREE.Vector3(LAYERS[i + 1].x - LAYERS[i + 1].d / 2, 0, 0)
      ];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, lineMaterial);
      scene.add(line);
    }
  }

  function buildFlowParticles() {
    const particleGeo = new THREE.SphereGeometry(0.12, 6, 6);
    const particleMat = new THREE.MeshBasicMaterial({
      color: 0x818CF8,
      transparent: true,
      opacity: 0.8
    });

    const startX = LAYERS[0].x;
    const endX = LAYERS[LAYERS.length - 1].x;
    const range = endX - startX;

    for (let i = 0; i < 20; i++) {
      const particle = new THREE.Mesh(particleGeo, particleMat.clone());
      particle.userData = {
        progress: Math.random(),
        speed: 0.002 + Math.random() * 0.003,
        yOffset: (Math.random() - 0.5) * 2,
        zOffset: (Math.random() - 0.5) * 2,
        startX: startX,
        range: range
      };
      particle.position.set(startX, 0, 0);
      scene.add(particle);
      flowParticles.push(particle);
    }
  }

  function onMouseMove(event) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(layerMeshes);

    // Reset previous hover
    if (hoveredLayer >= 0 && layerMeshes[hoveredLayer]) {
      layerMeshes[hoveredLayer].material.opacity = 0.75;
      layerMeshes[hoveredLayer].scale.set(1, 1, 1);
    }

    // Highlight layer cards
    document.querySelectorAll('.layer-card').forEach(c => c.classList.remove('active'));

    if (intersects.length > 0) {
      const idx = intersects[0].object.userData.layerIndex;
      hoveredLayer = idx;
      intersects[0].object.material.opacity = 0.95;
      intersects[0].object.scale.set(1.05, 1.05, 1.05);

      // Show tooltip
      const layer = LAYERS[idx];
      tooltip.textContent = `${layer.name} — ${layer.type}`;
      tooltip.style.left = (event.clientX - container.getBoundingClientRect().left + 15) + 'px';
      tooltip.style.top = (event.clientY - container.getBoundingClientRect().top - 30) + 'px';
      tooltip.classList.add('show');

      // Highlight corresponding card
      const card = document.querySelector(`.layer-card[data-layer="${idx}"]`);
      if (card) card.classList.add('active');

      container.style.cursor = 'pointer';
    } else {
      hoveredLayer = -1;
      tooltip.classList.remove('show');
      container.style.cursor = 'grab';
    }
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    const elapsed = clock.getElapsedTime();

    // Update flow particles
    for (const p of flowParticles) {
      p.userData.progress += p.userData.speed;
      if (p.userData.progress > 1) p.userData.progress = 0;

      const t = p.userData.progress;
      p.position.x = p.userData.startX + t * p.userData.range;
      p.position.y = p.userData.yOffset + Math.sin(t * Math.PI * 4 + elapsed) * 0.3;
      p.position.z = p.userData.zOffset + Math.cos(t * Math.PI * 3 + elapsed) * 0.3;

      // Fade at edges
      const fade = Math.sin(t * Math.PI);
      p.material.opacity = fade * 0.8;
    }

    // Subtle layer breathing
    for (let i = 0; i < layerMeshes.length; i++) {
      if (i !== hoveredLayer) {
        const breath = 1 + Math.sin(elapsed * 0.8 + i * 0.5) * 0.015;
        layerMeshes[i].scale.set(breath, breath, breath);
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }

  // Highlight layer from sidebar
  function highlightLayer(index) {
    if (index >= 0 && index < layerMeshes.length) {
      const mesh = layerMeshes[index];
      // Briefly flash
      mesh.material.opacity = 1;
      mesh.scale.set(1.15, 1.15, 1.15);
      setTimeout(() => {
        mesh.material.opacity = 0.75;
        mesh.scale.set(1, 1, 1);
      }, 400);
    }
  }

  return { init, highlightLayer };
})();
