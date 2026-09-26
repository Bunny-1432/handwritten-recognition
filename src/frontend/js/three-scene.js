/* ============================================================
   Three.js Scene — Hero Particle Background
   ============================================================ */
const ThreeScene = (() => {
  let scene, camera, renderer;
  let particles = [];
  let lines = [];
  let animationId;
  let clock;

  const PARTICLE_COUNT = 150;
  const CONNECTION_DISTANCE = 120;
  const COLORS = [0x6366F1, 0x8B5CF6, 0x06B6D4, 0xA78BFA, 0x818CF8];

  function init(container) {
    if (!container || typeof THREE === 'undefined') return false;

    clock = new THREE.Clock();

    // Scene
    scene = new THREE.Scene();

    // Camera
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera = new THREE.PerspectiveCamera(60, w / h, 1, 2000);
    camera.position.z = 500;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Particles
    createParticles();

    // Lines material
    createConnectionLines();

    // Resize handler
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    // Start animation
    animate();
    return true;
  }

  function createParticles() {
    const geometry = new THREE.SphereGeometry(2, 8, 8);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const color = COLORS[Math.floor(Math.random() * COLORS.length)];
      const material = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.3 + Math.random() * 0.4
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(
        (Math.random() - 0.5) * 800,
        (Math.random() - 0.5) * 600,
        (Math.random() - 0.5) * 400
      );

      mesh.userData = {
        velocityX: (Math.random() - 0.5) * 0.4,
        velocityY: (Math.random() - 0.5) * 0.3,
        velocityZ: (Math.random() - 0.5) * 0.2,
        baseX: mesh.position.x,
        baseY: mesh.position.y,
        phase: Math.random() * Math.PI * 2,
        amplitude: 10 + Math.random() * 30
      };

      scene.add(mesh);
      particles.push(mesh);
    }
  }

  function createConnectionLines() {
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x6366F1,
      transparent: true,
      opacity: 0.08
    });

    // Pre-allocate line geometry (we'll reuse positions)
    const maxLines = 300;
    for (let i = 0; i < maxLines; i++) {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(6);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const line = new THREE.Line(geometry, lineMaterial);
      line.visible = false;
      scene.add(line);
      lines.push(line);
    }
  }

  function updateConnections() {
    let lineIdx = 0;
    for (let i = 0; i < particles.length && lineIdx < lines.length; i++) {
      for (let j = i + 1; j < particles.length && lineIdx < lines.length; j++) {
        const dx = particles[i].position.x - particles[j].position.x;
        const dy = particles[i].position.y - particles[j].position.y;
        const dz = particles[i].position.z - particles[j].position.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < CONNECTION_DISTANCE) {
          const line = lines[lineIdx];
          const positions = line.geometry.attributes.position.array;
          positions[0] = particles[i].position.x;
          positions[1] = particles[i].position.y;
          positions[2] = particles[i].position.z;
          positions[3] = particles[j].position.x;
          positions[4] = particles[j].position.y;
          positions[5] = particles[j].position.z;
          line.geometry.attributes.position.needsUpdate = true;
          line.material.opacity = 0.08 * (1 - dist / CONNECTION_DISTANCE);
          line.visible = true;
          lineIdx++;
        }
      }
    }

    // Hide unused lines
    for (let i = lineIdx; i < lines.length; i++) {
      lines[i].visible = false;
    }
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    const elapsed = clock.getElapsedTime();

    // Update particles
    for (const p of particles) {
      const d = p.userData;
      p.position.x = d.baseX + Math.sin(elapsed * 0.3 + d.phase) * d.amplitude;
      p.position.y += d.velocityY;
      p.position.z += d.velocityZ;

      // Gentle drift
      d.baseX += d.velocityX * 0.3;

      // Wrap around boundaries
      if (p.position.y > 350) p.position.y = -350;
      if (p.position.y < -350) p.position.y = 350;
      if (d.baseX > 450) d.baseX = -450;
      if (d.baseX < -450) d.baseX = 450;
    }

    updateConnections();

    // Gentle camera sway
    camera.position.x = Math.sin(elapsed * 0.1) * 20;
    camera.position.y = Math.cos(elapsed * 0.08) * 15;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
  }

  function destroy() {
    if (animationId) cancelAnimationFrame(animationId);
    if (renderer) renderer.dispose();
  }

  return { init, destroy };
})();
