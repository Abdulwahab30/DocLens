// Purely decorative animated 3D backdrop, isolated from app.js so it can be swapped or
// removed without touching application logic. Loaded from a CDN <script> tag — no build step.
(function () {
  const canvas = document.getElementById("bg-scene");
  if (!canvas || typeof THREE === "undefined") return;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.z = 7;

  function resize() {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // Slowly rotating wireframe icosahedron — a subtle "data structure" motif.
  const coreGeometry = new THREE.IcosahedronGeometry(2.1, 1);
  const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0x7c5cff,
    wireframe: true,
    transparent: true,
    opacity: 0.35,
  });
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  scene.add(core);

  // Drifting particle field for depth.
  const particleCount = 450;
  const positions = new Float32Array(particleCount * 3);
  for (let i = 0; i < positions.length; i++) {
    positions[i] = (Math.random() - 0.5) * 32;
  }
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const particleMaterial = new THREE.PointsMaterial({
    color: 0x9d8cff,
    size: 0.045,
    transparent: true,
    opacity: 0.55,
  });
  const particles = new THREE.Points(particleGeometry, particleMaterial);
  scene.add(particles);

  let frame = 0;
  function animate() {
    requestAnimationFrame(animate);
    frame += 1;
    core.rotation.x += 0.0016;
    core.rotation.y += 0.0026;
    particles.rotation.y += 0.00045;
    camera.position.x = Math.sin(frame * 0.0006) * 0.6;
    camera.position.y = Math.cos(frame * 0.0005) * 0.4;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  animate();
})();
