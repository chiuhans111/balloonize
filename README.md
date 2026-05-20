# Balloonize — Realistic 3D Mylar Balloon WebGL Simulator

Balloonize is a high-fidelity WebGL2-powered interactive simulator that transforms any image or text logo into a realistic, shiny, 3D inflated mylar balloon. It features responsive physical elastic wave simulation, rich customized lighting, dynamic morphological contour trimming, and touch-wiggle interaction.

---

## 📂 Codebase Architecture & Files

* **`index.html`**
  * Controls the page structure, glassmorphic UI layout, collapsible **Tuning Pump** parameters, and events.
  * Contains the temporary 2D canvas generator to bake the default "Balloonize" round text graphic on page load.
  * Registers mouse/touch wiggling, image file upload, drag-and-drop, and paste events.
  * Captures the canvas to PNG via the **Save Image** button.
* **`engine.js`**
  * Defines the central orchestrator class `BalloonizeEngine`.
  * Manages WebGL2 context allocation (initialized with `{ preserveDrawingBuffer: true }` to support canvas capturing).
  * Allocates ping-pong framebuffers (`simA` and `simB`) to update the physics state.
  * Synces DOM control inputs (sliders) with the engine settings.
  * Drives the zero-idle sleep engine, waking the requestAnimationFrame loop upon user interaction or transition changes.
* **`shaders.js`**
  * Houses all GLSL ES 3.00 shader sources:
    * **Vertex Shader**: Simple full-screen quad pass-through.
    * **Trim Shader**: Solves the morphological boundary erosion and the Eikonal SDF distance field.
    * **Wave Solver Shader**: Computes the elastic physics using the discrete wave equation.
    * **Composite/Lighting Shader**: Renders the final glossy 3D mylar balloon optics.

---

## 🧠 Core Algorithms

### 1. Morphological Erosion Trimming & SDF (Trim Pass)
To shape the flat image into a balloon, a trim pass runs 150 times when an image is first loaded to shrink-wrap the mask contour:
* **Initial State**: A solid bounding box mask initialized over the canvas with a `0.05` margin.
* **Stopping Condition**:
  * **Transparent PNGs (`u_hasTransparency > 0.5`)**: The erosion solver instantly halts at any opaque pixel (`sourceImg.a >= 0.1`).
  * **JPEGs (`u_hasTransparency <= 0.5`)**: It falls back to color gradient analysis (`grad > u_gradientThreshold`) computed from neighboring pixel luminance differences.
* **Eikonal Rouy-Tourin SDF Solver**: Computes a smooth Euclidean Signed Distance Field (SDF) directly on the GPU, avoiding the faceted appearance of 8-way Chamfer solvers and creating high-quality rounded corners.

### 2. Elastic Physics Wave Solver (Wave Pass)
Simulates realistic material tension and damping on a 256x256 simulation grid:
* **Displacement**: Calculated using the wave equation:
  $$\text{Acceleration} = (\text{Tension})^2 \times \text{Laplacian}$$
* **Stability & Damping**:
  * Enforces the 2D CFL stability condition ($c \le 1/\sqrt{2} \approx 0.7071$).
  * Employs Gaussian curvature regularization to damp out high-frequency noise and maintain material integrity during interactions.

### 3. Glassmorphic Mylar Optics (Composite Pass)
Combines displacement, normal maps, and lighting calculations:
* **Normal Generation**: Calculated in real-time from the height/displacement map.
* **Optics**:
  * **Specular Core & Glow**: Generates the high-gloss core specular highlight and a secondary dimmed glowing wave reflection (`waveIntensity = 0.9` for balanced contrast).
  * **Rim Lighting**: Highlights the silhouette edges to enhance 3D depth.
  * **Environment Mapping**: Blends a radial gradient background onto the balloon reflection to simulate a studio lighting environment.

---

## 🎨 Layout & UI Customization

* **Theme**: Modern dark mode with Outfit (minimalist UI) and Fredoka (playful titles) Google Fonts.
* **Branding Subtitle**: Restored back to `balloonmorphism • inspired by figma` below the header.
* **Author Bylines**:
  * "Balloonize" title links directly to: `https://github.com/chiuhans111/balloonize` (opens in new tab).
  * "hanschiu" subtitle links to: `https://x.com/chiu_hans` (opens in new tab).
* **Compact GitHub Star Button**:
  * Positioned inline to the right of the author byline.
  * Styled with a dark-mode invert filter (`filter: invert(1) hue-rotate(180deg)`) and a subtle hover opacity transition (`0.7` to `1.0`).
* **Dual Action Buttons**:
  * **Inflate New Image 🎈**: Triggers file input upload.
  * **Save Image 📸**: Captures and downloads the current WebGL canvas state as a PNG.

---

## 🚀 How to Run Locally

Since WebGL textures require loading local image files, browsers will block cross-origin requests unless served from a local web server.

1. Install a lightweight server globally:
   ```bash
   npm install -g http-server
   # or use Live Server in VS Code
   ```
2. Start the server from the repository root:
   ```bash
   http-server -p 8080
   ```
3. Open `http://localhost:8080` in your browser.
