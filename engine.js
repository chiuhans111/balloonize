import {
    vertexShaderSource,
    trimFragmentShaderSource,
    waveFragmentShaderSource,
    compositeFragmentShaderSource
} from './shaders.js';

export class BalloonizeEngine {
    constructor(canvas, imageSource, options = {}) {
        this.canvas = canvas;
        // WebGL2 is required for float textures and texture() in GLSL
        this.gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
        if (!this.gl) throw new Error("WebGL2 not supported");

        // Enable float textures for ping-pong buffering
        if (!this.gl.getExtension('EXT_color_buffer_float')) {
            console.warn("EXT_color_buffer_float not supported");
        }

        this.pointerPos = [0.5, 0.5];
        this.targetPointerPos = [0.5, 0.5];
        this.pointerForce = 0.0;
        this.targetPointerForce = 0.0;
        this.isInteracting = false;
        
        this.simRes = 256;
        this.rafId = null;
        this.consecutiveIdleFrames = 0;
        
        this.inflationDepth = options.inflationDepth !== undefined ? options.inflationDepth : 1.0;
        this.entranceProgress = 1.0;

        this.hasTransparency = false;
        this.isBaked = false;
        this.isDraggingMask = false;
        this.maskThreshold = 0.75;

        this.physicsParams = Object.assign({
            tension: 0.7,
            damping: 0.67,
            diffusion: 0.15
        }, options.physicsParams || {});

        this.lightingParams = Object.assign({
            env: 1.4,
            az: -45,
            el: 56,
            specCore: 6.7,
            specGlow: 1.0,
            rim: 0.5
        }, options.lightingParams || {});
        
        this.initGL();
        this.initTextures();
        this.initBuffers();
        this.initEvents();

        this.setImage(imageSource).catch(err => console.error("Initial load failed:", err));
    }

    compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error("Shader compile error:", gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    createProgram(vsSource, fsSource) {
        const gl = this.gl;
        const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error("Program link error:", gl.getProgramInfoLog(program));
            return null;
        }
        return program;
    }

    initGL() {
        this.trimProgram = this.createProgram(vertexShaderSource, trimFragmentShaderSource);
        this.waveProgram = this.createProgram(vertexShaderSource, waveFragmentShaderSource);
        this.compositeProgram = this.createProgram(vertexShaderSource, compositeFragmentShaderSource);

        const gl = this.gl;
        this.quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,  1, -1, -1,  1,
            -1,  1,  1, -1,  1,  1,
        ]), gl.STATIC_DRAW);

        this.quadVAO = gl.createVertexArray();
        gl.bindVertexArray(this.quadVAO);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    }

    createTexture(width, height, internalFormat, format, type, data = null) {
        const gl = this.gl;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    initTextures() {
        const gl = this.gl;
        // Ping-Pong pair for Simulation (RGBA32F)
        // R: Current Scalar Wave (u_t), G: Previous (u_t-1), B: Mask, A: SDF
        this.simA = this.createTexture(this.simRes, this.simRes, gl.RGBA32F, gl.RGBA, gl.FLOAT);
        this.simB = this.createTexture(this.simRes, this.simRes, gl.RGBA32F, gl.RGBA, gl.FLOAT);
        
        // Image Texture (RGBA8) - Use LINEAR filtering to smooth out noise
        this.imageTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.imageTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    initBuffers() {
        const gl = this.gl;
        this.fboA = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboA);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.simA, 0);

        this.fboB = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboB);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.simB, 0);
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    setImage(source) {
        return new Promise((resolve, reject) => {
            let src;
            let isObjectURL = false;
            if (source instanceof File || source instanceof Blob) {
                src = URL.createObjectURL(source);
                isObjectURL = true;
            } else if (typeof source === 'string') {
                src = source;
            } else {
                reject(new Error("Unsupported source type"));
                return;
            }

            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                const gl = this.gl;
                gl.bindTexture(gl.TEXTURE_2D, this.imageTex);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

                const maxCanvasDim = 768;
                const aspect = img.width / img.height;
                if (aspect >= 1.0) {
                    this.canvas.width = maxCanvasDim;
                    this.canvas.height = Math.round(maxCanvasDim / aspect);
                } else {
                    this.canvas.width = Math.round(maxCanvasDim * aspect);
                    this.canvas.height = maxCanvasDim;
                }

                this.loadedImage = img;
                this.recomputeMask();

                if (isObjectURL) {
                    URL.revokeObjectURL(src);
                }
                resolve();
            };
            img.onerror = (err) => {
                if (isObjectURL) {
                    URL.revokeObjectURL(src);
                }
                reject(err);
            };
            img.src = src;
        });
    }

    initEvents() {
        const slTension = document.getElementById('slider-tension');
        const slDamping = document.getElementById('slider-damping');
        const slDiffusion = document.getElementById('slider-diffusion');
        const slMaskThreshold = document.getElementById('slider-mask-threshold');

        const slEnv = document.getElementById('slider-env');
        const slAz = document.getElementById('slider-azimuth');
        const slEl = document.getElementById('slider-elevation');
        const slSpecCore = document.getElementById('slider-spec-core');
        const slSpecGlow = document.getElementById('slider-spec-glow');
        const slRim = document.getElementById('slider-rim');
        
        if (slTension) {
            slTension.value = this.physicsParams.tension;
            slDamping.value = this.physicsParams.damping;
            slDiffusion.value = this.physicsParams.diffusion;
            
            if (slMaskThreshold) slMaskThreshold.value = this.maskThreshold;

            slEnv.value = this.lightingParams.env;
            slAz.value = this.lightingParams.az;
            slEl.value = this.lightingParams.el;
            slSpecCore.value = this.lightingParams.specCore;
            slSpecGlow.value = this.lightingParams.specGlow;
            slRim.value = this.lightingParams.rim;

            const updateParams = () => {
                this.physicsParams.tension = parseFloat(slTension.value);
                this.physicsParams.damping = parseFloat(slDamping.value);
                this.physicsParams.diffusion = parseFloat(slDiffusion.value);
                
                if (slMaskThreshold) this.maskThreshold = parseFloat(slMaskThreshold.value);

                this.lightingParams.env = parseFloat(slEnv.value);
                this.lightingParams.az = parseFloat(slAz.value);
                this.lightingParams.el = parseFloat(slEl.value);
                this.lightingParams.specCore = parseFloat(slSpecCore.value);
                this.lightingParams.specGlow = parseFloat(slSpecGlow.value);
                this.lightingParams.rim = parseFloat(slRim.value);

                const elT = document.getElementById('val-tension');
                const elD = document.getElementById('val-damping');
                const elDi = document.getElementById('val-diffusion');
                const elMT = document.getElementById('val-mask-threshold');
                
                const elE = document.getElementById('val-env');
                const elAz = document.getElementById('val-azimuth');
                const elEl = document.getElementById('val-elevation');
                const elSC = document.getElementById('val-spec-core');
                const elSG = document.getElementById('val-spec-glow');
                const elR = document.getElementById('val-rim');

                if (elT) elT.innerText = this.physicsParams.tension.toFixed(2);
                if (elD) elD.innerText = this.physicsParams.damping.toFixed(2);
                if (elDi) elDi.innerText = this.physicsParams.diffusion.toFixed(2);
                if (elMT) elMT.innerText = this.maskThreshold.toFixed(2);
                
                if (elE) elE.innerText = this.lightingParams.env.toFixed(1);
                if (elAz) elAz.innerText = this.lightingParams.az.toFixed(0);
                if (elEl) elEl.innerText = this.lightingParams.el.toFixed(0);
                if (elSC) elSC.innerText = this.lightingParams.specCore.toFixed(1);
                if (elSG) elSG.innerText = this.lightingParams.specGlow.toFixed(1);
                if (elR) elR.innerText = this.lightingParams.rim.toFixed(1);

                this.wake();
            };

            slTension.addEventListener('input', updateParams);
            slDamping.addEventListener('input', updateParams);
            slDiffusion.addEventListener('input', updateParams);
            
            if (slMaskThreshold) {
                slMaskThreshold.addEventListener('input', () => {
                    updateParams();
                    this.isDraggingMask = true;
                    if (this.loadedImage) this.recomputeMask(false);
                });
                slMaskThreshold.addEventListener('change', () => {
                    this.isDraggingMask = false;
                    this.entranceProgress = 0.0;
                    this.wake();
                });
            }

            slEnv.addEventListener('input', updateParams);
            slAz.addEventListener('input', updateParams);
            slEl.addEventListener('input', updateParams);
            slSpecCore.addEventListener('input', updateParams);
            slSpecGlow.addEventListener('input', updateParams);
            slRim.addEventListener('input', updateParams);
            
            // Sync initial state with DOM
            updateParams();
            this.boundUpdateParams = updateParams;
        }

        const updatePointer = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = 1.0 - (e.clientY - rect.top) / rect.height; // WebGL UV origin is bottom-left
            this.pointerPos = [x, y];
        };

        this.boundPointerDown = (e) => {
            this.canvas.setPointerCapture(e.pointerId);
            updatePointer(e);
            this.pointerForce = 1.0;
            this.isInteracting = true;
            this.wake();
        };

        this.boundPointerMove = (e) => {
            updatePointer(e);
            if (this.isInteracting) {
                this.pointerForce = 1.0;
            } else {
                this.pointerForce = 0.5;
            }
            this.wake();
        };

        this.boundPointerUp = (e) => {
            this.canvas.releasePointerCapture(e.pointerId);
            this.isInteracting = false;
            this.pointerForce = 0.5;
            this.wake();
        };

        this.boundPointerLeave = (e) => {
            if (!this.isInteracting) {
                this.pointerForce = 0.0;
                this.wake();
            }
        };

        this.canvas.addEventListener('pointerdown', this.boundPointerDown);
        this.canvas.addEventListener('pointermove', this.boundPointerMove);
        this.canvas.addEventListener('pointerup', this.boundPointerUp);
        this.canvas.addEventListener('pointerleave', this.boundPointerLeave);
    }

    destroy() {
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }

        if (this.boundUpdateParams) {
            const slTension = document.getElementById('slider-tension');
            const slDamping = document.getElementById('slider-damping');
            const slDiffusion = document.getElementById('slider-diffusion');
            const slEnv = document.getElementById('slider-env');
            const slAz = document.getElementById('slider-azimuth');
            const slEl = document.getElementById('slider-elevation');
            const slSpecCore = document.getElementById('slider-spec-core');
            const slSpecGlow = document.getElementById('slider-spec-glow');
            const slRim = document.getElementById('slider-rim');
            
            if (slTension) {
                slTension.removeEventListener('input', this.boundUpdateParams);
                slDamping.removeEventListener('input', this.boundUpdateParams);
                slDiffusion.removeEventListener('input', this.boundUpdateParams);
                slEnv.removeEventListener('input', this.boundUpdateParams);
                slAz.removeEventListener('input', this.boundUpdateParams);
                slEl.removeEventListener('input', this.boundUpdateParams);
                slSpecCore.removeEventListener('input', this.boundUpdateParams);
                slSpecGlow.removeEventListener('input', this.boundUpdateParams);
                slRim.removeEventListener('input', this.boundUpdateParams);
            }
        }

        if (this.boundPointerDown) this.canvas.removeEventListener('pointerdown', this.boundPointerDown);
        if (this.boundPointerMove) this.canvas.removeEventListener('pointermove', this.boundPointerMove);
        if (this.boundPointerUp) this.canvas.removeEventListener('pointerup', this.boundPointerUp);
        if (this.boundPointerLeave) this.canvas.removeEventListener('pointerleave', this.boundPointerLeave);

        const gl = this.gl;
        if (gl) {
            gl.deleteProgram(this.trimProgram);
            gl.deleteProgram(this.waveProgram);
            gl.deleteProgram(this.compositeProgram);
            
            gl.deleteBuffer(this.quadVBO);
            gl.deleteVertexArray(this.quadVAO);
            
            gl.deleteTexture(this.simA);
            gl.deleteTexture(this.simB);
            gl.deleteTexture(this.imageTex);
            
            gl.deleteFramebuffer(this.fboA);
            gl.deleteFramebuffer(this.fboB);
        }
    }

    swapPingPong() {
        let tempSim = this.simA;
        this.simA = this.simB;
        this.simB = tempSim;

        let tempFbo = this.fboA;
        this.fboA = this.fboB;
        this.fboB = tempFbo;
    }

    getUniformLocationCached(program, name) {
        if (!this.uniformLocs) {
            this.uniformLocs = new Map();
        }
        let programLocs = this.uniformLocs.get(program);
        if (!programLocs) {
            programLocs = new Map();
            this.uniformLocs.set(program, programLocs);
        }
        let loc = programLocs.get(name);
        if (loc === undefined) {
            loc = this.gl.getUniformLocation(program, name);
            programLocs.set(name, loc);
        }
        return loc;
    }

    runPass(program, destFbo, uniforms = {}) {
        const gl = this.gl;
        gl.useProgram(program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, destFbo);
        if (destFbo) {
            gl.viewport(0, 0, this.simRes, this.simRes);
        } else {
            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }

        gl.bindVertexArray(this.quadVAO);

        let texUnit = 0;
        for (const [name, val] of Object.entries(uniforms)) {
            const loc = this.getUniformLocationCached(program, name);
            if (!loc) continue;

            if (val instanceof WebGLTexture) {
                gl.activeTexture(gl.TEXTURE0 + texUnit);
                gl.bindTexture(gl.TEXTURE_2D, val);
                gl.uniform1i(loc, texUnit);
                texUnit++;
            } else if (Array.isArray(val)) {
                if (val.length === 2) gl.uniform2fv(loc, val);
                else if (val.length === 3) gl.uniform3fv(loc, val);
                else if (val.length === 4) gl.uniform4fv(loc, val);
            } else if (typeof val === 'number') {
                gl.uniform1f(loc, val);
            }
        }

        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    startTrimLoop() {
        const gpuThreshold = (1.0 - this.maskThreshold) * 0.6;

        // Run 150 passes to erode mask inward to shape boundaries and build SDF
        for (let i = 0; i < 150; i++) {
            this.runPass(this.trimProgram, this.fboB, {
                u_imageTexture: this.imageTex,
                u_simState: this.simA,
                u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
                u_imageTexelSize: [1.0 / this.loadedImage.width, 1.0 / this.loadedImage.height],
                u_gradientThreshold: gpuThreshold,
                u_hasTransparency: this.hasTransparency ? 1.0 : 0.0
            });
            this.swapPingPong();
        }
        
        this.isBaked = true;
        
        // Copy state to simB to initialize before wave loop starts
        this.runPass(this.waveProgram, this.fboB, {
            u_simState: this.simA,
            u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
            u_pointerPos: [0.0, 0.0],
            u_pointerForce: 0.0,
            u_pressure: 0.0
        });
        
        this.renderComposite();
    }

    wake() {
        if (!this.rafId) {
            this.consecutiveIdleFrames = 0;
            this.loop();
        }
    }

    loop() {
        let currentPressure = 0.05;
        if (this.entranceProgress < 1.0) {
            this.entranceProgress = Math.min(1.0, this.entranceProgress + 0.0125);
            currentPressure = 0.05 * this.entranceProgress;
            this.consecutiveIdleFrames = 0; // Keep awake during transition
        }

        // 1. Wave Solver Pass
        this.runPass(this.waveProgram, this.fboB, {
            u_simState: this.simA,
            u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
            u_pointerPos: this.pointerPos,
            u_pointerForce: this.pointerForce,
            u_tension: this.physicsParams ? this.physicsParams.tension : 0.5,
            u_damping: this.physicsParams ? this.physicsParams.damping : 0.75,
            u_diffusion: this.physicsParams ? this.physicsParams.diffusion : 0.15,
            u_pressure: currentPressure
        });

        if (this.isInteracting) {
            this.consecutiveIdleFrames = 0;
        } else {
            this.consecutiveIdleFrames++;
        }

        this.swapPingPong();

        // 2. Composite Pass
        this.renderComposite();

        // Pause loop if resting and the pop transition is complete
        if (this.consecutiveIdleFrames > 120 && this.entranceProgress >= 1.0) {
            this.rafId = null;
        } else {
            this.rafId = requestAnimationFrame(() => this.loop());
        }
    }

    renderComposite() {
        let lightDir = [-0.6, 0.6, 0.8];
        if (this.lightingParams) {
            let az = this.lightingParams.az * Math.PI / 180.0;
            let el = this.lightingParams.el * Math.PI / 180.0;
            
            let r_xy = Math.cos(el);
            let lx = Math.sin(az) * r_xy;
            let ly = Math.cos(az) * r_xy;
            let lz = Math.sin(el);
            
            lightDir = [lx, ly, lz];
        }

        this.runPass(this.compositeProgram, null, {
            u_imageTexture: this.imageTex,
            u_simState: this.simA,
            u_simTexelSize: [1.0 / this.simRes, 1.0 / this.simRes],
            u_screenTexelSize: [1.0 / this.canvas.width, 1.0 / this.canvas.height],
            u_showBoundary: this.isDraggingMask ? 1.0 : 0.0,
            u_envIntensity: this.lightingParams ? this.lightingParams.env : 1.4,
            u_lightDir: lightDir,
            u_specCore: this.lightingParams ? this.lightingParams.specCore : 6.7,
            u_specGlow: this.lightingParams ? this.lightingParams.specGlow : 1.0,
            u_rim: this.lightingParams ? this.lightingParams.rim : 0.5,
            u_diffusion: this.physicsParams ? this.physicsParams.diffusion : 0.15,
            u_inflationDepth: this.inflationDepth,
            u_entranceProgress: this.entranceProgress
        });
    }

    calculateOptimalThreshold(imgData) {
        const width = this.simRes;
        const height = this.simRes;
        const size = width * height;

        // 1. Compute raw luminance
        const rawLum = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            const r = imgData[i * 4] / 255;
            const g = imgData[i * 4 + 1] / 255;
            const b = imgData[i * 4 + 2] / 255;
            rawLum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }

        // 2. Smooth luminance with a 3x3 box filter to suppress noise
        const lum = new Float32Array(size);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                let sum = 0;
                let count = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    const ny = y + dy;
                    if (ny >= 0 && ny < height) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const nx = x + dx;
                            if (nx >= 0 && nx < width) {
                                sum += rawLum[ny * width + nx];
                                count++;
                            }
                        }
                    }
                }
                lum[idx] = sum / count;
            }
        }

        // 3. Compute gradients (using a symmetrical 2-pixel step)
        const grad = new Float32Array(size);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                const l_left  = lum[y * width + Math.max(0, x - 2)];
                const l_right = lum[y * width + Math.min(width - 1, x + 2)];
                const l_up    = lum[Math.min(height - 1, y + 2) * width + x];
                const l_down  = lum[Math.max(0, y - 2) * width + x];
                const dx = l_left - l_right;
                const dy = l_down - l_up;
                grad[idx] = Math.sqrt(dx * dx + dy * dy);
            }
        }

        // 4. Run flood-fill mask simulation for each candidate threshold
        const candidates = [];
        for (let t = 0.30; t <= 0.98; t += 0.02) {
            candidates.push(t);
        }

        const ratios = [];
        const queue = new Int32Array(size);

        for (let cIdx = 0; cIdx < candidates.length; cIdx++) {
            const threshold = candidates[cIdx];
            const gpuThreshold = (1.0 - threshold) * 0.6;

            const visited = new Uint8Array(size);
            let qHead = 0;
            let qTail = 0;

            // Seed queue unconditionally with absolute outer border pixels
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
                        visited[idx] = 1;
                        queue[qTail++] = idx;
                    }
                }
            }

            while (qHead < qTail) {
                const idx = queue[qHead++];
                const cx = idx % width;
                const cy = Math.floor(idx / width);

                const neighbors = [];
                if (cx > 0) neighbors.push(idx - 1);
                if (cx < width - 1) neighbors.push(idx + 1);
                if (cy > 0) neighbors.push(idx - width);
                if (cy < height - 1) neighbors.push(idx + width);

                for (let i = 0; i < neighbors.length; i++) {
                    const nIdx = neighbors[i];
                    if (!visited[nIdx] && grad[nIdx] < gpuThreshold) {
                        visited[nIdx] = 1;
                        queue[qTail++] = nIdx;
                    }
                }
            }

            ratios.push((size - qTail) / size);
        }

        // Find the plateau of minimal sensitivity, prioritizing mask sizes around 40-60% (closer to 60% preferred)
        let bestScore = Infinity;
        let bestThreshold = 0.75;

        for (let i = 1; i < candidates.length - 1; i++) {
            const ratio = ratios[i];
            // Ignore completely empty or full masks
            if (ratio >= 0.05 && ratio <= 0.95) {
                const sensitivity = Math.abs(ratios[i + 1] - ratios[i - 1]);
                
                // Calculate penalty based on target range of 40% to 60% (preferring 60%)
                let ratioPenalty = Math.abs(ratio - 0.60);
                if (ratio < 0.40) {
                    ratioPenalty += (0.40 - ratio) * 4.0; // Penalty grows for going below 40%
                } else if (ratio > 0.60) {
                    ratioPenalty += (ratio - 0.60) * 4.0; // Penalty grows for going above 60%
                }
                
                // Score combines sensitivity and ratio target proximity
                const score = sensitivity + ratioPenalty * 0.5;
                if (score < bestScore) {
                    bestScore = score;
                    bestThreshold = candidates[i];
                }
            }
        }

        console.log(`Auto-threshold tuned via sensitivity minimization: ${bestThreshold.toFixed(2)} (best score: ${bestScore.toFixed(4)})`);
        return bestThreshold;
    }

    recomputeMask(resetThreshold = true) {
        if (!this.loadedImage) return;

        const gl = this.gl;
        this.isBaked = false;
        
        // Detect transparent pixels
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = this.simRes;
        tmpCanvas.height = this.simRes;
        const ctx = tmpCanvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(this.loadedImage, 0, 0, this.simRes, this.simRes);
        const imgData = ctx.getImageData(0, 0, this.simRes, this.simRes).data;
        
        let hasTransparency = false;
        for (let i = 0; i < this.simRes * this.simRes; i++) {
            if (imgData[i * 4 + 3] < 240) {
                hasTransparency = true;
                break;
            }
        }
        this.hasTransparency = hasTransparency;

        // Compute average background color from the 1-pixel border margin
        let bgR = 0, bgG = 0, bgB = 0;
        let bgCount = 0;
        for (let y = 0; y < this.simRes; y++) {
            for (let x = 0; x < this.simRes; x++) {
                if (x === 0 || x === this.simRes - 1 || y === 0 || y === this.simRes - 1) {
                    const idx = y * this.simRes + x;
                    bgR += imgData[idx * 4];
                    bgG += imgData[idx * 4 + 1];
                    bgB += imgData[idx * 4 + 2];
                    bgCount++;
                }
            }
        }
        this.bgColor = [bgR / bgCount / 255, bgG / bgCount / 255, bgB / bgCount / 255];

        if (resetThreshold && !hasTransparency) {
            this.maskThreshold = this.calculateOptimalThreshold(imgData);
            const slMaskThreshold = document.getElementById('slider-mask-threshold');
            if (slMaskThreshold) {
                slMaskThreshold.value = this.maskThreshold;
                const elMT = document.getElementById('val-mask-threshold');
                if (elMT) elMT.innerText = this.maskThreshold.toFixed(2);
            }
        }

        // Initialize simulation state A with a bounding box mask
        const initData = new Float32Array(this.simRes * this.simRes * 4);
        for(let i = 0; i < this.simRes * this.simRes; i++) {
            const x = i % this.simRes;
            const y = Math.floor(i / this.simRes);
            
            let mask = 0.0;
            if (x > 0 && x < this.simRes - 1 && y > 0 && y < this.simRes - 1) {
                mask = 1.0;
                if (hasTransparency) {
                    const imgIdx = y * this.simRes + x;
                    if (imgData[imgIdx * 4 + 3] < 50) {
                        mask = 0.0;
                    }
                }
            }
            
            initData[i*4 + 0] = 0; // u_t
            initData[i*4 + 1] = 0; // u_t-1
            initData[i*4 + 2] = mask; // Mask
            initData[i*4 + 3] = mask; // SDF
        }
        
        gl.bindTexture(gl.TEXTURE_2D, this.simA);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.simRes, this.simRes, 0, gl.RGBA, gl.FLOAT, initData);
        
        gl.bindTexture(gl.TEXTURE_2D, this.simB);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.simRes, this.simRes, 0, gl.RGBA, gl.FLOAT, new Float32Array(this.simRes * this.simRes * 4));
        
        this.entranceProgress = 0.0;
        this.startTrimLoop();
        this.wake();
    }
}
