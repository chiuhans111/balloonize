import {
    vertexShaderSource,
    trimFragmentShaderSource,
    waveFragmentShaderSource,
    compositeFragmentShaderSource
} from './shaders.js';

export class BalloonizeEngine {
    constructor(canvas, imageSource, options = {}) {
        this.canvas = canvas;
        // Need webgl2 for float textures and texture() in glsl
        this.gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
        if (!this.gl) throw new Error("WebGL2 not supported");

        // Enable float textures for ping-pong buffer
        if (!this.gl.getExtension('EXT_color_buffer_float')) {
            console.warn("EXT_color_buffer_float not supported");
        }

        this.pointerPos = [0.5, 0.5];
        this.pointerForce = 0.0;
        this.isInteracting = false;
        
        this.simRes = 256;
        this.rafId = null;
        this.consecutiveIdleFrames = 0;
        
        this.inflationDepth = options.inflationDepth !== undefined ? options.inflationDepth : 1.0;
        this.entranceProgress = 1.0;
        this.maskThreshold = options.maskThreshold !== undefined ? options.maskThreshold : 0.05;

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

        this.debugParams = {
            laplacian: true,
            puff: true,
            pointer: true,
            curvature: true,
            diffusion: true
        };
        
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
        // Ping-Pong pair for SimulationPingPong (RGBA32F, 256x256)
        // R: Current Scalar Wave (u_t), G: Previous (u_t-1), B: Mask, A: SDF
        this.simA = this.createTexture(this.simRes, this.simRes, gl.RGBA32F, gl.RGBA, gl.FLOAT);
        this.simB = this.createTexture(this.simRes, this.simRes, gl.RGBA32F, gl.RGBA, gl.FLOAT);
        
        // Placeholder for ImageTexture (RGBA8)
        this.imageTex = this.createTexture(1, 1, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255,255,255,255]));
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

                const maxCanvasDim = 512;
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
        const slEffect = document.getElementById('slider-effect');
        const slTension = document.getElementById('slider-tension');
        const slDamping = document.getElementById('slider-damping');
        const slDiffusion = document.getElementById('slider-diffusion');
        
        const slEnv = document.getElementById('slider-env');
        const slAz = document.getElementById('slider-azimuth');
        const slEl = document.getElementById('slider-elevation');
        const slSpecCore = document.getElementById('slider-spec-core');
        const slSpecGlow = document.getElementById('slider-spec-glow');
        const slRim = document.getElementById('slider-rim');
        const slMaskThreshold = document.getElementById('slider-mask-threshold');

        const chkLaplacian = document.getElementById('chk-laplacian');
        const chkPuff = document.getElementById('chk-puff');
        const chkPointer = document.getElementById('chk-pointer');
        const chkCurvature = document.getElementById('chk-curvature');
        const chkDiffusion = document.getElementById('chk-diffusion');
        
        if (slTension) {
            // Initialize slider values to match the current parameter values (from constructor options)
            if (slEffect) slEffect.value = this.inflationDepth;
            slTension.value = this.physicsParams.tension;
            slDamping.value = this.physicsParams.damping;
            slDiffusion.value = this.physicsParams.diffusion;
            
            slEnv.value = this.lightingParams.env;
            slAz.value = this.lightingParams.az;
            slEl.value = this.lightingParams.el;
            slSpecCore.value = this.lightingParams.specCore;
            slSpecGlow.value = this.lightingParams.specGlow;
            slRim.value = this.lightingParams.rim;
            if (slMaskThreshold) slMaskThreshold.value = this.maskThreshold;

            if (chkLaplacian) {
                chkLaplacian.checked = this.debugParams.laplacian;
                chkPuff.checked = this.debugParams.puff;
                chkPointer.checked = this.debugParams.pointer;
                chkCurvature.checked = this.debugParams.curvature;
                chkDiffusion.checked = this.debugParams.diffusion;
            }
 
            const updateParams = () => {
                if (slEffect) this.inflationDepth = parseFloat(slEffect.value);
                this.physicsParams.tension = parseFloat(slTension.value);
                this.physicsParams.damping = parseFloat(slDamping.value);
                this.physicsParams.diffusion = parseFloat(slDiffusion.value);
                
                this.lightingParams.env = parseFloat(slEnv.value);
                this.lightingParams.az = parseFloat(slAz.value);
                this.lightingParams.el = parseFloat(slEl.value);
                this.lightingParams.specCore = parseFloat(slSpecCore.value);
                this.lightingParams.specGlow = parseFloat(slSpecGlow.value);
                this.lightingParams.rim = parseFloat(slRim.value);
                
                let maskThresholdChanged = false;
                if (slMaskThreshold) {
                    const newVal = parseFloat(slMaskThreshold.value);
                    if (newVal !== this.maskThreshold) {
                        this.maskThreshold = newVal;
                        maskThresholdChanged = true;
                    }
                }

                if (chkLaplacian) {
                    this.debugParams.laplacian = chkLaplacian.checked;
                    this.debugParams.puff = chkPuff.checked;
                    this.debugParams.pointer = chkPointer.checked;
                    this.debugParams.curvature = chkCurvature.checked;
                    this.debugParams.diffusion = chkDiffusion.checked;
                }
 
                const elEf = document.getElementById('val-effect');
                const elT = document.getElementById('val-tension');
                const elD = document.getElementById('val-damping');
                const elDi = document.getElementById('val-diffusion');
                const elE = document.getElementById('val-env');
                const elAz = document.getElementById('val-azimuth');
                const elEl = document.getElementById('val-elevation');
                const elSC = document.getElementById('val-spec-core');
                const elSG = document.getElementById('val-spec-glow');
                const elR = document.getElementById('val-rim');
                const elMT = document.getElementById('val-mask-threshold');
 
                if (elEf) elEf.innerText = this.inflationDepth.toFixed(2);
                if (elT) elT.innerText = this.physicsParams.tension.toFixed(2);
                if (elD) elD.innerText = this.physicsParams.damping.toFixed(2);
                if (elDi) elDi.innerText = this.physicsParams.diffusion.toFixed(2);
                
                if (elE) elE.innerText = this.lightingParams.env.toFixed(1);
                if (elAz) elAz.innerText = this.lightingParams.az.toFixed(0);
                if (elEl) elEl.innerText = this.lightingParams.el.toFixed(0);
                if (elSC) elSC.innerText = this.lightingParams.specCore.toFixed(1);
                if (elSG) elSG.innerText = this.lightingParams.specGlow.toFixed(1);
                if (elR) elR.innerText = this.lightingParams.rim.toFixed(1);
                if (elMT) elMT.innerText = this.maskThreshold.toFixed(2);
 
                if (maskThresholdChanged) {
                    this.recomputeMask(false);
                } else {
                    this.wake();
                }
            };
            if (slEffect) slEffect.addEventListener('input', updateParams);
            slTension.addEventListener('input', updateParams);
            slDamping.addEventListener('input', updateParams);
            slDiffusion.addEventListener('input', updateParams);
            
            slEnv.addEventListener('input', updateParams);
            slAz.addEventListener('input', updateParams);
            slEl.addEventListener('input', updateParams);
            slSpecCore.addEventListener('input', updateParams);
            slSpecGlow.addEventListener('input', updateParams);
            slRim.addEventListener('input', updateParams);
            if (slMaskThreshold) slMaskThreshold.addEventListener('input', updateParams);

            if (chkLaplacian) {
                chkLaplacian.addEventListener('change', updateParams);
                chkPuff.addEventListener('change', updateParams);
                chkPointer.addEventListener('change', updateParams);
                chkCurvature.addEventListener('change', updateParams);
                chkDiffusion.addEventListener('change', updateParams);
            }
            
            // Call once immediately to sync JS state with DOM (fixes browser form caching)
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
            this.pointerForce = 1.0; // Stronger click (force x2)
            this.isInteracting = true;
            this.wake();
        };

        this.boundPointerMove = (e) => {
            updatePointer(e);
            if (this.isInteracting) {
                this.pointerForce = 1.0; // Dragging state (deep)
            } else {
                this.pointerForce = 0.5; // Hover state (used to be click state)
            }
            this.wake();
        };

        this.boundPointerUp = (e) => {
            this.canvas.releasePointerCapture(e.pointerId);
            this.isInteracting = false;
            this.pointerForce = 0.5; // Revert to hover state
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
        // Cancel animation loop
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }

        // Remove event listeners
        if (this.boundUpdateParams) {
            const slEffect = document.getElementById('slider-effect');
            const slTension = document.getElementById('slider-tension');
            const slDamping = document.getElementById('slider-damping');
            const slDiffusion = document.getElementById('slider-diffusion');
            const slEnv = document.getElementById('slider-env');
            const slAz = document.getElementById('slider-azimuth');
            const slEl = document.getElementById('slider-elevation');
            const slSpecCore = document.getElementById('slider-spec-core');
            const slSpecGlow = document.getElementById('slider-spec-glow');
            const slRim = document.getElementById('slider-rim');
            const slMaskThreshold = document.getElementById('slider-mask-threshold');
            
            if (slTension) {
                if (slEffect) slEffect.removeEventListener('input', this.boundUpdateParams);
                slTension.removeEventListener('input', this.boundUpdateParams);
                slDamping.removeEventListener('input', this.boundUpdateParams);
                slDiffusion.removeEventListener('input', this.boundUpdateParams);
                slEnv.removeEventListener('input', this.boundUpdateParams);
                slAz.removeEventListener('input', this.boundUpdateParams);
                slEl.removeEventListener('input', this.boundUpdateParams);
                slSpecCore.removeEventListener('input', this.boundUpdateParams);
                slSpecGlow.removeEventListener('input', this.boundUpdateParams);
                slRim.removeEventListener('input', this.boundUpdateParams);
                if (slMaskThreshold) slMaskThreshold.removeEventListener('input', this.boundUpdateParams);
            }
        }

        if (this.boundPointerDown) this.canvas.removeEventListener('pointerdown', this.boundPointerDown);
        if (this.boundPointerMove) this.canvas.removeEventListener('pointermove', this.boundPointerMove);
        if (this.boundPointerUp) this.canvas.removeEventListener('pointerup', this.boundPointerUp);
        if (this.boundPointerLeave) this.canvas.removeEventListener('pointerleave', this.boundPointerLeave);

        // Delete WebGL objects to prevent GPU memory leaks
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
            const loc = gl.getUniformLocation(program, name);
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
        // Run 150 passes to erode the mask inward to the shape boundaries and build SDF
        for (let i = 0; i < 150; i++) {
            this.runPass(this.trimProgram, this.fboB, {
                u_imageTexture: this.imageTex,
                u_simState: this.simA,
                u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
                u_maskThreshold: this.maskThreshold,
                u_hasTransparency: this.hasTransparency ? 1.0 : 0.0,
                u_maxBgDiff: this.maxBgDiff !== undefined ? this.maxBgDiff : 0.5
            });
            this.swapPingPong();
        }
        
        // CRITICAL FIX: Force simB to perfectly match simA before the interactive wave loop starts
        // We run a dummy pass that just copies the state over.
        this.runPass(this.waveProgram, this.fboB, {
            u_simState: this.simA,
            u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
            u_pointerPos: [0.0, 0.0],
            u_pointerForce: 0.0,
            u_pressure: 0.0,
            u_enableLaplacian: 0.0,
            u_enablePuff: 0.0,
            u_enablePointer: 0.0,
            u_enableCurvature: 0.0,
            u_enableDiffusion: 0.0
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
        if (this.entranceProgress === undefined) this.entranceProgress = 1.0;

        let currentPressure = 0.06;
        if (this.entranceProgress < 1.0) {
            this.entranceProgress = Math.min(1.0, this.entranceProgress + 0.0125); // Pop breath and scan wave over 80 frames (~1.3s)
            // Over-inflate slightly (1.1x) to give it a "stretchy" pop look
            currentPressure = 0.06 * (this.entranceProgress * 1.1); 
            
            this.consecutiveIdleFrames = 0; // Prevent sleeping during the pop transition
        }
        // Decay the force automatically so the bubble doesn't "feed" on itself
        if (this.isInteracting) {
            // Keep the force alive but decaying slightly while dragging
            this.pointerForce = Math.max(0.1, this.pointerForce * 0.95);
        } else {
            // Decay the hover force to 0.0 when mouse stops moving
            this.pointerForce = Math.max(0.0, this.pointerForce * 0.92);
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
            u_pressure: currentPressure,
            u_enableLaplacian: this.debugParams.laplacian ? 1.0 : 0.0,
            u_enablePuff: this.debugParams.puff ? 1.0 : 0.0,
            u_enablePointer: this.debugParams.pointer ? 1.0 : 0.0,
            u_enableCurvature: this.debugParams.curvature ? 1.0 : 0.0,
            u_enableDiffusion: this.debugParams.diffusion ? 1.0 : 0.0
        });

        if (this.isInteracting) {
            this.consecutiveIdleFrames = 0;
        } else {
            this.consecutiveIdleFrames++;
        }

        this.swapPingPong();

        // 2. Composite Pass
        this.renderComposite();

        // Zero-Idle Sleep Engine: pause loop if resting AND the pop breath is finished (entranceProgress === 1.0)
        if (this.consecutiveIdleFrames > 120 && this.entranceProgress >= 1.0) {
            this.rafId = null;
        } else {
            this.rafId = requestAnimationFrame(() => this.loop());
        }
    }

    renderComposite() {
        let lightDir = [-0.6, 0.6, 0.8]; // default if not set
        if (this.lightingParams) {
            let az = this.lightingParams.az * Math.PI / 180.0;
            let el = this.lightingParams.el * Math.PI / 180.0;
            
            // Azimuth acts like a clock on the XY plane
            // Elevation pulls the light out of the screen (Z axis)
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

    calculateOptimalThreshold(imgData, w, h) {
        // 1. Sample 4 corners slightly inset to establish the global background color
        const corners = [
            (Math.floor(h * 0.05) * w + Math.floor(w * 0.05)) * 4,
            (Math.floor(h * 0.05) * w + Math.floor(w * 0.95)) * 4,
            (Math.floor(h * 0.95) * w + Math.floor(w * 0.05)) * 4,
            (Math.floor(h * 0.95) * w + Math.floor(w * 0.95)) * 4
        ];
        
        let bgR = 0, bgG = 0, bgB = 0;
        corners.forEach(idx => {
            bgR += imgData[idx]; bgG += imgData[idx+1]; bgB += imgData[idx+2];
        });
        bgR /= 4; bgG /= 4; bgB /= 4;

        // 2. Calculate Background Color Differences (for Indestructible Core Failsafe)
        let maxBgDist = 0;
        for (let i = 0; i < imgData.length; i += 4) {
            let r = imgData[i], g = imgData[i+1], b = imgData[i+2];
            let rMean = (r + bgR) * 0.5;
            let dR = r - bgR, dG = g - bgG, dB = b - bgB;
            let weightR = 2.0 + rMean / 256.0;
            let weightG = 4.0;
            let weightB = 2.0 + (255.0 - rMean) / 256.0;
            
            let dist = Math.sqrt(weightR * dR * dR + weightG * dG * dG + weightB * dB * dB);
            if (dist > maxBgDist) maxBgDist = dist;
        }
        this.maxBgDiff = maxBgDist / 765.0; // Normalized to 0.0 - 1.0 color distance

        // 3. Calculate Perceptual Gradients (Sobel-style approximation)
        let gradHist = new Array(256).fill(0);
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                let idx = (y * w + x) * 4;
                let idxR = (y * w + (x + 1)) * 4;
                let idxD = ((y + 1) * w + x) * 4;

                let r1 = imgData[idx], g1 = imgData[idx+1], b1 = imgData[idx+2];
                let rR = imgData[idxR], gR = imgData[idxR+1], bR = imgData[idxR+2];
                let rD = imgData[idxD], gD = imgData[idxD+1], bD = imgData[idxD+2];

                let dXr = r1 - rR, dXg = g1 - gR, dXb = b1 - bR;
                let dYr = r1 - rD, dYg = g1 - gD, dYb = b1 - bD;

                let gradX = Math.sqrt(dXr * dXr + dXg * dXg + dXb * dXb);
                let gradY = Math.sqrt(dYr * dYr + dYg * dYg + dYb * dYb);
                let gradMagnitude = Math.sqrt(gradX * gradX + gradY * gradY);

                let bin = Math.min(255, Math.floor((gradMagnitude / 442.0) * 255));
                gradHist[bin]++;
            }
        }

        // 4. MSER-Inspired Stability Search (Finding the Plateau)
        // We look for the threshold 't' where changing 't' adds the fewest new edge pixels.
        // This corresponds to a deep valley in the gradient histogram.
        
        // Apply a slight Gaussian smooth to the histogram to ignore micro-spikes
        let smoothedHist = new Array(256).fill(0);
        for(let i = 2; i < 254; i++) {
            smoothedHist[i] = (gradHist[i-2] + gradHist[i-1]*2 + gradHist[i]*4 + gradHist[i+1]*2 + gradHist[i+2]) / 10.0;
        }

        let bestThresholdBin = 10; // Start slightly above 0 to ignore raw JPEG noise floor
        let minChange = Infinity;
        let stabilityRun = 0;
        
        // Scan from the noise floor up to the 60% mark
        for (let i = 15; i < 180; i++) {
            let rateOfChange = smoothedHist[i]; 
            
            if (rateOfChange < minChange) {
                minChange = rateOfChange;
                bestThresholdBin = i;
                stabilityRun = 0;
            } else if (rateOfChange === minChange) {
                // If we are on a perfectly flat plateau, pick the middle of the plateau
                stabilityRun++;
                bestThresholdBin = i - Math.floor(stabilityRun / 2);
            }
        }
        
        // Return normalized threshold with a slight safety buffer pushing it to the upper edge of the plateau,
        // enforcing a minimum of 0.05.
        const result = Math.max(0.05, (bestThresholdBin + 15) / 255.0);
        console.log("Calculated optimal MSER threshold:", result);
        return result;
    }

    recomputeMask(resetThreshold = true) {
        if (!this.loadedImage) return;

        const gl = this.gl;
        
        // Detect if the uploaded image has transparent pixels
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

        if (resetThreshold) {
            // Calculate SOTA Perceptual Otsu Threshold on CPU
            this.optimalOtsuThreshold = this.calculateOptimalThreshold(imgData, this.simRes, this.simRes);
            this.maskThreshold = Math.min(0.5, this.optimalOtsuThreshold);
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
            
            const margin = this.simRes * 0.05;
            let mask = 0.0;
            if (x > margin && x < this.simRes - margin && y > margin && y < this.simRes - margin) {
                mask = 1.0;
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
