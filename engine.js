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
        this.gl = canvas.getContext('webgl2');
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

        // Generalize editable parameters
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

                if (isObjectURL) {
                    URL.revokeObjectURL(src);
                }

                // Run trim simulation and wake render loop
                this.startTrimLoop();
                this.wake();
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
        
        const slEnv = document.getElementById('slider-env');
        const slAz = document.getElementById('slider-azimuth');
        const slEl = document.getElementById('slider-elevation');
        const slSpecCore = document.getElementById('slider-spec-core');
        const slSpecGlow = document.getElementById('slider-spec-glow');
        const slRim = document.getElementById('slider-rim');
        
        if (slTension) {
            // Initialize slider values to match the current parameter values (from constructor options)
            slTension.value = this.physicsParams.tension;
            slDamping.value = this.physicsParams.damping;
            slDiffusion.value = this.physicsParams.diffusion;
            
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
                
                this.lightingParams.env = parseFloat(slEnv.value);
                this.lightingParams.az = parseFloat(slAz.value);
                this.lightingParams.el = parseFloat(slEl.value);
                this.lightingParams.specCore = parseFloat(slSpecCore.value);
                this.lightingParams.specGlow = parseFloat(slSpecGlow.value);
                this.lightingParams.rim = parseFloat(slRim.value);

                const elT = document.getElementById('val-tension');
                const elD = document.getElementById('val-damping');
                const elDi = document.getElementById('val-diffusion');
                const elE = document.getElementById('val-env');
                const elAz = document.getElementById('val-azimuth');
                const elEl = document.getElementById('val-elevation');
                const elSC = document.getElementById('val-spec-core');
                const elSG = document.getElementById('val-spec-glow');
                const elR = document.getElementById('val-rim');

                if (elT) elT.innerText = this.physicsParams.tension.toFixed(2);
                if (elD) elD.innerText = this.physicsParams.damping.toFixed(2);
                if (elDi) elDi.innerText = this.physicsParams.diffusion.toFixed(2);
                
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
            
            slEnv.addEventListener('input', updateParams);
            slAz.addEventListener('input', updateParams);
            slEl.addEventListener('input', updateParams);
            slSpecCore.addEventListener('input', updateParams);
            slSpecGlow.addEventListener('input', updateParams);
            slRim.addEventListener('input', updateParams);
            
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
                u_gradientThreshold: 0.15 // Slightly higher to ensure it stops at sharp edges
            });
            this.swapPingPong();
        }
        
        // CRITICAL FIX: Force simB to perfectly match simA before the interactive wave loop starts
        // We run a dummy pass that just copies the state over.
        this.runPass(this.waveProgram, this.fboB, {
            u_simState: this.simA,
            u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
            u_pointerPos: [0.0, 0.0],
            u_pointerForce: 0.0
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
        // 1. Wave Solver Pass
        this.runPass(this.waveProgram, this.fboB, {
            u_simState: this.simA,
            u_texelSize: [1.0 / this.simRes, 1.0 / this.simRes],
            u_pointerPos: this.pointerPos,
            u_pointerForce: this.pointerForce,
            u_tension: this.physicsParams ? this.physicsParams.tension : 0.5,
            u_damping: this.physicsParams ? this.physicsParams.damping : 0.75,
            u_diffusion: this.physicsParams ? this.physicsParams.diffusion : 0.15
        });

        if (this.isInteracting) {
            this.consecutiveIdleFrames = 0;
        } else {
            this.consecutiveIdleFrames++;
        }

        this.swapPingPong();

        // 2. Composite Pass
        this.renderComposite();

        // Zero-Idle Sleep Engine: pause loop if resting
        if (this.consecutiveIdleFrames > 120) {
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
            u_diffusion: this.physicsParams ? this.physicsParams.diffusion : 0.15
        });
    }
}
