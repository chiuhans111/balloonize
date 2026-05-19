export const vertexShaderSource = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;
    // Flip Y to match typical image coordinates if needed, 
    // but WebGL native bottom-left origin usually expects standard mapping.
    // We'll flip the image during texture upload (UNPACK_FLIP_Y_WEBGL).
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

export const trimFragmentShaderSource = `#version 300 es
precision highp float;
uniform sampler2D u_imageTexture;
uniform sampler2D u_simState;
uniform vec2 u_texelSize;
uniform vec2 u_imageTexelSize;
uniform float u_gradientThreshold;
uniform float u_hasTransparency;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 currentState = texture(u_simState, v_uv);
    float currentMask = currentState.b;
    float currentSDF = currentState.a;
    
    if (currentMask == 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Fetch the actual source image pixel
    vec4 sourceImg = texture(u_imageTexture, v_uv);

    // 1. Shrink-Wrap Erosion Logic (sampled at simulation texel scale with 2x step for noise suppression)
    float l_left  = dot(texture(u_imageTexture, v_uv + vec2(-u_texelSize.x * 2.0, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_right = dot(texture(u_imageTexture, v_uv + vec2( u_texelSize.x * 2.0, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_up    = dot(texture(u_imageTexture, v_uv + vec2(0.0,  u_texelSize.y * 2.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_down  = dot(texture(u_imageTexture, v_uv + vec2(0.0, -u_texelSize.y * 2.0)).rgb, vec3(0.299, 0.587, 0.114));
    
    float grad = length(vec2(l_left - l_right, l_down - l_up));

    float m_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).b;
    float m_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).b;
    float m_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).b;
    float m_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).b;

    // Check if we are currently on the outer edge of the mask
    float edgeProximity = 4.0 - (m_left + m_right + m_up + m_down);
    
    if (edgeProximity > 0.0) {
        // Morphological Cleanup: instantly trim isolated background noise dots
        if (m_left + m_right + m_up + m_down <= 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }

        // HYBRID STOPPING CONDITION:
        // Keep eating the mask ONLY if we haven't hit the shape boundary.
        // For transparent images (PNGs), stop instantly when hitting opaque pixels (alpha >= 0.1).
        // For flattened images (JPEGs), fallback to the luminance gradient threshold.
        bool keepEating = (u_hasTransparency > 0.5) ? (sourceImg.a < 0.1) : (grad < u_gradientThreshold);
        if (keepEating) {
            fragColor = vec4(0.0, 0.0, 0.0, 0.0); // Trim this pixel out
            return;
        }
    }

    // 2. TRUE EUCLIDEAN SDF (Eikonal Equation Solver)
    if (m_left == 0.0 || m_right == 0.0 || m_up == 0.0 || m_down == 0.0) {
        currentSDF = 0.004; // Edge boundary
    } else {
        float s_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).a;
        float s_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).a;
        float s_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).a;
        float s_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).a;
        
        float step = 0.004;
        float h = min(s_left, s_right);
        float v = min(s_up, s_down);
        
        // Rouy-Tourin scheme for smooth, non-faceted corners
        if (abs(h - v) < step) {
            currentSDF = 0.5 * (h + v + sqrt(max(0.0, 2.0 * step * step - (h - v) * (h - v))));
        } else {
            currentSDF = min(h, v) + step;
        }
    }

    fragColor = vec4(currentState.r, currentState.g, 1.0, currentSDF);
}
`;

export const waveFragmentShaderSource = `#version 300 es
precision highp float;
uniform sampler2D u_simState;
uniform vec2 u_texelSize;
uniform vec2 u_pointerPos;
uniform float u_pointerForce;
uniform float u_tension;
uniform float u_damping;
uniform float u_diffusion;
uniform float u_pressure;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 center = texture(u_simState, v_uv);
    float u_t = center.r;
    float u_t_minus = center.g;
    float mask = center.b;
    float sdf = center.a; 

    // 3x3 Kernel for the Hessian Matrix
    float u_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).r;
    float u_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).r;
    float u_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).r;
    float u_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).r;
    
    // DIFFUSION: Kill high-frequency noise (ringing) by slightly averaging with neighbors
    float localAvg = (u_left + u_right + u_up + u_down) * 0.25;
    u_t = mix(u_t, localAvg, u_diffusion); 
    
    // Re-sample neighbors based on diffused center for stable laplacian
    float laplacian = (u_left + u_right + u_up + u_down) - 4.0 * u_t;
 
    float u_ul = texture(u_simState, v_uv + vec2(-u_texelSize.x,  u_texelSize.y)).r;
    float u_ur = texture(u_simState, v_uv + vec2( u_texelSize.x,  u_texelSize.y)).r;
    float u_dl = texture(u_simState, v_uv + vec2(-u_texelSize.x, -u_texelSize.y)).r;
    float u_dr = texture(u_simState, v_uv + vec2( u_texelSize.x, -u_texelSize.y)).r;
 
    // --- GAUSSIAN CURVATURE REGULARIZER ---
    float z_xx = u_right + u_left - 2.0 * u_t;
    float z_yy = u_up + u_down - 2.0 * u_t;
    float z_xy = (u_ur + u_dl - u_ul - u_dr) * 0.25;
 
    float K = (z_xx * z_yy) - (z_xy * z_xy);
    float developablePenalty = clamp(abs(K) * 400.0, 0.0, 0.15); // Softened penalty
    
    float tension = u_tension * mask; 
    float pressure = u_pressure * mask;      
    
    float damping = u_damping - developablePenalty;              
    
    float acceleration = (tension * tension) * laplacian + pressure;
    float u_t_plus = 2.0 * u_t - u_t_minus + acceleration;
    
    // Low-pass spatial filter blend executed directly inside the solver pass
    // to instantly extinguish micro-scale numerical grid snapping noise
    float spatialSmooth = (u_left + u_right + u_up + u_down) * 0.25;
    u_t_plus = mix(u_t_plus, spatialSmooth, u_diffusion * 0.5); // use diffusion to control numerical solver damping as well!
    
    u_t_plus *= damping;

    float brushRadius = 0.15;
    float distToPointer = length(v_uv - u_pointerPos);
    if (distToPointer < brushRadius && abs(u_pointerForce) > 0.0) {
        float dentShape = smoothstep(brushRadius, 0.0, distToPointer);
        dentShape = dentShape * dentShape; // Square for a much softer, cushioned spatial profile
        u_t_plus -= abs(u_pointerForce) * dentShape * mask * 0.6; 
    }

    float plastic_limit = pow(sdf, 0.35) * 6.5; 
    if (u_t_plus > plastic_limit) u_t_plus = plastic_limit; 
    if (sdf <= 0.0) u_t_plus = 0.0;

    fragColor = vec4(u_t_plus, u_t, mask, sdf);
}
`;

export const compositeFragmentShaderSource = `#version 300 es
precision highp float;
uniform sampler2D u_imageTexture;
uniform sampler2D u_simState;
uniform vec2 u_simTexelSize;
uniform vec2 u_screenTexelSize;
uniform float u_envIntensity;
uniform vec3 u_lightDir;
uniform float u_specCore;
uniform float u_specGlow;
uniform float u_rim;
uniform float u_showBoundary;
uniform float u_diffusion;
uniform float u_inflationDepth;
uniform float u_entranceProgress;

in vec2 v_uv;
out vec4 fragColor;

vec3 srgbToLinear(vec3 color) { return pow(color, vec3(2.2)); }
vec3 linearToSrgb(vec3 color) { return pow(color, vec3(1.0 / 2.2)); }
vec3 acesFilm(vec3 x) {
    float a = 2.51f; float b = 0.03f; float c = 2.43f; float d = 0.59f; float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// HERMITE CUBIC SMOOTH SAMPLER
vec4 sampleSmooth(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 pixel = uv / texSize - 0.5;
    vec2 f = fract(pixel);
    f = f * f * (3.0 - 2.0 * f); 
    vec2 p0 = (floor(pixel) + 0.5) * texSize;
    vec4 c00 = texture(tex, p0);
    vec4 c10 = texture(tex, p0 + vec2(texSize.x, 0.0));
    vec4 c01 = texture(tex, p0 + vec2(0.0, texSize.y));
    vec4 c11 = texture(tex, p0 + texSize);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// 9-TAP GAUSSIAN KERNEL
vec4 sampleGaussian(vec2 uv, vec2 texSize, float radius) {
    vec2 off = texSize * radius;
    vec4 sum = vec4(0.0);
    sum += sampleSmooth(u_simState, uv + vec2(-off.x, -off.y), texSize) * 0.0625;
    sum += sampleSmooth(u_simState, uv + vec2( 0.0,   -off.y), texSize) * 0.1250;
    sum += sampleSmooth(u_simState, uv + vec2( off.x, -off.y), texSize) * 0.0625;
    sum += sampleSmooth(u_simState, uv + vec2(-off.x,  0.0),   texSize) * 0.1250;
    sum += sampleSmooth(u_simState, uv + vec2( 0.0,    0.0),   texSize) * 0.2500;
    sum += sampleSmooth(u_simState, uv + vec2( off.x,  0.0),   texSize) * 0.1250;
    sum += sampleSmooth(u_simState, uv + vec2(-off.x,  off.y), texSize) * 0.0625;
    sum += sampleSmooth(u_simState, uv + vec2( 0.0,    off.y), texSize) * 0.1250;
    sum += sampleSmooth(u_simState, uv + vec2( off.x,  off.y), texSize) * 0.0625;
    return sum;
}

// Crease Function
float getCreases(vec2 uv, float wave, float sdf, float mask, vec2 tangent) {
    vec2 noiseOffset = vec2(sin(uv.y * 12.0 + wave), cos(uv.x * 12.0 - wave)) * 0.03;
    vec2 perturbedUV = uv + noiseOffset;
    float crimpPhase = dot(perturbedUV * 75.0, tangent);
    float inwardFadeEnd = 0.12 + sin(uv.x * 15.0) * cos(uv.y * 15.0) * 0.05;
    float buckleZone = smoothstep(inwardFadeEnd, 0.0, sdf) * smoothstep(-0.02, 0.02, sdf) * mask;
    float ridge = 1.0 - pow(abs(sin(crimpPhase + wave * 2.0)), 1.5);
    return (ridge * 2.0 - 1.0) * buckleZone * 0.065 * u_inflationDepth * u_entranceProgress;
}

float calcTotalDepth(float wave, float sdf) {
    float baseInflation = 2.5; 
    return (wave + pow(max(sdf, 0.0), 0.50) * baseInflation) * u_inflationDepth * u_entranceProgress;
}

float calcSeamDepth(vec2 uv, float sdf) {
    vec2 offset = u_simTexelSize * 1.5;
    vec3 c_left   = texture(u_imageTexture, uv - vec2(offset.x, 0.0)).rgb;
    vec3 c_right  = texture(u_imageTexture, uv + vec2(offset.x, 0.0)).rgb;
    vec3 c_up     = texture(u_imageTexture, uv + vec2(0.0, offset.y)).rgb;
    vec3 c_down   = texture(u_imageTexture, uv - vec2(0.0, offset.y)).rgb;
    float dx = length(c_right - c_left);
    float dy = length(c_up - c_down);
    return -pow(smoothstep(0.1, 0.5, length(vec2(dx, dy))), 1.5) * 0.12 * smoothstep(0.01, 0.06, sdf) * u_inflationDepth * u_entranceProgress;
}

// Detailed Procedural Studio Environment
vec3 getProceduralEnv(vec3 r) {
    float phi = atan(r.z, r.x);
    vec3 col = vec3(0.0);
    
    // Left Studio Window Pane Structure
    float winX = smoothstep(0.4, 0.42, sin(phi * 6.0));
    float winY = smoothstep(0.4, 0.42, sin(r.y * 12.0));
    float windowBounds = smoothstep(0.3, 0.8, r.x) * smoothstep(-0.5, 0.6, r.y);
    col += vec3(3.0, 3.2, 3.5) * winX * winY * windowBounds;
    
    // Right Softbox LED Matrix
    float ledX = smoothstep(0.5, 0.9, sin(phi * 30.0));
    float ledY = smoothstep(0.5, 0.9, sin(r.y * 30.0));
    float boxBounds = smoothstep(0.4, 0.8, -r.x) * smoothstep(-0.4, 0.5, r.y);
    col += vec3(2.5, 2.0, 1.6) * ledX * ledY * boxBounds;
    
    // Horizon repeating window slots (wrapping edge)
    float horizonMask = smoothstep(0.5, 0.0, abs(r.z));
    float windowSlots = smoothstep(0.3, 0.6, sin(phi * 12.0));
    col += vec3(2.0, 2.5, 3.0) * windowSlots * horizonMask;
    
    // Overhead light strip
    float overhead = smoothstep(0.8, 0.95, r.y) * smoothstep(0.7, 0.9, sin(r.x * 20.0));
    col += vec3(1.5, 1.8, 2.2) * overhead;
    
    // Ambient room fill
    col += mix(vec3(0.02, 0.02, 0.04), vec3(0.1, 0.12, 0.15), smoothstep(-1.0, 1.0, r.y));
    return col;
}

void main() {
    // 1. GAUSSIAN BASE PASS
    float blurRadius = 1.0 + u_diffusion * 6.0; 
    vec4 g_state = sampleGaussian(v_uv, u_simTexelSize, blurRadius);
    
    // Keep raw mask and SDF sharp for perfect boundary silhouettes
    vec4 rawState = texture(u_simState, v_uv);
    float mask = rawState.b; 
    float sharpSDF = rawState.a;
    float sdf = g_state.a;
    float wave = g_state.r;
    
    // 2. GEOMETRIC DERIVATIVE PASS
    float offsetMult = 2.0 + u_diffusion * 8.0;
    vec2 offset = u_simTexelSize * offsetMult;
    
    vec4 g_left  = sampleGaussian(v_uv - vec2(offset.x, 0.0), u_simTexelSize, blurRadius);
    vec4 g_right = sampleGaussian(v_uv + vec2(offset.x, 0.0), u_simTexelSize, blurRadius);
    vec4 g_up    = sampleGaussian(v_uv + vec2(0.0, offset.y), u_simTexelSize, blurRadius);
    vec4 g_down  = sampleGaussian(v_uv - vec2(0.0, offset.y), u_simTexelSize, blurRadius);
    
    vec2 slope_normal = normalize(vec2(g_left.a - g_right.a, g_down.a - g_up.a) + 0.0001);
    vec2 tangent = vec2(-slope_normal.y, slope_normal.x);
    vec2 bleedOffset = slope_normal * clamp(wave * 2.0, -1.0, 1.0) * u_simTexelSize;
    vec2 warpedUV = v_uv - bleedOffset;

    // Macro-depth (smooth balloon dome geometry)
    float h_left  = calcTotalDepth(g_left.r,  g_left.a);
    float h_right = calcTotalDepth(g_right.r, g_right.a);
    float h_up    = calcTotalDepth(g_up.r,    g_up.a);
    float h_down  = calcTotalDepth(g_down.r,  g_down.a);
    
    float dZdx_macro = (h_right - h_left) * 0.5 / offset.x; 
    float dZdy_macro = (h_up - h_down) * 0.5 / offset.y;

    // Decoupled tight-scale derivatives (keeps seams and creases sharp)
    float s_left  = calcSeamDepth(warpedUV - vec2(u_simTexelSize.x, 0.0), sharpSDF);
    float s_right = calcSeamDepth(warpedUV + vec2(u_simTexelSize.x, 0.0), sharpSDF);
    float s_up    = calcSeamDepth(warpedUV + vec2(0.0, u_simTexelSize.y), sharpSDF);
    float s_down  = calcSeamDepth(warpedUV - vec2(0.0, u_simTexelSize.y), sharpSDF);
    
    float dZdx_seam = (s_right - s_left) * 0.5 / u_simTexelSize.x;
    float dZdy_seam = (s_up - s_down) * 0.5 / u_simTexelSize.y;
    
    float creases = getCreases(v_uv, wave, sdf, mask, tangent);
    float c_l = getCreases(v_uv - vec2(u_simTexelSize.x, 0.0), wave, sdf, mask, tangent);
    float c_r = getCreases(v_uv + vec2(u_simTexelSize.x, 0.0), wave, sdf, mask, tangent);
    float c_u = getCreases(v_uv + vec2(0.0, u_simTexelSize.y), wave, sdf, mask, tangent);
    float c_d = getCreases(v_uv - vec2(0.0, u_simTexelSize.y), wave, sdf, mask, tangent);

    float dZdx = dZdx_macro + dZdx_seam + (((c_r - c_l) * 0.5) / u_simTexelSize.x); 
    float dZdy = dZdy_macro + dZdy_seam + (((c_u - c_d) * 0.5) / u_simTexelSize.y);
    
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 28.0));
    
    // 3. LIGHTING AND SHADING
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 refVec = reflect(-viewDir, normal);
    
    vec4 texColorRaw = texture(u_imageTexture, warpedUV);
    vec3 albedo = srgbToLinear(texColorRaw.rgb);
    
    // Detailed room environment reflections
    vec3 studioEnv = getProceduralEnv(refVec) * u_envIntensity;
    
    // Specular Highlight
    vec3 pointLightDir = normalize(u_lightDir);
    float h_dot_l = max(dot(refVec, pointLightDir), 0.0);
    float specPointCore = pow(h_dot_l, 128.0) * u_specCore;   
    float specPointGlow = pow(h_dot_l, 24.0) * u_specGlow;    
    studioEnv += vec3(1.0, 0.96, 0.92) * (specPointCore + specPointGlow);
    
    // Backlit Rim Glow (Removed from studioEnv to prevent Fresnel squashing)
    float NdotV = max(dot(normal, viewDir), 0.0);
    float rimGlow = pow(1.0 - NdotV, 1.2) * 1.2; // Softened exponent (1.2 instead of 4.0) to cover more inwards
    vec3 rimColor = vec3(0.9, 0.95, 1.0) * rimGlow * u_rim * mask;

    // Diffuse Ambient component
    float ndl = max(dot(normal, pointLightDir), 0.0);
    vec3 ambientComponent = albedo * (ndl * 0.55 + 0.12); 

    // Enhanced Plastic/Mylar Fresnel (0.15 base reflectivity for glossy finish)
    float fresnel = 0.15 + 0.85 * pow(1.0 - NdotV, 5.0);
    vec3 color = mix(ambientComponent, studioEnv, fresnel);
    color += vec3(3.5) * specPointCore * albedo; 
    color += rimColor; // Added outside Fresnel mix so it can wrap inward freely

    color = linearToSrgb(acesFilm(color));

    // Mix flat unshaded original texture with shaded 3D color
    // This allows it to quickly breathe into 3D on entrance (u_entranceProgress: 0 -> 1)
    // while still respecting the manual slider (u_inflationDepth)
    float currentInflation = u_inflationDepth * u_entranceProgress;
    color = mix(texColorRaw.rgb, color, currentInflation);

    float smoothedSDF = smoothstep(0.0, 0.06, sharpSDF); // Keep edge sharp
    
    // Blend transparent background pixels with the HTML page background color (#0d0d11)
    vec3 pageBg = vec3(0.051, 0.051, 0.067); // Match body bg #0d0d11
    vec4 origTex = texture(u_imageTexture, v_uv);
    vec3 bgColor = mix(pageBg, origTex.rgb, origTex.a); 
    
    // AI gradient circle wave out (one-time expanding wave ring inside the balloon region)
    vec2 centerUV = vec2(0.5, 0.5);
    float dist = length(v_uv - centerUV);
    
    // Ring expands from center (radius 0.0 to 1.1)
    float ringRadius = u_entranceProgress * 1.1;
    float ringWidth = 0.18; // Balanced soft volumetric glow band
    float ring = smoothstep(ringWidth, 0.0, abs(dist - ringRadius));
    
    // Soft radial fade near center and far borders
    ring *= smoothstep(1.1, 0.2, dist);
    
    // Wave colors
    vec3 waveColor1 = vec3(1.0, 0.478, 0.349);   // Peach/Orange #ff7a59
    vec3 waveColor2 = vec3(1.0, 0.32, 0.48);     // Pink/Coral #ff527b
    vec3 waveColor3 = vec3(0.043, 0.576, 0.901); // Neon Blue #0b93e6
    
    vec3 waveCol = mix(waveColor1, waveColor2, sin(dist * 6.0 - u_entranceProgress * 5.0) * 0.5 + 0.5);
    waveCol = mix(waveCol, waveColor3, cos(dist * 4.0 + u_entranceProgress * 3.0) * 0.5 + 0.5);
    
    // Fade the wave intensity out to 0.0 as u_entranceProgress reaches 1.0
    // S-curve smoothstep fade for soft trailing decay
    float waveFade = 1.0 - smoothstep(0.0, 1.0, u_entranceProgress);
    float waveIntensity = 0.9 * waveFade * u_inflationDepth; 
    
    // Add the bright glowing scanner ring directly onto the balloon color
    color += waveCol * ring * waveIntensity * smoothedSDF;
    
    // Apply soft drop shadow to the background (diminishes as balloon flattens)
    float shadowIntensity = smoothstep(0.0, 0.12, length(bleedOffset)) * mask * currentInflation;
    bgColor = mix(bgColor, vec3(0.0), shadowIntensity * 0.45);
    
    // True selection contour (while dragging mask threshold slider)
    if (u_showBoundary > 0.5) {
        float m_center = rawState.b;
        
        // 1. Force the balloon to remain flat (original texture)
        color = texColorRaw.rgb;
        
        // 2. Dim the outside of the mask (where m_center is 0.0) by 55%
        float outsideMask = 1.0 - m_center;
        bgColor = mix(bgColor, bgColor * 0.45, outsideMask);
        
        // 3. Draw a single, clean, sharp theme-colored contour line (1.2 texels)
        float m_left   = texture(u_simState, v_uv + vec2(-u_simTexelSize.x * 1.2, 0.0)).b;
        float m_right  = texture(u_simState, v_uv + vec2( u_simTexelSize.x * 1.2, 0.0)).b;
        float m_up     = texture(u_simState, v_uv + vec2(0.0,  u_simTexelSize.y * 1.2)).b;
        float m_down   = texture(u_simState, v_uv + vec2(0.0, -u_simTexelSize.y * 1.2)).b;
        float edge = max(max(abs(m_center - m_left), abs(m_center - m_right)), max(abs(m_center - m_up), abs(m_center - m_down)));
        
        // Theme pink/coral color matching the app UI accent (#ff5e7b)
        vec3 themeColor = vec3(1.0, 0.368, 0.482);
        
        // Overlay the contour cleanly on top
        color = mix(color, themeColor, edge);
        bgColor = mix(bgColor, themeColor, edge);
    } else {
        // Add the bright glowing scanner ring directly onto the balloon color (only when NOT dragging)
        color += waveCol * ring * waveIntensity * smoothedSDF;
    }
    
    fragColor = vec4(mix(bgColor, color, smoothedSDF), 1.0);
}
`;
