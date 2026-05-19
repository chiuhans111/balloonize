export const vertexShaderSource = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;
    // Image is flipped during texture upload (UNPACK_FLIP_Y_WEBGL)
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

    vec4 sourceImg = texture(u_imageTexture, v_uv);

    // Compute luminance gradient with a 2.0 texel step to filter high-frequency noise
    vec3 lumaCoef = vec3(0.299, 0.587, 0.114);
    float l_l = dot(texture(u_imageTexture, v_uv + vec2(-u_texelSize.x * 2.0, 0.0)).rgb, lumaCoef);
    float l_r = dot(texture(u_imageTexture, v_uv + vec2( u_texelSize.x * 2.0, 0.0)).rgb, lumaCoef);
    float l_u = dot(texture(u_imageTexture, v_uv + vec2(0.0,  u_texelSize.y * 2.0)).rgb, lumaCoef);
    float l_d = dot(texture(u_imageTexture, v_uv + vec2(0.0, -u_texelSize.y * 2.0)).rgb, lumaCoef);
    
    float grad = length(vec2(l_l - l_r, l_d - l_u));

    float m_l = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).b;
    float m_r = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).b;
    float m_u = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).b;
    float m_d = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).b;

    // Check if we are currently on the outer edge of the mask
    float edgeProximity = 4.0 - (m_l + m_r + m_u + m_d);
    
    if (edgeProximity > 0.0) {
        // Morphological Cleanup: instantly trim isolated background noise dots
        if (m_l + m_r + m_u + m_d <= 1.0) {
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

    // Eikonal solver for Euclidean SDF calculation
    if (m_l == 0.0 || m_r == 0.0 || m_u == 0.0 || m_d == 0.0) {
        currentSDF = 0.004; 
    } else {
        float s_l = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).a;
        float s_r = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).a;
        float s_u = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).a;
        float s_d = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).a;
        
        float step = 0.004;
        float h = min(s_l, s_r);
        float v = min(s_u, s_d);
        
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

    // Sample direct neighbors
    float u_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).r;
    float u_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).r;
    float u_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).r;
    float u_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).r;
    
    // Apply spatial diffusion to suppress high-frequency noise
    float localAvg = (u_left + u_right + u_up + u_down) * 0.25;
    u_t = mix(u_t, localAvg, u_diffusion); 
    
    float laplacian = (u_left + u_right + u_up + u_down) - 4.0 * u_t;
 
    // Sample diagonal neighbors for curvature computation
    float u_ul = texture(u_simState, v_uv + vec2(-u_texelSize.x,  u_texelSize.y)).r;
    float u_ur = texture(u_simState, v_uv + vec2( u_texelSize.x,  u_texelSize.y)).r;
    float u_dl = texture(u_simState, v_uv + vec2(-u_texelSize.x, -u_texelSize.y)).r;
    float u_dr = texture(u_simState, v_uv + vec2( u_texelSize.x, -u_texelSize.y)).r;
 
    // Gaussian curvature regularization
    float z_xx = u_right + u_left - 2.0 * u_t;
    float z_yy = u_up + u_down - 2.0 * u_t;
    float z_xy = (u_ur + u_dl - u_ul - u_dr) * 0.25;
    float K = (z_xx * z_yy) - (z_xy * z_xy);
    float developablePenalty = clamp(abs(K) * 400.0, 0.0, 0.15);
    
    float tension = u_tension * mask; 
    float pressure = u_pressure * mask;      
    // Scale down damping near the boundary (low SDF) to absorb wave energy at the sides
    float boundaryFade = smoothstep(0.0, 0.08, sdf);
    float damping = u_damping - developablePenalty;
    damping = mix(damping * 0.8, damping, boundaryFade);
    
    // Enforce 2D CFL stability condition: wave speed must be <= 1.0 / sqrt(2.0) ≈ 0.7071
    float c = tension * 0.7071;
    float acceleration = (c * c) * laplacian + pressure;
    float u_t_plus = 2.0 * u_t - u_t_minus + acceleration;
    
    // Low-pass filter for numerical stability
    float spatialSmooth = (u_left + u_right + u_up + u_down) * 0.25;
    u_t_plus = mix(u_t_plus, spatialSmooth, u_diffusion * 0.5);
    u_t_plus *= damping;

    // Interactive pointer deformation
    float brushRadius = 0.15;
    float distToPointer = length(v_uv - u_pointerPos);
    if (distToPointer < brushRadius && abs(u_pointerForce) > 0.0) {
        float dentShape = smoothstep(brushRadius, 0.0, distToPointer);
        dentShape = dentShape * dentShape;
        u_t_plus -= abs(u_pointerForce) * dentShape * mask * 0.6; 
    }

    // Apply plastic limit based on SDF and mask constraint
    float plastic_limit = pow(max(sdf, 1e-5), 0.35) * 6.5; 
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

// Bicubic texture sampler using Hermite interpolation
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

// 9-tap Gaussian blur filter
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

// Compute procedural wrinkles/creases near the balloon boundaries
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
    float baseInflation = 2.2; 
    return (wave + pow(max(sdf, 0.0), 0.65) * baseInflation) * u_inflationDepth * u_entranceProgress;
}

// Compute depth offsets along image high-contrast seams
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

// Procedural environment map representing studio lighting
vec3 getProceduralEnv(vec3 r) {
    float phi = atan(r.z, r.x);
    vec3 col = vec3(0.0);
    
    // Left window pane
    float winX = smoothstep(0.4, 0.42, sin(phi * 6.0));
    float winY = smoothstep(0.4, 0.42, sin(r.y * 12.0));
    float windowBounds = smoothstep(0.3, 0.8, r.x) * smoothstep(-0.5, 0.6, r.y);
    col += vec3(3.0, 3.2, 3.5) * winX * winY * windowBounds;
    
    // Right LED softbox
    float ledX = smoothstep(0.5, 0.9, sin(phi * 30.0));
    float ledY = smoothstep(0.5, 0.9, sin(r.y * 30.0));
    float boxBounds = smoothstep(0.4, 0.8, -r.x) * smoothstep(-0.4, 0.5, r.y);
    col += vec3(2.5, 2.0, 1.6) * ledX * ledY * boxBounds;
    
    // Horizon light slots
    float horizonMask = smoothstep(0.5, 0.0, abs(r.z));
    float windowSlots = smoothstep(0.3, 0.6, sin(phi * 12.0));
    col += vec3(2.0, 2.5, 3.0) * windowSlots * horizonMask;
    
    // Overhead light strip
    float overhead = smoothstep(0.8, 0.95, r.y) * smoothstep(0.7, 0.9, sin(r.x * 20.0));
    col += vec3(1.5, 1.8, 2.2) * overhead;
    
    // Ambient room lighting
    col += mix(vec3(0.02, 0.02, 0.04), vec3(0.1, 0.12, 0.15), smoothstep(-1.0, 1.0, r.y));
    return col;
}

void main() {
    // 1. Gaussian-filtered base simulation properties
    float blurRadius = 1.0 + u_diffusion * 6.0; 
    vec4 g_state = sampleGaussian(v_uv, u_simTexelSize, blurRadius);
    
    vec4 rawState = texture(u_simState, v_uv);
    float mask = rawState.b; 
    float sharpSDF = rawState.a;
    float sdf = g_state.a;
    float wave = g_state.r;
    
    // 2. Compute derivatives for shading normals
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

    // Macro-depth derivatives
    float h_left  = calcTotalDepth(g_left.r,  g_left.a);
    float h_right = calcTotalDepth(g_right.r, g_right.a);
    float h_up    = calcTotalDepth(g_up.r,    g_up.a);
    float h_down  = calcTotalDepth(g_down.r,  g_down.a);
    
    float dZdx_macro = (h_right - h_left) * 0.5 / offset.x; 
    float dZdy_macro = (h_up - h_down) * 0.5 / offset.y;

    // Seam-depth derivatives
    float s_left  = calcSeamDepth(warpedUV - vec2(u_simTexelSize.x, 0.0), sharpSDF);
    float s_right = calcSeamDepth(warpedUV + vec2(u_simTexelSize.x, 0.0), sharpSDF);
    float s_up    = calcSeamDepth(warpedUV + vec2(0.0, u_simTexelSize.y), sharpSDF);
    float s_down  = calcSeamDepth(warpedUV - vec2(0.0, u_simTexelSize.y), sharpSDF);
    
    float dZdx_seam = (s_right - s_left) * 0.5 / u_simTexelSize.x;
    float dZdy_seam = (s_up - s_down) * 0.5 / u_simTexelSize.y;
    
    // Crease derivatives
    float creases = getCreases(v_uv, wave, sdf, mask, tangent);
    float c_l = getCreases(v_uv - vec2(u_simTexelSize.x, 0.0), wave, sdf, mask, tangent);
    float c_r = getCreases(v_uv + vec2(u_simTexelSize.x, 0.0), wave, sdf, mask, tangent);
    float c_u = getCreases(v_uv + vec2(0.0, u_simTexelSize.y), wave, sdf, mask, tangent);
    float c_d = getCreases(v_uv - vec2(0.0, u_simTexelSize.y), wave, sdf, mask, tangent);

    float dZdx = dZdx_macro + dZdx_seam + (((c_r - c_l) * 0.5) / u_simTexelSize.x); 
    float dZdy = dZdy_macro + dZdy_seam + (((c_u - c_d) * 0.5) / u_simTexelSize.y);
    
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 28.0));
    
    // 3. Shading and specular highlights
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 refVec = reflect(-viewDir, normal);
    
    vec4 texColorRaw = texture(u_imageTexture, warpedUV);
    vec3 albedo = srgbToLinear(texColorRaw.rgb);
    vec3 studioEnv = getProceduralEnv(refVec) * u_envIntensity;
    
    vec3 pointLightDir = normalize(u_lightDir);
    float h_dot_l = max(dot(refVec, pointLightDir), 0.0);
    float specPointCore = pow(h_dot_l, 128.0) * u_specCore;   
    float specPointGlow = pow(h_dot_l, 24.0) * u_specGlow;    
    studioEnv += vec3(1.0, 0.96, 0.92) * (specPointCore + specPointGlow);
    
    // Rim lighting
    float NdotV = max(dot(normal, viewDir), 0.0);
    float rimGlow = pow(1.0 - NdotV, 1.2) * 1.2;
    vec3 rimColor = vec3(0.9, 0.95, 1.0) * rimGlow * u_rim * mask;

    // Ambient and diffuse shading
    float ndl = max(dot(normal, pointLightDir), 0.0);
    vec3 ambientComponent = albedo * (ndl * 0.55 + 0.12); 

    // Mylar fresnel model
    float fresnel = 0.15 + 0.85 * pow(1.0 - NdotV, 5.0);
    vec3 color = mix(ambientComponent, studioEnv, fresnel);
    color += vec3(3.5) * specPointCore * albedo; 
    color += rimColor;

    color = linearToSrgb(acesFilm(color));

    float currentInflation = u_inflationDepth * u_entranceProgress;
    color = mix(texColorRaw.rgb, color, currentInflation);

    float smoothedSDF = smoothstep(0.0, 0.06, sharpSDF);
    
    // Background composition
    vec3 pageBg = vec3(0.051, 0.051, 0.067);
    vec4 origTex = texture(u_imageTexture, v_uv);
    vec3 bgColor = mix(pageBg, origTex.rgb, origTex.a); 
    
    // Sample simulation mask at 1.5 screen pixels offset to create a very tight 1.5px border shadow
    float m1 = texture(u_simState, v_uv + vec2(-u_screenTexelSize.x * 1.5, 0.0)).b;
    float m2 = texture(u_simState, v_uv + vec2( u_screenTexelSize.x * 1.5, 0.0)).b;
    float m3 = texture(u_simState, v_uv + vec2(0.0, -u_screenTexelSize.y * 1.5)).b;
    float m4 = texture(u_simState, v_uv + vec2(0.0,  u_screenTexelSize.y * 1.5)).b;
    float avgMask = (m1 + m2 + m3 + m4) * 0.25;
    float borderShadow = avgMask * (1.0 - mask) * currentInflation;
    
    // Apply shadow at 35% opacity
    bgColor = mix(bgColor, vec3(0.0), borderShadow * 0.35);
    
    // Scanner ring entrance animation
    vec2 centerUV = vec2(0.5, 0.5);
    float dist = length(v_uv - centerUV);
    float ringRadius = u_entranceProgress * 1.1;
    float ringWidth = 0.18;
    float ring = smoothstep(ringWidth, 0.0, abs(dist - ringRadius));
    ring *= smoothstep(1.1, 0.2, dist);
    
    vec3 waveColor1 = vec3(1.0, 0.478, 0.349);   // Peach/Orange #ff7a59
    vec3 waveColor2 = vec3(1.0, 0.32, 0.48);     // Pink/Coral #ff527b
    vec3 waveColor3 = vec3(0.043, 0.576, 0.901); // Neon Blue #0b93e6
    
    vec3 waveCol = mix(waveColor1, waveColor2, sin(dist * 6.0 - u_entranceProgress * 5.0) * 0.5 + 0.5);
    waveCol = mix(waveCol, waveColor3, cos(dist * 4.0 + u_entranceProgress * 3.0) * 0.5 + 0.5);
    
    float waveFade = 1.0 - smoothstep(0.0, 1.0, u_entranceProgress);
    float waveIntensity = 0.9 * waveFade * u_inflationDepth; 
    
    // Apply drop shadow
    float shadowIntensity = smoothstep(0.0, 0.12, length(bleedOffset)) * mask * currentInflation;
    bgColor = mix(bgColor, vec3(0.0), shadowIntensity * 0.45);
    
    // Mask boundary visualization (during threshold adjustment)
    if (u_showBoundary > 0.5) {
        float m_center = rawState.b;
        color = texColorRaw.rgb;
        float outsideMask = 1.0 - m_center;
        bgColor = mix(bgColor, bgColor * 0.45, outsideMask);
        
        float m_left   = texture(u_simState, v_uv + vec2(-u_simTexelSize.x * 1.2, 0.0)).b;
        float m_right  = texture(u_simState, v_uv + vec2( u_simTexelSize.x * 1.2, 0.0)).b;
        float m_up     = texture(u_simState, v_uv + vec2(0.0,  u_simTexelSize.y * 1.2)).b;
        float m_down   = texture(u_simState, v_uv + vec2(0.0, -u_simTexelSize.y * 1.2)).b;
        float edge = max(max(abs(m_center - m_left), abs(m_center - m_right)), max(abs(m_center - m_up), abs(m_center - m_down)));
        
        vec3 themeColor = vec3(1.0, 0.368, 0.482);
        color = mix(color, themeColor, edge);
        bgColor = mix(bgColor, themeColor, edge);
    } else {
        color += waveCol * ring * waveIntensity * smoothedSDF;
    }
    
    fragColor = vec4(mix(bgColor, color, smoothedSDF), 1.0);
}
`;
