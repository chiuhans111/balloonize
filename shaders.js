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
uniform float u_gradientThreshold;

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

    // 1. Edge Erosion Logic
    float l_left  = dot(texture(u_imageTexture, v_uv + vec2(-u_texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_right = dot(texture(u_imageTexture, v_uv + vec2( u_texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_up    = dot(texture(u_imageTexture, v_uv + vec2(0.0,  u_texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float l_down  = dot(texture(u_imageTexture, v_uv + vec2(0.0, -u_texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    
    float grad = length(vec2(l_left - l_right, l_down - l_up));

    float m_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).b;
    float m_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).b;
    float m_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).b;
    float m_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).b;

    float edgeProximity = 4.0 - (m_left + m_right + m_up + m_down);
    if (edgeProximity > 0.0 && grad < u_gradientThreshold) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0); // Trim this pixel out
        return;
    }

    // 2. Manhattan Distance Field Generation (Plastic Structural Limit)
    if (m_left == 0.0 || m_right == 0.0 || m_up == 0.0 || m_down == 0.0) {
        currentSDF = 0.004; // Edge boundary
    } else {
        float s_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).a;
        float s_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).a;
        float s_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).a;
        float s_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).a;
        
        // Interior pixel grows distance based on neighbors
        float minSDF = min(min(s_left, s_right), min(s_up, s_down));
        currentSDF = minSDF + 0.004;
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
    
    float u_ul = texture(u_simState, v_uv + vec2(-u_texelSize.x,  u_texelSize.y)).r;
    float u_ur = texture(u_simState, v_uv + vec2( u_texelSize.x,  u_texelSize.y)).r;
    float u_dl = texture(u_simState, v_uv + vec2(-u_texelSize.x, -u_texelSize.y)).r;
    float u_dr = texture(u_simState, v_uv + vec2( u_texelSize.x, -u_texelSize.y)).r;
    
    float laplacian = (u_left + u_right + u_up + u_down) - 4.0 * u_t;

    // --- GAUSSIAN CURVATURE REGULARIZER ---
    float z_xx = u_right + u_left - 2.0 * u_t;
    float z_yy = u_up + u_down - 2.0 * u_t;
    float z_xy = (u_ur + u_dl - u_ul - u_dr) * 0.25;

    float K = (z_xx * z_yy) - (z_xy * z_xy);
    float developablePenalty = clamp(abs(K) * 500.0, 0.0, 0.2);
    
    // Tuned: update faster, damp a bit
    float tension = 0.5 * mask;        
    float pressure = 0.05 * mask;      
    
    float damping = 0.82 - developablePenalty;              
    
    float acceleration = (tension * tension) * laplacian + pressure;
    float u_t_plus = 2.0 * u_t - u_t_minus + acceleration;
    u_t_plus *= damping;

    float brushRadius = 0.15;
    float distToPointer = length(v_uv - u_pointerPos);
    if (distToPointer < brushRadius && abs(u_pointerForce) > 0.0) {
        float dentShape = pow(1.0 - (distToPointer / brushRadius), 2.0);
        u_t_plus -= abs(u_pointerForce) * dentShape * mask * 2.0; 
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

float calcTotalDepth(float wave, float sdf) {
    float baseInflation = 2.8; 
    float safeSDF = max(sdf, 0.0); 
    return wave + pow(safeSDF, 0.4) * baseInflation;
}

float calcSeamDepth(vec2 uv, float sdf) {
    vec2 offset = u_simTexelSize * 1.5;
    vec3 c_left   = texture(u_imageTexture, uv - vec2(offset.x, 0.0)).rgb;
    vec3 c_right  = texture(u_imageTexture, uv + vec2(offset.x, 0.0)).rgb;
    vec3 c_up     = texture(u_imageTexture, uv + vec2(0.0, offset.y)).rgb;
    vec3 c_down   = texture(u_imageTexture, uv - vec2(0.0, offset.y)).rgb;
    
    float dx = length(c_right - c_left);
    float dy = length(c_up - c_down);
    float edge = smoothstep(0.1, 0.5, length(vec2(dx, dy)));
    
    return -pow(edge, 1.5) * 0.15 * smoothstep(0.01, 0.06, sdf);
}

void main() {
    vec4 state = sampleSmooth(u_simState, v_uv, u_simTexelSize);
    float mask = state.b; float sdf = state.a;
    
    // 4-way cross sampling for low-pass box filtering
    vec4 st_left  = sampleSmooth(u_simState, v_uv - vec2(u_simTexelSize.x, 0.0), u_simTexelSize);
    vec4 st_right = sampleSmooth(u_simState, v_uv + vec2(u_simTexelSize.x, 0.0), u_simTexelSize);
    vec4 st_up    = sampleSmooth(u_simState, v_uv + vec2(0.0, u_simTexelSize.y), u_simTexelSize);
    vec4 st_down  = sampleSmooth(u_simState, v_uv - vec2(0.0, u_simTexelSize.y), u_simTexelSize);
    
    // Low-pass filtered wave channel to keep surface texture perfectly smooth
    float wave = (state.r + st_left.r + st_right.r + st_up.r + st_down.r) * 0.2;

    vec2 slope_normal = normalize(vec2(st_left.a - st_right.a, st_down.a - st_up.a) + 0.0001);
    vec2 bleedOffset = slope_normal * clamp(wave * 2.0, -1.0, 1.0) * u_simTexelSize;
    vec2 warpedUV = v_uv - bleedOffset;

    // Build the clean composite depth field
    float h_left  = calcTotalDepth(st_left.r,  st_left.a)  + calcSeamDepth(warpedUV - vec2(u_simTexelSize.x, 0.0), st_left.a);
    float h_right = calcTotalDepth(st_right.r, st_right.a) + calcSeamDepth(warpedUV + vec2(u_simTexelSize.x, 0.0), st_right.a);
    float h_up    = calcTotalDepth(st_up.r,    st_up.a)    + calcSeamDepth(warpedUV + vec2(0.0, u_simTexelSize.y), st_up.a);
    float h_down  = calcTotalDepth(st_down.r,  st_down.a)  + calcSeamDepth(warpedUV - vec2(0.0, u_simTexelSize.y), st_down.a);
    
    float dZdx = (h_right - h_left) * 0.5; 
    float dZdy = (h_up - h_down) * 0.5;
    
    // Solid, highly inflated normal distribution
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 0.25));
    
    // --- HIGH-CONTRAST DUAL STUDIO LIGHTING MATRIX ---
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 refVec = reflect(-viewDir, normal);
    
    vec4 texColorRaw = texture(u_imageTexture, warpedUV);
    vec3 albedo = srgbToLinear(texColorRaw.rgb);
    
    vec3 studioEnv = vec3(0.0);
    
    // 1. Intense Starburst Point Light (Top-Left Position)
    vec3 pointLightDir = normalize(vec3(-0.6, 0.6, 0.8));
    float specPoint = pow(max(dot(refVec, pointLightDir), 0.0), 300.0);
    studioEnv += vec3(12.0, 11.0, 10.0) * specPoint;
    
    // 2. Main Softbox Panel Reflection (Top-Right Angle)
    float softbox1 = smoothstep(0.7, 0.95, dot(refVec, normalize(vec3(0.7, 0.5, 0.4))));
    studioEnv += vec3(2.0, 1.8, 1.5) * softbox1;
    
    // 3. Cool Floor Bounce (Bottom Hemispherical Lighting)
    float softbox2 = smoothstep(0.3, 0.9, dot(refVec, normalize(vec3(0.0, -0.9, 0.4))));
    studioEnv += vec3(0.3, 0.5, 0.8) * softbox2;
    
    // 4. Pure Analytical Rim Edge Glow (High acceptance angle)
    float rimGlow = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
    studioEnv += vec3(0.8, 0.9, 1.0) * rimGlow * smoothstep(-0.2, 1.0, refVec.y) * 0.5;

    // High-contrast deep base shadows
    float ndl = max(dot(normal, pointLightDir), 0.0);
    vec3 ambientComponent = albedo * (ndl * 0.5 + 0.1); 

    // Pure Plastic Fresnel
    float NdotV = max(dot(normal, viewDir), 0.0);
    float fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - NdotV, 5.0);
    
    // Master Composite
    vec3 color = ambientComponent;
    color = mix(color, studioEnv, fresnel);
    
    // Add direct specular light bloom back onto the mesh surface for visual pop
    color += vec3(5.0) * specPoint * albedo;

    color = linearToSrgb(acesFilm(color));

    float highResAlpha = texColorRaw.a;
    float smoothedSDF = smoothstep(0.0, 0.06, sdf);
    float finalAlphaMask = max(highResAlpha, smoothedSDF);

    vec3 bgColor = vec3(0.04, 0.03, 0.04); 
    float shadowIntensity = smoothstep(0.0, 0.12, length(bleedOffset)) * mask;
    bgColor = mix(bgColor, vec3(0.0), shadowIntensity * 0.95);
    
    fragColor = vec4(mix(bgColor, color, finalAlphaMask), 1.0);
}
`;
