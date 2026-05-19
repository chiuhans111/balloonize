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

    float u_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).r;
    float u_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).r;
    float u_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).r;
    float u_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).r;
    
    float laplacian = (u_left + u_right + u_up + u_down) - 4.0 * u_t;

    // FAT BALLOON PHYSICS
    float tension = 0.4 * mask;        
    float pressure = 0.05 * mask;      // 10x higher pressure: keeps the balloon tightly inflated
    float damping = 0.85;              // High damping instantly kills watery ripples
    
    float acceleration = (tension * tension) * laplacian + pressure;

    float u_t_plus = 2.0 * u_t - u_t_minus + acceleration;
    u_t_plus *= damping;

    // THICK POINTER SQUISH
    float brushRadius = 0.15;
    float distToPointer = length(v_uv - u_pointerPos);
    
    if (distToPointer < brushRadius && abs(u_pointerForce) > 0.0) {
        float dentShape = pow(1.0 - (distToPointer / brushRadius), 2.0);
        u_t_plus -= abs(u_pointerForce) * dentShape * mask * 2.0; 
    }

    // PERFECT DOME CONSTRAINT
    // Lower exponent (0.35) makes the volume rise aggressively at the edges 
    // and flatten at the top, creating a "full/stuffed" look.
    float plastic_limit = pow(sdf, 0.35) * 6.5; 
    
    if (u_t_plus > plastic_limit) {
        u_t_plus = plastic_limit; 
    }

    if (sdf <= 0.0) {
        u_t_plus = 0.0;
    }

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

// 1. UPGRADED: HERMITE CUBIC INTERPOLATION
// By applying a smoothstep curve to the fractional coordinate, we upgrade 
// bilinear interpolation to smooth cubic interpolation. This makes the 1st derivative 
// (the normals) continuous, completely eliminating faceted/pixelated specular highlights!
vec4 sampleSmooth(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 pixel = uv / texSize - 0.5;
    vec2 f = fract(pixel);
    
    // The magic line: Smooths the interpolation slope
    f = f * f * (3.0 - 2.0 * f); 
    
    vec2 p0 = (floor(pixel) + 0.5) * texSize;
    
    vec4 c00 = texture(tex, p0);
    vec4 c10 = texture(tex, p0 + vec2(texSize.x, 0.0));
    vec4 c01 = texture(tex, p0 + vec2(0.0, texSize.y));
    vec4 c11 = texture(tex, p0 + texSize);
    
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

float calcTotalDepth(float wave, float sdf) {
    float edgeThickness = 30.0; // Looser edge thickness for smaller icons
    float baseInflation = 0.85; // Massive increase in base artificial volume
    float edge = clamp(sdf * edgeThickness, 0.0, 1.0);
    float artificialDome = sqrt(max(0.0, 1.0 - pow(1.0 - edge, 2.0))) * baseInflation;
    return wave + artificialDome;
}

void main() {
    vec4 state = sampleSmooth(u_simState, v_uv, u_simTexelSize);
    float wave = state.r;
    float mask = state.b;
    float sdf = state.a;
    
    vec4 st_left  = sampleSmooth(u_simState, v_uv - vec2(u_simTexelSize.x, 0.0), u_simTexelSize);
    vec4 st_right = sampleSmooth(u_simState, v_uv + vec2(u_simTexelSize.x, 0.0), u_simTexelSize);
    vec4 st_up    = sampleSmooth(u_simState, v_uv + vec2(0.0, u_simTexelSize.y), u_simTexelSize);
    vec4 st_down  = sampleSmooth(u_simState, v_uv - vec2(0.0, u_simTexelSize.y), u_simTexelSize);
    
    vec2 edge_normal = normalize(vec2(st_left.b - st_right.b, st_down.b - st_up.b) + 0.0001);

    vec2 bleedOffset = edge_normal * clamp(wave * 2.0, -1.0, 1.0) * u_simTexelSize;
    vec2 warpedUV = v_uv - bleedOffset;

    vec2 tangent = vec2(-edge_normal.y, edge_normal.x);
    float crimpPhase = dot(v_uv * 180.0, tangent);
    float borderZone = smoothstep(1.0, 0.5, mask) * smoothstep(0.0, 0.3, mask);
    float creases = sin(crimpPhase + wave * 8.0) * borderZone * 0.04;

    float h_left  = calcTotalDepth(st_left.r,  st_left.a);
    float h_right = calcTotalDepth(st_right.r, st_right.a);
    float h_up    = calcTotalDepth(st_up.r,    st_up.a);
    float h_down  = calcTotalDepth(st_down.r,  st_down.a);
    
    float dZdx = ((h_right - h_left) * 0.5) + dFdx(creases); 
    float dZdy = ((h_up - h_down) * 0.5) + dFdy(creases);
    
    // Lower Z-weight (0.07) forces the lighting engine to treat the slopes 
    // as incredibly steep, maximizing the shiny, bursting, high-tension feel
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 0.07));
    
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 mainLightDir = normalize(vec3(0.3, 0.7, 0.8));
    vec3 halfMain = normalize(mainLightDir + viewDir);
    float specMain = pow(max(dot(normal, halfMain), 0.0), 300.0) * 2.5; 
    
    vec3 boxLightDir = normalize(vec3(-0.4, 0.6, 0.5));
    vec3 halfBox = normalize(boxLightDir + viewDir);
    float specBox = pow(max(dot(normal, halfBox), 0.0), 40.0) * 0.8;

    vec3 rimLightDir = normalize(vec3(-0.8, -0.2, -0.5));
    vec3 halfRim = normalize(rimLightDir + viewDir);
    float specRim = pow(max(dot(normal, halfRim), 0.0), 60.0) * 0.4;
    
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 4.0);

    vec4 texColor = texture(u_imageTexture, warpedUV);
    float diffuse = max(dot(normal, mainLightDir), 0.0);
    
    vec3 finalColor = texColor.rgb * (diffuse * 0.4 + 0.7);
    finalColor += vec3(specMain + specBox + specRim);
    finalColor += (texColor.rgb + vec3(0.8)) * fresnel * 0.8;

    // 2. PERFECT HIGH-RES EDGES (Source Alpha Masking)
    // Instead of cutting the shape out using the low-res 256x256 SDF, 
    // we extract the pristine, native anti-aliased alpha channel from the source image.
    // Because it is sampled using 'warpedUV', the perfect edge moves organically with the physics!
    float highResAlpha = texColor.a;
    
    // Fallback mask in case the input image is a solid JPEG with no alpha channel
    float smoothedSDF = smoothstep(0.0, 0.06, sdf);
    
    // Use the cleanest available mask
    float finalAlphaMask = max(highResAlpha, smoothedSDF);

    // Render Contact Shadow & Background
    vec3 bgColor = vec3(0.1); // Match the body background
    float shadowIntensity = smoothstep(0.0, 0.15, length(bleedOffset)) * mask;
    
    // Darken background underneath the balloon based on distance
    bgColor = mix(bgColor, vec3(0.0), shadowIntensity * 0.8);
    
    // Cleanly composite the balloon over the background using our perfect high-res alpha
    finalColor = mix(bgColor, finalColor, finalAlphaMask);

    fragColor = vec4(finalColor, 1.0);
}
`;
