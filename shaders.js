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
    // Using a square root of the distance field creates a mathematically perfect, fat round dome
    float plastic_limit = pow(sdf, 0.5) * 5.0; 
    
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

// 1. MANUAL BILINEAR FILTER
// Smooths the low-res 256x256 NEAREST texture into a continuous high-res field
vec4 sampleSmooth(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 pixel = uv / texSize - 0.5;
    vec2 f = fract(pixel);
    vec2 p0 = (floor(pixel) + 0.5) * texSize;
    
    vec4 c00 = texture(tex, p0);
    vec4 c10 = texture(tex, p0 + vec2(texSize.x, 0.0));
    vec4 c01 = texture(tex, p0 + vec2(0.0, texSize.y));
    vec4 c11 = texture(tex, p0 + texSize);
    
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    // Read perfectly smoothed states instead of blocky pixels
    vec4 state = sampleSmooth(u_simState, v_uv, u_simTexelSize);
    float wave = state.r;
    float mask = state.b;
    float sdf = state.a;
    
    float m_left  = sampleSmooth(u_simState, v_uv - vec2(u_simTexelSize.x, 0.0), u_simTexelSize).b;
    float m_right = sampleSmooth(u_simState, v_uv + vec2(u_simTexelSize.x, 0.0), u_simTexelSize).b;
    float m_up    = sampleSmooth(u_simState, v_uv + vec2(0.0, u_simTexelSize.y), u_simTexelSize).b;
    float m_down  = sampleSmooth(u_simState, v_uv - vec2(0.0, u_simTexelSize.y), u_simTexelSize).b;
    
    vec2 edge_normal = normalize(vec2(m_left - m_right, m_down - m_up) + 0.0001);

    vec2 bleedOffset = edge_normal * clamp(wave * 2.0, -1.0, 1.0) * u_simTexelSize;
    vec2 warpedUV = v_uv - bleedOffset;

    vec2 tangent = vec2(-edge_normal.y, edge_normal.x);
    float crimpPhase = dot(v_uv * 180.0, tangent);
    float borderZone = smoothstep(1.0, 0.5, mask) * smoothstep(0.0, 0.3, mask);
    float creases = sin(crimpPhase + wave * 8.0) * borderZone * 0.04;

    // Calculate pristine, anti-aliased surface normals from the smoothed height map
    float w_left = sampleSmooth(u_simState, v_uv - vec2(u_simTexelSize.x, 0.0), u_simTexelSize).r;
    float w_right = sampleSmooth(u_simState, v_uv + vec2(u_simTexelSize.x, 0.0), u_simTexelSize).r;
    float w_up = sampleSmooth(u_simState, v_uv + vec2(0.0, u_simTexelSize.y), u_simTexelSize).r;
    float w_down = sampleSmooth(u_simState, v_uv - vec2(0.0, u_simTexelSize.y), u_simTexelSize).r;
    
    float dZdx = ((w_right - w_left) * 0.5) + dFdx(creases); 
    float dZdy = ((w_up - w_down) * 0.5) + dFdy(creases);
    
    // Z-weight set for thick, rounded plastic
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 0.18));
    
    // 2. MYLAR PLASTIC STUDIO LIGHTING
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    
    // Key Light (Sharp, intense flash)
    vec3 mainLightDir = normalize(vec3(0.3, 0.7, 0.8));
    vec3 halfMain = normalize(mainLightDir + viewDir);
    float specMain = pow(max(dot(normal, halfMain), 0.0), 300.0) * 2.5; 
    
    // Box Light (Broad, soft studio reflection)
    vec3 boxLightDir = normalize(vec3(-0.4, 0.6, 0.5));
    vec3 halfBox = normalize(boxLightDir + viewDir);
    float specBox = pow(max(dot(normal, halfBox), 0.0), 40.0) * 0.8;

    // Rim Light (Floor/environment bounce for volume)
    vec3 rimLightDir = normalize(vec3(-0.8, -0.2, -0.5));
    vec3 halfRim = normalize(rimLightDir + viewDir);
    float specRim = pow(max(dot(normal, halfRim), 0.0), 60.0) * 0.4;
    
    // Edge Fresnel (The glowing plastic wrap rim)
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 4.0);

    vec4 texColor = texture(u_imageTexture, warpedUV);
    float diffuse = max(dot(normal, mainLightDir), 0.0);
    
    // High ambient keeps the base colors saturated and bright
    vec3 finalColor = texColor.rgb * (diffuse * 0.4 + 0.7);
    
    // Add all specular reflections additively
    finalColor += vec3(specMain + specBox + specRim);
    
    // Add Fresnel glow
    finalColor += (texColor.rgb + vec3(0.8)) * fresnel * 0.8;

    // 3. PERFECT ANTI-ALIASED EDGES
    // Uses the bilinearly smoothed SDF to generate a razor-sharp vector mask
    float aaMask = smoothstep(0.0, 0.02, sdf + (wave * 0.01));

    if (aaMask < 0.99) {
        float shadowIntensity = smoothstep(0.0, 0.15, length(bleedOffset)) * mask;
        vec3 bgColor = vec3(0.1); // Match the dark layout background
        finalColor = mix(bgColor, finalColor, max(aaMask, 1.0 - shadowIntensity));
    }

    fragColor = vec4(finalColor, 1.0);
}
`;
