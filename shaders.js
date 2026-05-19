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

void main() {
    vec4 state = texture(u_simState, v_uv);
    float wave = state.r;
    float mask = state.b;
    
    float m_left  = texture(u_simState, v_uv + vec2(-u_simTexelSize.x, 0.0)).b;
    float m_right = texture(u_simState, v_uv + vec2( u_simTexelSize.x, 0.0)).b;
    float m_up    = texture(u_simState, v_uv + vec2(0.0,  u_simTexelSize.y)).b;
    float m_down  = texture(u_simState, v_uv + vec2(0.0, -u_simTexelSize.y)).b;
    
    vec2 edge_normal = normalize(vec2(m_left - m_right, m_down - m_up) + 0.0001);

    vec2 bleedOffset = edge_normal * clamp(wave * 2.0, -1.0, 1.0) * u_simTexelSize;
    vec2 warpedUV = v_uv - bleedOffset;

    vec2 tangent = vec2(-edge_normal.y, edge_normal.x);
    float crimpPhase = dot(v_uv * 180.0, tangent);
    float borderZone = smoothstep(1.0, 0.5, mask) * smoothstep(0.0, 0.3, mask);
    float creases = sin(crimpPhase + wave * 8.0) * borderZone * 0.04;

    float w_left = texture(u_simState, v_uv + vec2(-u_simTexelSize.x, 0.0)).r;
    float w_right = texture(u_simState, v_uv + vec2(u_simTexelSize.x, 0.0)).r;
    float w_up = texture(u_simState, v_uv + vec2(0.0, u_simTexelSize.y)).r;
    float w_down = texture(u_simState, v_uv + vec2(0.0, -u_simTexelSize.y)).r;
    
    float dZdx = ((w_right - w_left) * 0.5) + dFdx(creases); 
    float dZdy = ((w_up - w_down) * 0.5) + dFdy(creases);
    
    // CRITICAL VISUAL FIX: Z-weight increased to 0.25 makes the surface look fat and smoothly rounded
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 0.25));
    
    // DUAL STUDIO LIGHTING (Mylar balloon look)
    vec3 mainLightDir = normalize(vec3(0.4, 0.8, 1.0));
    vec3 fillLightDir = normalize(vec3(-0.6, -0.4, 0.5));
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    
    vec3 halfMain = normalize(mainLightDir + viewDir);
    vec3 halfFill = normalize(fillLightDir + viewDir);
    
    float diffuse = max(dot(normal, mainLightDir), 0.0);
    float bounce = max(dot(normal, fillLightDir), 0.0) * 0.3; // Soft ambient fill
    
    float specMain = pow(max(dot(normal, halfMain), 0.0), 150.0) * 1.5; // Sharp key highlight
    float specFill = pow(max(dot(normal, halfFill), 0.0), 50.0) * 0.5;  // Broad fill highlight
    
    // EDGE FRESNEL (Plastic Sheen)
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

    vec4 texColor = texture(u_imageTexture, warpedUV);
    
    // Composite: Color + Lighting + Highlights + Sheen
    vec3 finalColor = texColor.rgb * (diffuse * 0.6 + bounce + 0.5);
    finalColor += vec3(specMain + specFill);
    finalColor += (texColor.rgb + 0.5) * fresnel * 0.6; 

    float softMask = texture(u_simState, warpedUV).b;
    if (softMask < 0.99) {
        float shadowIntensity = smoothstep(0.0, 0.1, length(bleedOffset)) * mask;
        finalColor = mix(vec3(0.1), finalColor, 1.0 - shadowIntensity);
    }

    fragColor = vec4(finalColor, 1.0);
}
`;
