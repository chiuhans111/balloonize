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
uniform sampler2D u_simState; // Current mask status
uniform vec2 u_texelSize;
uniform float u_gradientThreshold;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 currentState = texture(u_simState, v_uv);
    float currentMask = currentState.b;
    
    // Quick escape if already marked as background
    if (currentMask == 0.0) {
        fragColor = currentState;
        return;
    }

    // Compute simple local luminance gradient via Sobel-style differences
    float l_left  = dot(texture(u_imageTexture, v_uv + vec2(-u_texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_right = dot(texture(u_imageTexture, v_uv + vec2( u_texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l_up    = dot(texture(u_imageTexture, v_uv + vec2(0.0,  u_texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float l_down  = dot(texture(u_imageTexture, v_uv + vec2(0.0, -u_texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    
    float grad = length(vec2(l_left - l_right, l_down - l_up));

    // Sample neighbors' mask states
    float m_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).b;
    float m_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).b;
    float m_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).b;
    float m_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).b;

    // If an edge neighbor is background and our local gradient is weak, erode inward
    float edgeProximity = 4.0 - (m_left + m_right + m_up + m_down);
    if (edgeProximity > 0.0 && grad < u_gradientThreshold) {
        currentMask = 0.0; // Trim this pixel out
    }

    fragColor = vec4(currentState.r, currentState.g, currentMask, currentState.a);
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

    // 4-way discrete Laplacian
    float u_left  = texture(u_simState, v_uv + vec2(-u_texelSize.x, 0.0)).r;
    float u_right = texture(u_simState, v_uv + vec2( u_texelSize.x, 0.0)).r;
    float u_up    = texture(u_simState, v_uv + vec2(0.0,  u_texelSize.y)).r;
    float u_down  = texture(u_simState, v_uv + vec2(0.0, -u_texelSize.y)).r;
    
    float laplacian = (u_left + u_right + u_up + u_down) - 4.0 * u_t;

    // Wave equation constraints
    float waveSpeed = 0.4 * mask; // Absolute wall boundary conditions outside mask
    float damping = 0.98;
    
    float u_t_plus = 2.0 * u_t - u_t_minus + (waveSpeed * waveSpeed) * laplacian;
    u_t_plus *= damping;

    // Inject pointer force interaction
    float distToPointer = length(v_uv - u_pointerPos);
    if (distToPointer < 0.04 && u_pointerForce > 0.0) {
        u_t_plus += u_pointerForce * (1.0 - distToPointer / 0.04);
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
    
    // Calculate precise geometric normal of the boundary edge
    float m_left  = texture(u_simState, v_uv + vec2(-u_simTexelSize.x, 0.0)).b;
    float m_right = texture(u_simState, v_uv + vec2( u_simTexelSize.x, 0.0)).b;
    float m_up    = texture(u_simState, v_uv + vec2(0.0,  u_simTexelSize.y)).b;
    float m_down  = texture(u_simState, v_uv + vec2(0.0, -u_simTexelSize.y)).b;
    
    vec2 edge_normal = normalize(vec2(m_left - m_right, m_down - m_up) + 0.0001);

    // 1. Boundary Bleed Shift
    vec2 bleedOffset = edge_normal * clamp(wave * 2.0, -1.0, 1.0) * u_simTexelSize;
    vec2 warpedUV = v_uv - bleedOffset;

    // 2. Analytical Perpendicular Crimp Wrinkles
    vec2 tangent = vec2(-edge_normal.y, edge_normal.x);
    float crimpPhase = dot(v_uv * 180.0, tangent);
    float borderZone = smoothstep(1.0, 0.5, mask) * smoothstep(0.0, 0.3, mask);
    float creases = sin(crimpPhase + wave * 8.0) * borderZone * 0.04;

    // 3. Central Difference Shading (Replaces broken dFdx)
    // We sample the wave+crease height of neighboring pixels directly
    float h_center = wave + creases;
    
    float w_left = texture(u_simState, v_uv + vec2(-u_simTexelSize.x, 0.0)).r;
    float w_right = texture(u_simState, v_uv + vec2(u_simTexelSize.x, 0.0)).r;
    float w_up = texture(u_simState, v_uv + vec2(0.0, u_simTexelSize.y)).r;
    float w_down = texture(u_simState, v_uv + vec2(0.0, -u_simTexelSize.y)).r;
    
    // Add crease approximation to neighbors for sharp lighting
    float dZdx = ((w_right - w_left) * 0.5) + dFdx(creases); 
    float dZdy = ((w_up - w_down) * 0.5) + dFdy(creases);
    
    // The Z-weight determines the "thickness" of the fluid. Lower = thicker/glassier.
    vec3 normal = normalize(vec3(-dZdx, -dZdy, 0.015));
    
    // 4. Lighting & Specular Calculation
    vec3 lightDir = normalize(vec3(0.3, 0.7, 0.8));
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 halfVector = normalize(lightDir + viewDir);
    
    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(normal, halfVector), 0.0), 120.0);

    // Fetch original image pixel using the warped/refracted UV
    vec4 texColor = texture(u_imageTexture, warpedUV);
    
    // High-contrast rim/ambient mixing
    vec3 finalColor = texColor.rgb * (diffuse * 0.5 + 0.6) + vec3(specular * 1.5);

    // 5. Exterior Drop Shadow handling
    float softMask = texture(u_simState, warpedUV).b;
    if (softMask < 0.99) {
        // Create a fake ambient occlusion shadow just outside the mask
        float shadowIntensity = smoothstep(0.0, 0.1, length(bleedOffset)) * mask;
        finalColor = mix(vec3(0.1), finalColor, 1.0 - shadowIntensity);
    }

    fragColor = vec4(finalColor, 1.0);
}
`;
