#version 330 core
in vec4 worldPosition;
// in vec3 vertColor;
in vec3 OutNormal;
in vec3 OutCoord;

out vec4 OutColor;

uniform sampler2D texture1;
uniform mat4 transform;

void main()
{
    vec2 uv = OutCoord.xy;
    // uv.y = 1 - uv.y;

    vec4 tex_col  = texture(texture1, uv);
    float zero = 0.000000001;

    vec4 tempColor = vec4(OutNormal, 0.0f) * zero + worldPosition * zero + tex_col;
    
    OutColor = tempColor;
    OutColor.w = 1.0f;
}