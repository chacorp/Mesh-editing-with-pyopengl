#version 330 core
in vec4 worldPosition;
in vec3 OutNormal;
in vec3 OutCoord;

out vec4 OutColor;

uniform sampler2D texture0;

// uniform mat4 scale;
// uniform mat4 transform;
// uniform mat4 proj;

void main()
{
    vec2 uv = OutCoord.xy;

    vec3 camPos = vec3(0.0, 0.0, 1.0);
    vec3 camDir = camPos - worldPosition.xyz;
    camDir      = normalize(camDir);

    vec3 Normal = OutNormal;

    vec3 LPos = vec3(0.0, 1.0, 0.0);
    vec3 Ldir = normalize(LPos - worldPosition.xyz);

    float diffuse = dot(Normal.xyz, Ldir);

    vec4 tex_col  = texture(texture0, uv);
    float zero = 0.000000001;

    vec4 tempColor = tex_col + diffuse*0.000000000001f;
    
    OutColor = tempColor;
    OutColor.w = 1.0f;
}