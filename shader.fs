#version 330 core
in vec4 worldPosition;
in vec3 OutNormal;
in vec3 OutCoord;

out vec4 OutColor;

uniform sampler2D texture0;
uniform sampler2D texture1;

// uniform mat4 scale;
uniform mat4 transform;
uniform mat4 proj;

void main()
{
    vec2 uv = OutCoord.xy;
    // uv.y = 1 - uv.y;

    vec3 camPos = vec3(0.0, 0.0, 1.0);
    vec3 camDir = camPos - worldPosition.xyz;
    camDir      = normalize(camDir);

    vec3 Normal = OutNormal;
    // Normal = (transpose(inverse(transform)) * vec4(Normal.xyz, 0.0f)).xyz;
    // Normal = normalize(Normal);

    vec3 LPos = vec3(0.0, 1.0, 0.0);
    vec3 Ldir = normalize(LPos - worldPosition.xyz);

    float diffuse = dot(Normal.xyz, Ldir);

    // float diffuse = dot(OutNormal.xyz, Ldir);

    vec4 tex_col  = texture(texture0, uv);
    float zero = 0.000000001;

    // vec4 tempColor = vec4(OutNormal, 0.0f) * zero + tex_col *0.5f  + diffuse*0.5f;
    vec4 tempColor = tex_col *0.5f  + diffuse*0.5f;
    // vec4 tempColor = vec4(OutNormal, 0.0f) * zero + tex_col  + diffuse*zero;
    
    OutColor = tempColor;
    OutColor.w = 1.0f;
}