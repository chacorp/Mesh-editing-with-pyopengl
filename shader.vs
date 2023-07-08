#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

uniform mat4 scale;
uniform mat4 transform;
uniform mat4 proj;
uniform vec3 trans;

uniform sampler2D texture1;
uniform vec3 d_range;

// out vec3 vertColor;
out vec4 worldPosition;
out vec3 OutNormal;
out vec3 OutCoord;
// out mat3 TBN;


void main()
{
    // vec3 bitangent = cross(normal, tangent);
    // mat3 TBN = mat3(tangent, bitangent, normal);

    vec4 tempPosition = vec4(position.x, position.y, position.z, 1.0f);
    
    vec4 disp  = texture(texture1, texcoord.xy);
    disp = disp * (d_range.y - d_range.x) + vec4(d_range.x);
    // disp = vec4(disp.x, -disp.y, -disp.z, 0.0f);
    // disp = vec4(disp.x, disp.y, disp.z, 0.0f);

    tempPosition = tempPosition - vec4(disp.xyz, 0.0f);

    tempPosition = transform * tempPosition + vec4(trans, 0.0f);
    tempPosition = proj * tempPosition;

    worldPosition = tempPosition; // world space
    // worldPosition = transform * tempPosition; // world space

    gl_Position = worldPosition; // world space

    OutNormal = normal;
    OutCoord  = texcoord;
}