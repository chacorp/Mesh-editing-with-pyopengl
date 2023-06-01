#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

uniform mat4 transform;

// out vec3 vertColor;
out vec4 worldPosition;
out vec3 OutNormal;
out vec3 OutCoord;
// out mat3 TBN;


void main()
{
    vec3 bitangent = cross(normal, tangent);
    // mat3 TBN = mat3(tangent, bitangent, normal);

    vec4 tempPosition = vec4(position.x, position.y, position.z, 1.0f);

    tempPosition = transform * tempPosition;

    worldPosition = tempPosition; // world space
    // worldPosition = transform * tempPosition; // world space

    gl_Position = worldPosition; // world space

    OutNormal = normal;
    OutCoord  = texcoord;
}