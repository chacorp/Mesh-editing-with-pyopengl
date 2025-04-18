#version 330 core
layout(location = 0) in vec3 to_coord;
layout(location = 1) in vec3 from_coord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

uniform mat4 scale;
uniform mat4 transform;
uniform mat4 proj;
uniform vec3 trans;

// uniform sampler2D texture1;
uniform vec3 d_range;

// out vec3 vertColor;
out vec4 worldPosition;
out vec3 OutNormal;
out vec3 OutCoord;
// out mat3 TBN;


void main()
{
    vec4 tempPosition = vec4(to_coord.x, to_coord.y, to_coord.z, 1.0f)*2-1;
    vec4 texPosition = vec4(from_coord.x, from_coord.y, from_coord.z, 1.0f)*2-1;

    // worldPosition = texPosition + (0.000000000001f * tempPosition); // uv space (from_mesh)
    worldPosition = (texPosition*0.000000000001f) + tempPosition; // uv space (to_mesh)

    gl_Position = worldPosition; // world space

    OutNormal = normal;
    OutCoord  = from_coord;
    // OutCoord  = to_coord;

}