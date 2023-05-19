#version 330 core
layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 vert_color;
layout(location = 1) in vec3 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

uniform mat4 transform;
uniform float timer_y;
// uniform float timer_x;
uniform sampler2D texture3;

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
    // gl_Position = vec4((texcoord.xy*2.0 - 1.0), 0.0, 1.0f); // UV space
    
    // vertColor = vec3(gl_Position.z / gl_Position.w, 0.0, 0.0);
    //OutNormal = (transform * vec4(normal, 1.0)).xyz;
    // OutNormal = vec3(texcoord.xy, 0.0) + normal * 0.000000001;

    OutNormal = normal;
    // OutNormal = (transform * vec4(normal.x, normal.y, -normal.z, 1.0f)).xyz;
    OutCoord  = texcoord;
}