#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

uniform mat4 transform;
uniform mat4 proj;

uniform vec2 mouse;

uniform vec3 trans;
uniform vec3 m_ldm1;
uniform vec3 m_ldm2;

uniform float show_m;

out vec4 worldPosition;
out vec3 OutNormal;
out vec3 OutCoord;
out vec4 nearVert;
// out mat3 TBN;


// mat4 Ortho(float left, float right, float bottom, float top, float near, float far)
// {
//     mat4 T = Translate(-(left+right)/2, -(top+bottom)/2, near/2);
//     mat4 S = Scale(2/(left-right), 2/(top-bottom), 1/(far-near));
//     mat4 V = S*T;
//     return V;
// }

void main()
{
    vec3 bitangent = cross(normal, tangent);
    // mat3 TBN = mat3(tangent, bitangent, normal);
    vec4 tempPosition;
    if (texcoord.z == 1.0f){
        tempPosition = vec4(position.x + trans.x, position.y + trans.y, position.z + trans.z, 1.0f);
        tempPosition = proj * transform * tempPosition;
        
    }
    else{
        tempPosition = vec4(position.x, position.y, position.z + trans.z, 1.0f);
        tempPosition = proj * tempPosition;
    }

    worldPosition = tempPosition; // world space
    // worldPosition = transform * tempPosition; // world space
    gl_Position = worldPosition * 2.0f; // world space


    vec2 mdir = worldPosition.xy - mouse.xy;
    float mdis = mdir.x * mdir.x + mdir.y * mdir.y;
    mdis = sqrt(mdis);
    if (mdis < 0.12f){
        nearVert = vec4(worldPosition.xyz, 1.0f);
    }else{
        nearVert = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    OutNormal = normal;
    OutCoord  = texcoord;
}