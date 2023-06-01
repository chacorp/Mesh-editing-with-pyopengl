#version 330 core
in vec4 worldPosition;
in vec3 OutNormal;
in vec3 OutCoord;
in vec4 nearVert;

out vec4 OutColor;

uniform sampler2D texture0; // texture
uniform sampler2D texture1; // ref image eye left
uniform sampler2D texture2; // ref image eye right
uniform sampler2D texture3; // ref image lips

uniform mat4 transform;

uniform float _alpha;
uniform float show_bg;
uniform float show_m;
uniform float show_ldm;

uniform vec2 mouse;

uniform vec3 m_ldm1;
uniform vec3 m_ldm2;

void main()
{
    vec2 uv = OutCoord.xy;
    // uv.y = 1 - uv.y;

    vec4 tex_col;
    if (OutCoord.z == 1.0f)
        tex_col  = texture(texture0, uv);
    else
        if (show_bg == 1.0f){
            tex_col  = texture(texture1, uv);
        }
        else if (show_bg == 2.0f){
            tex_col  = texture(texture2, uv);
        }
        else if (show_bg == 3.0f){
            tex_col  = texture(texture3, uv);
        }

    float zero = 0.000000001;

    vec3 LPos = vec3(0.0, 1.0, 1.0);
    vec3 Ldir = normalize(LPos - worldPosition.xyz);

    // vec3 Normal = (transpose(transform) * vec4(OutNormal.xyz, 0.0f)).xyz;
    // Normal = normalize(Normal);

    // float diffuse = dot(Normal.xyz, Ldir);


    // vec4 tempColor = tex_col + (diffuse * zero);
    // vec4 tempColor = tex_col * 0.8 + diffuse * 0.2;
    // vec4 tempColor = tex_col * zero + diffuse * zero + vec4(OutNormal, 1.0f);
    vec4 tempColor = tex_col + zero * vec4(OutNormal, 1.0f);
    
    // vec4 Normal = vec4(OutNormal.x, OutNormal.y, OutNormal.z, 1.0f);
    // vec4 tempColor = tex_col * zero + Normal * 0.5f + 0.5f;

    if (OutCoord.z == 1.0f){
        // tempColor.xyz = tempColor.xyz * _alpha;
        if (tex_col.x + tex_col.y + tex_col.z == 0f){
            tempColor.w = 0.0f;
        }
        OutColor = tempColor;
        if (show_m == 1.0f){
            OutColor.w = tempColor.w *_alpha;
        }
        else{
            OutColor.w = 0.0f;
        }
    }
    else{
        if (show_bg != 0.0f){
            OutColor = tempColor;
        }
        else{
            OutColor = tempColor * zero;
        }
    }
    vec2 mdir = worldPosition.xy - mouse.xy;
    float mdis = mdir.x * mdir.x + mdir.y * mdir.y;
    if (sqrt(mdis) < 0.01f){
        OutColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
    }

    // vec2 nvdir = worldPosition.xy - nearVert.xy;
    // float nvdis = nvdir.x * nvdir.x + nvdir.y * nvdir.y;
    // if (sqrt(nvdis) < 0.00001f && nearVert.w == 1.0f){
    //     OutColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    // }

    if (show_ldm == 1.0f){
        vec2 ldm1dir = worldPosition.xy - m_ldm1.xy;
        float ldm1Dis = ldm1dir.x * ldm1dir.x + ldm1dir.y * ldm1dir.y;
        if (sqrt(ldm1Dis) < 0.02f){
            OutColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
        }
        vec2 ldm2dir = worldPosition.xy - m_ldm2.xy;
        float ldm2Dis = ldm2dir.x * ldm2dir.x + ldm2dir.y * ldm2dir.y;
        if (sqrt(ldm2Dis) < 0.02f){
            OutColor = vec4(0.0f, 0.5f, 1.0f, 1.0f);
        }
    }
}