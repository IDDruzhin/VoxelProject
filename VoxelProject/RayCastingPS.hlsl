struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 texCoord : TEXCOORD;
};

cbuffer RenderingConstantBuffer : register(b0)
{
	float4x4 WorldViewProj;
	float stepSize;
	float stepRatio;
};

uint textureIndex : register(b1);
SamplerState s : register(s0);
RWTexture2D<float4> renderTexture : register(u1);
RWTexture2D<float4> backCoordTexture : register(u2);
Texture1D<float4> palette : register(t0);
Texture1D<float> segmentsOpacity : register(t1);
//Texture3D<uint2> textures[] : register(t2);
//Texture3D<float2> textures[] : register(t2);
//Texture3D<float2> textures[] : register(t1);
Texture3D<uint2> textures[] : register(t2);

float4 main(PS_INPUT input) : SV_TARGET
{
	float3 dir = input.texCoord - backCoordTexture[input.pos.xy].xyz;
	float lenView = input.pos.z - backCoordTexture[input.pos.xy].w;
	float lenTex = length(dir);
	dir = stepRatio * lenTex / lenTex;
	float startLen = ceil(input.pos.z / stepSize) * stepSize;
	float3 cur = (startLen - input.pos.z) * lenTex / lenView + input.texCoord;
	uint stepsCount = (backCoordTexture[input.pos.xy].w - startLen) / stepSize;
	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	for (uint i = 0; i < stepsCount; i++)
	{
		if (color.w < 0.1f)
		{
			break;
		}
		smp = textures[textureIndex][cur];
		color.xyz = color.xyz + color.w * palette[smp.x] * segmentsOpacity[smp.y];
		color.w = color.w * (1.0f - segmentsOpacity[smp.y]);
		cur += dir;
	}
	renderTexture[input.pos.xy] = color;
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}