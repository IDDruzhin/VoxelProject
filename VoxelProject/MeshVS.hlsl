cbuffer RenderingConstantBuffer : register(b0)
{
	float4x4 WorldViewProj;
	float voxelSize;
	float stepRatio;
};

struct VS_INPUT
{
	float4 pos : POSITION;
	float3 texCoord : TEXCOORD;
};

struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 texCoord : TEXCOORD;
};

PS_INPUT main(VS_INPUT input)
{
	PS_INPUT output;
	output.pos = mul(input.pos, WorldViewProj);
	output.texCoord = input.texCoord;
	return output;
}