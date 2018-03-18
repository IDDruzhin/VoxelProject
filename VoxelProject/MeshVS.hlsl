cbuffer RenderingConstantBuffer : register(b1)
{
	float4x4 WorldViewProj;
	float4x4 WorldView;
	float voxelSize;
	float stepRatio;
};

struct VS_INPUT
{
	float4 pos : POSITION;
	float4 texCoord : TEXCOORD;
};

struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 texCoord : TEXCOORD;
	float3 eyeSpacePos : TEXCOORD1;
};

PS_INPUT main(VS_INPUT input)
{
	PS_INPUT output;
	float3 eyeSpacePos = mul(input.pos, WorldView);
	output.pos = mul(input.pos, WorldViewProj);
	output.texCoord = input.texCoord;
	output.eyeSpacePos = eyeSpacePos;
	return output;
}