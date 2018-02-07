struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 texCoord : TEXCOORD;
};

cbuffer RenderingConstantBuffer : register(b0)
{
	float4x4 WorldViewProj;
};

cbuffer DirectConstants : register(b1)
{
	uint textureIndex;
}

SamplerState s : register(s0);
RWTexture2D<float4> renderTexture : register(u1);
RWTexture2D<float4> backCoordTexture : register(u2);
Texture1D<float4> palette : register(t0);
Texture1D<float4> segmentsOpacity : register(t1);
Texture3D<uint2> textures[] : register(t3);

float4 main(PS_INPUT input) : SV_TARGET
{
	//uint2 index = textures[textureIndex].Sample(s,input.texCoord);
	//return textures[textureIndex].Sample(s,input.texCoord);
	//renderTexture[input.pos.xy] = float4(input.texCoord, 1.0f) - backCoordTexture[input.pos.xy];
	renderTexture[input.pos.xy] = backCoordTexture[input.pos.xy];
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
	//return float4(1.0f, 1.0f, 1.0f, 1.0f);
}