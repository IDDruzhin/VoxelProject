cbuffer RenderingConstantBuffer : register(b0)
{
	float4x4 WorldViewProj;
};

SamplerState s : register(s0);
RWTexture2D<float4> positionTexture : register(u1);
RWTexture2D<float4> renderTexture : register(u2);
Texture1D<float4> palette : register(t0);
Texture1D<float4> segmentsOpacity : register(t1);
Texture3D<int2> textures[] : register(t3);

float4 main() : SV_TARGET
{
	return float4(1.0f, 1.0f, 1.0f, 1.0f);
}