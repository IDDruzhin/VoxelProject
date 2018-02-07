struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 texCoord : TEXCOORD;
};

RWTexture2D<float4> backCoordTexture : register(u2);

float4 main(PS_INPUT input) : SV_TARGET
{
	backCoordTexture[input.pos.xy] = float4(input.texCoord, 1.0f);
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}