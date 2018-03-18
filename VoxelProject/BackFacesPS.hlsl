struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 texCoord : TEXCOORD;
	float3 eyeSpacePos : TEXCOORD1;
};

RWTexture2D<float4> backCoordTexture : register(u2);

float4 main(PS_INPUT input) : SV_TARGET
{
	float4 texCoord = input.texCoord;
	texCoord.w = length(input.eyeSpacePos);
	backCoordTexture[input.pos.xy] = texCoord;
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}