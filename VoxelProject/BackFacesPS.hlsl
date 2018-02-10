struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 texCoord : TEXCOORD;
	float3 eyeSpacePos : TEXCOORD1;
};

RWTexture2D<float4> backCoordTexture : register(u2);

//float4 main(PS_INPUT input) : SV_TARGET
float4 main(PS_INPUT input) : SV_TARGET
{
	//backCoordTexture[input.pos.xy] = float4(input.texCoord, input.pos.z);
	float4 texCoord = input.texCoord;
	texCoord.w = length(input.eyeSpacePos);
	//float4 eyeSpacePos = input.texCoord;
	//eyeSpacePos.w = length(eyeSpacePos.xyz) - eyeSpacePos.w;
	backCoordTexture[input.pos.xy] = texCoord;
	//backCoordTexture[input.pos.xy] = input.texCoord;
	//backCoordTexture[input.pos.xy] = float4(input.texCoord, depth);
	//return input.texCoord;
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}