struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 tex : TEXCOORD;
};

float4 main(PS_INPUT input) : SV_TARGET
{
	return float4(input.tex, 1.0f);
	//return float4(1.0f, 1.0f, 1.0f, 1.0f);
}