struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 tex : TEXCOORD;
};

RWTexture2D<float4> renderTexture : register(u0);

float4 main(PS_INPUT input) : SV_TARGET
{
	//renderTexture[input.pos.xy] = float4(input.tex, 1.0f);
	//return renderTexture[input.pos.xy];
	return float4(input.tex, 1.0f);
	//return float4(1.0f, 1.0f, 1.0f, 1.0f);
	//return float4(input.tex.x, 0.0f, 0.0f, 1.0f);
}