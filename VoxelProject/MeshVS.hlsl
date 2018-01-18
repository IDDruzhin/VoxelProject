struct VS_INPUT
{
	float4 pos : POSITION;
	float3 tex : TEXCOORD;
};

struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 tex : TEXCOORD;
};

PS_INPUT main(VS_INPUT input)
{
	PS_INPUT output;
	output.pos = input.pos;
	output.tex = input.tex;
	return output;
}