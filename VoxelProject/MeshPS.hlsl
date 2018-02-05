struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 tex : TEXCOORD;
};

RWTexture2D<float4> positionTexture : register(u1);

float4 main(PS_INPUT input) : SV_TARGET
{
	renderTexture[input.pos.xy] = float4(input.tex, 1.0f);
	//discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
	//return renderTexture[input.pos.xy];
	//float O;
	//InterlockedExchange(renderTexture[input.pos.xy], 1.0f,O);
	//renderTexture[input.pos.xy] = input.tex.x;
	//int Uo;
	//InterlockedExchange(renderTexture[input.pos.xy].x, 255,Uo);
	//return renderTexture[input.pos.xy];
	//renderTexture[input.pos.xy] = 1.0f;
	//InterlockedExchange(renderTexture[input.pos.xy].x, 1.0f);
	
	//return float4(input.tex, 1.0f);
	//return float4(1.0f, 1.0f, 1.0f, 1.0f);
	//return float4(input.tex.x, 0.0f, 0.0f, 1.0f);
}