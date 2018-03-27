struct PS_INPUT
{
	float4 pos: SV_POSITION;
	float4 col: COLOR;
};

cbuffer BonesConstantBuffer : register(b1)
{
	float4x4 Bones[256];
};

float4 color : register(b0);
uint selected : register(b2);

PS_INPUT main(float4 pos : POSITION, uint id : SV_InstanceID)
{
	PS_INPUT Out;
	Out.pos = mul(pos, Bones[id]);
	if (id == selected)
	{
		Out.col = float4(1.0f, 0.0f, 0.0f, 1.0f);
	}
	else
	{
		Out.col = color;
	}
	return Out;
}