struct PS_INPUT
{
	float4 pos: SV_POSITION;
};

cbuffer BonesConstantBuffer : register(b1)
{
	float4x4 Bones[256];
};

PS_INPUT main(float4 pos : POSITION, uint id : SV_InstanceID)
{
	PS_INPUT Out;
	Out.pos = mul(pos, Bones[id]);
	return Out;
}