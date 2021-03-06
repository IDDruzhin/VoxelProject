struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 texCoord : TEXCOORD;
	float3 eyeSpacePos : TEXCOORD1;
};


cbuffer RenderingConstantBuffer : register(b1)
{
	float4x4 WorldViewProj;
	float4x4 WorldView;
	float stepSize;
	float stepRatio;
};

uint textureIndex : register(b0);
RWTexture2D<float4> renderTexture : register(u1);
RWTexture2D<float4> backCoordTexture : register(u2);
Texture1D<float4> palette : register(t0);
Texture1D<float> segmentsOpacity : register(t1);
Texture3D<uint2> textures[] : register(t2);

float4 main(PS_INPUT input) : SV_TARGET
{
	float curDist = length(input.eyeSpacePos);
	float3 dir = stepRatio * normalize(backCoordTexture[input.pos.xy].xyz - input.texCoord.xyz);
	float startLen = ceil(curDist / stepSize) * stepSize;
	if (startLen > backCoordTexture[input.pos.xy].w)
	{
		discard;
	}
	uint stepsCount = (backCoordTexture[input.pos.xy].w - startLen) / stepSize;
	float3 cur = dir * (startLen - curDist) / stepSize + input.texCoord.xyz;
	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	for (uint s = 0; s <= stepsCount; s++)
	{
		if (color.w < 0.1f)
		{
			break;
		}
		smp = textures[textureIndex][cur];
		color.xyz = color.xyz + color.w * palette[smp.x] * segmentsOpacity[smp.y];
		color.w = color.w * (1.0f - segmentsOpacity[smp.y]);
		cur += dir;
	}
	renderTexture[input.pos.xy] = color;
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}