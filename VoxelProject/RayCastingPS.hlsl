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
//uint textureIndex : register(b0, space1);
SamplerState s : register(s0);
RWTexture2D<float4> renderTexture : register(u1);
RWTexture2D<float4> backCoordTexture : register(u2);
Texture1D<float4> palette : register(t0);
Texture1D<float> segmentsOpacity : register(t1);
//Texture3D<uint2> textures[] : register(t2);
//Texture3D<float2> textures[] : register(t2);
//Texture3D<float2> textures[] : register(t1);
Texture3D<uint2> textures[] : register(t2);
//Texture3D<uint2> textures[] : register(t1);

float4 main(PS_INPUT input) : SV_TARGET
{
	/*
	float3 dir = backCoordTexture[input.pos.xy].xyz - input.texCoord;
	float lenView = backCoordTexture[input.pos.xy].w - input.pos.z;
	float lenTex = length(dir);
	dir = stepRatio * dir / lenTex;
	float startLen = ceil(input.pos.z / stepSize) * stepSize;
	//float3 cur = (startLen - input.pos.z) * lenTex / lenView + input.texCoord;
	float3 cur = input.texCoord;
	uint stepsCount = (backCoordTexture[input.pos.xy].w - startLen) / stepSize;
	//uint stepsCount = 300;
	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	for (uint i = 0; i < stepsCount; i++)
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
	*/
	/*
	float3 dir = backCoordTexture[input.pos.xy].xyz - input.texCoord;
	float lenView = backCoordTexture[input.pos.xy].w - input.pos.z * input.pos.w;
	float lenTex = length(dir);
	dir = stepRatio * dir / lenTex;
	float startLen = ceil(input.pos.z * input.pos.w / stepSize) * stepSize;
	float3 cur = (startLen - input.pos.z * input.pos.w) * lenTex / lenView + input.texCoord;
	//float3 cur = input.texCoord;
	uint stepsCount = (backCoordTexture[input.pos.xy].w - startLen) / stepSize;
	//uint stepsCount = 300;
	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	for (uint i = 0; i < stepsCount; i++)
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
	*/
	
	float4 test;

	float curDist = length(input.eyeSpacePos);
	float3 dir = stepRatio * normalize(backCoordTexture[input.pos.xy].xyz - input.texCoord.xyz);
	//float lenView = backCoordTexture[input.pos.xy].w - curDist;
	//float lenTex = length(dir);
	float startLen = ceil(curDist / stepSize) * stepSize;
	//float3 cur = (startLen - input.texCoord.w) * lenTex / lenView + input.texCoord.xyz;
	//float3 cur = input.texCoord.xyz;
	//float3 cur = input.texCoord.xyz;
	//float3 cur = dir * (startLen - curDist) / lenView + input.texCoord.xyz;
	uint stepsCount = (backCoordTexture[input.pos.xy].w - startLen) / stepSize;
	//dir = stepRatio * dir / lenTex;
	float3 cur = dir * (startLen - curDist) / stepSize + input.texCoord.xyz;
	/*
	test.x = stepSize / lenView;
	test.y = stepRatio / lenTex;
	test.z = lenView / stepSize;
	//stepsCount = length(backCoordTexture[input.pos.xy].xyz - cur) / stepRatio;
	test.w = lenTex / stepRatio;
	backCoordTexture[input.pos.xy] = test;
	*/
	//uint stepsCount = lenTex / stepRatio;
	//uint stepsCount = lenView / stepSize;
	//uint stepsCount = length(backCoordTexture[input.pos.xy].xyz - cur) / stepRatio;
	//dir = stepRatio * dir / lenTex;
	//uint stepsCount = 2000;
	//cur = input.texCoord.xyz;
	//stepsCount = length(backCoordTexture[input.pos.xy].xyz - cur) / stepRatio;

	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	[loop] for (uint i = 0; i <= stepsCount; i++)
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
	
	/*
	float3 dir = backCoordTexture[input.pos.xy].xyz - input.texCoord.xyz;
	float lenView = backCoordTexture[input.pos.xy].w - input.texCoord.w;
	float lenTex = length(dir);
	float startLen = ceil(input.texCoord.w / stepSize) * stepSize;
	//float3 cur = (startLen - input.texCoord.w) * lenTex / lenView + input.texCoord.xyz;
	//float3 cur = input.texCoord.xyz;
	float3 cur = input.texCoord.xyz;
	uint stepsCount = length(dir) / stepRatio;
	dir = stepRatio * dir / lenTex;
	//uint stepsCount = 2000;
	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	for (uint i = 0; i < stepsCount; i++)
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
	*/
	
	/*
	float3 dir = normalize(backCoordTexture[input.pos.xy].xyz - input.texCoord);
	float lenTex = length(dir);
	float3 cur = input.texCoord;
	float4 color = renderTexture[input.pos.xy];
	uint2 smp;
	[loop] for (uint i = 0; i < 400; i++)
	{
		smp = textures[textureIndex][cur];
		if (smp.y != 0)
		{
			color = float4(1, 0, 0, 0);
		}
		cur += dir;
	}
	renderTexture[input.pos.xy] = color;
	*/
	
	/*
	if (textureIndex != 0)
	{
		renderTexture[input.pos.xy] = float4(textureIndex/100.0f,0,0,0);
	}
	*/
	//return input.texCoord;
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}