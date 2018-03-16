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

//SamplerState s : register(s0);

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
	float3 coefs;
	float4 curColor;
	float4 tmpColor;
	float4 smpColor;
	int3 neighborsOffsets;
	int3 curIndexes;
	//uint3 curNeighbor;
	[loop] for (uint s = 0; s <= stepsCount; s++)
	{
		if (color.w < 0.1f)
		{
			break;
		}

		//Linear interpolation
		curIndexes = cur;
		coefs = cur - curIndexes - float3(0.5f, 0.5f, 0.5f);
		neighborsOffsets = sign(coefs);
		coefs = abs(coefs);

		smp = textures[textureIndex][curIndexes];
		smpColor.xyz = palette[smp.x].xyz;
		smpColor.w = segmentsOpacity[smp.y];
		curColor = smpColor * (1.0f - coefs.x) * (1.0f - coefs.y) * (1.0f - coefs.z);

		smp = textures[textureIndex][uint3(curIndexes.x, curIndexes.y, curIndexes.z + neighborsOffsets.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}	
		curColor += tmpColor * (1.0f - coefs.x) * (1.0f - coefs.y) * coefs.z;

		smp = textures[textureIndex][uint3(curIndexes.x, curIndexes.y + neighborsOffsets.y, curIndexes.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}
		curColor += tmpColor * (1.0f - coefs.x) * coefs.y * (1.0f - coefs.z);

		smp = textures[textureIndex][uint3(curIndexes.x, curIndexes.y + neighborsOffsets.y, curIndexes.z + neighborsOffsets.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}
		curColor += tmpColor * (1.0f - coefs.x) * coefs.y * coefs.z;

		smp = textures[textureIndex][uint3(curIndexes.x + neighborsOffsets.x, curIndexes.y, curIndexes.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}
		curColor += tmpColor * coefs.x * (1.0f - coefs.y) * (1.0f - coefs.z);

		smp = textures[textureIndex][uint3(curIndexes.x + neighborsOffsets.x, curIndexes.y, curIndexes.z + neighborsOffsets.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}
		curColor += tmpColor * coefs.x * (1.0f - coefs.y) * coefs.z;

		smp = textures[textureIndex][uint3(curIndexes.x + neighborsOffsets.x, curIndexes.y + neighborsOffsets.y, curIndexes.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}
		curColor += tmpColor * coefs.x * coefs.y * (1.0f - coefs.z);

		smp = textures[textureIndex][uint3(curIndexes.x + neighborsOffsets.x, curIndexes.y + neighborsOffsets.y, curIndexes.z + neighborsOffsets.z)];
		tmpColor.w = segmentsOpacity[smp.y];
		if (tmpColor.w == 0)
		{
			tmpColor = smpColor;
		}
		else
		{
			tmpColor.xyz = palette[smp.x].xyz;
		}
		curColor += tmpColor * coefs.x * coefs.y * coefs.z;

		color.xyz = color.xyz + color.w * curColor.xyz * curColor.w;
		color.w = color.w * (1.0f - curColor.w);
		cur += dir;
	}
	renderTexture[input.pos.xy] = color;
	discard;
	return float4(0.0f,0.0f,0.0f,0.0f);
}