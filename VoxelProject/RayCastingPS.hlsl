struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float3 texCoord : TEXCOORD;
};

cbuffer RenderingConstantBuffer : register(b0)
{
	float4x4 WorldViewProj;
};

uint textureIndex : register(b1);
SamplerState s : register(s0);
RWTexture2D<float4> renderTexture : register(u1);
RWTexture2D<float4> backCoordTexture : register(u2);
Texture1D<float4> palette : register(t0);
Texture1D<float> segmentsOpacity : register(t1);
//Texture3D<uint2> textures[] : register(t2);
//Texture3D<float2> textures[] : register(t2);
//Texture3D<float2> textures[] : register(t1);
Texture3D<uint2> textures[] : register(t2);

float4 main(PS_INPUT input) : SV_TARGET
{
	//uint2 index = textures[textureIndex].Sample(s,input.texCoord);
	//float index = textures[textureIndex][uint3(0,0,0)];
	//float index = textures[textureIndex].Load(uint4(0,0,0,0));

	//uint2 index = textures[textureIndex].Sample(s,input.texCoord);

	//uint2 index = textures[textureIndex].Sample(s,input.texCoord);
	//uint2 index = textures[textureIndex].SampleLevel(s,input.texCoord,0);

	/*
	if (index.x != 0)
	{
		//renderTexture[input.pos.xy] = palette[index.x];
		renderTexture[input.pos.xy] = float4(1.0f, 0.0f, 0.0f, 1.0f);
		//return float4(1.0f, 0.0f, 0.0f, 0.0f);
	}
	*/
	uint width;
	uint height;
	uint depth;
	textures[textureIndex].GetDimensions(width, height, depth);
	uint2 smp = textures[textureIndex][float3(input.texCoord.x*width, input.texCoord.y*height, input.texCoord.z*depth)];
	//uint2 smp = textures[textureIndex][float3(input.texCoord.x*(width - 1), input.texCoord.y*(height - 1), input.texCoord.z*(depth - 1))];
	//uint2 smp = textures[textureIndex].Load(uint4(input.texCoord.x*(width - 1), input.texCoord.y*(height-1), input.texCoord.z*(depth-1), 0));
	//uint2 smp = textures[textureIndex].Load(uint4(ceil(input.texCoord.x*width)-1, input.texCoord.y*height, input.texCoord.z*depth, 0));
	//float2 smp = textures[textureIndex].Sample(s,input.texCoord);
	//if ((renderTexture[input.pos.xy].x == 0) && (renderTexture[input.pos.xy].y == 0))
	if (smp.y!=0)
	{
		//renderTexture[input.pos.xy] = float4(smp.x, smp.y, 0.0f, 0.0f);
		renderTexture[input.pos.xy] = float4(1.0f, 0.0f, 0.0f, 0.0f);
	}
	
	return float4(smp.x, smp.y, 0.0f, 0.0f);
	

	//renderTexture[input.pos.xy] = float4(textureIndex/256.0f,0,0,1);
	//uint2 index = textures[textureIndex].SampleLevel(s,input.texCoord,0);
	//uint2 index = textures[textureIndex].Load(input.texCoord);
	//renderTexture[input.pos.xy] = palette[index.x];
	//return textures[textureIndex].Sample(s,input.texCoord);
	//renderTexture[input.pos.xy] = float4(input.texCoord, 1.0f) - backCoordTexture[input.pos.xy];
	//renderTexture[input.pos.xy] = backCoordTexture[input.pos.xy];
	discard;
	//return palette[index.x];
	return float4(0.0f,0.0f,0.0f,0.0f);
	//return float4(1.0f, 1.0f, 1.0f, 1.0f);
}