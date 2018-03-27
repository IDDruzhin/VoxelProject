uint selected : register(b0);

float4 main(uint id : SV_InstanceID) : SV_TARGET
{
	/*
	if (id != selected)
	{
		return float4(0.5f, 0.5f, 0.5f, 1.0f);
	}
	*/
	return float4(1.0f, 0.0f, 0.0f, 1.0f);
}