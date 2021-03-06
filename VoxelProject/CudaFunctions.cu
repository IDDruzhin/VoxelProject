#include "stdafx.h"
#include "CudaFunctions.cuh"


__global__ void GetVoxelsAnatomicalSegmentationKernel(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, unsigned char* segmentationTransferTable, int eps, RGBVoxel* voxels, int width, int height, int curDepth, int curNumber, int* count)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
	int curPos = (y*width + x) * 3;
	int4 pixel;
	pixel.z = segmentedImage[curPos];
	pixel.y = segmentedImage[curPos + 1];
	pixel.x = segmentedImage[curPos + 2];
	if ((pixel.x == segmentationTable[0].color.x) && (pixel.y == segmentationTable[0].color.y) && (pixel.z == segmentationTable[0].color.z))
	{
		return;
	}
	for (int i = 1; i < segmentsCount; i++)
	{
		if ((curDepth >= segmentationTable[i].start) && (curDepth <= segmentationTable[i].finish))
		{
			if (abs(pixel.x - segmentationTable[i].color.x)<eps && abs(pixel.y - segmentationTable[i].color.y)<eps && abs(pixel.z - segmentationTable[i].color.z)<eps)
			{
				if (segmentationTransferTable[i] == 0)
				{
					return;
				}
				int curCount = atomicAdd(count, 1);
				voxels[curCount].color.w = segmentationTransferTable[i];
				voxels[curCount].color.z = anatomicalImage[curPos];
				voxels[curCount].color.y = anatomicalImage[curPos + 1];
				voxels[curCount].color.x = anatomicalImage[curPos + 2];
				voxels[curCount].index = width * height * curNumber + width * y + x;
				return;
			}
		}
	}
}

void GetVoxelsAnatomicalSegmentation(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, unsigned char* segmentationTransferTable, int eps, RGBVoxel* voxels, int width, int height, int curDepth, int curNumber, int* count)
{
	dim3 blockSize(32, 32);
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	GetVoxelsAnatomicalSegmentationKernel <<<gridSize, blockSize >>> (anatomicalImage, segmentedImage, segmentationTable, segmentsCount, segmentationTransferTable, eps, voxels, width, height, curDepth, curNumber, count);
}

struct CompareVoxelsRed
{
	__host__ __device__ bool operator()(const RGBVoxel &a, const RGBVoxel &b) { return (a.color.x < b.color.x); };
};
struct CompareVoxelsGreen
{
	__host__ __device__ bool operator()(const RGBVoxel &a, const RGBVoxel &b) { return (a.color.y < b.color.y); };
};
struct CompareVoxelsBlue
{
	__host__ __device__ bool operator()(const RGBVoxel &a, const RGBVoxel &b) { return (a.color.z < b.color.z); };
};
struct CompareVoxelsIndex
{
	__host__ __device__ bool operator()(const RGBVoxel &a, const RGBVoxel &b) { return (a.index < b.index); };
};

struct RGBToPalette
{
	int paletteIndex;
	RGBToPalette(int _palettedIndex)
	{
		paletteIndex = _palettedIndex;
	}
	__host__ __device__ RGBVoxel operator()(const RGBVoxel &a)
	{ 
		RGBVoxel result;
		result.index = a.index;
		result.color.x = paletteIndex;
		result.color.y = a.color.w;
		result.color.z = 0;
		result.color.w = 0;
		return result;
	};
};

struct ReduceColors
{
	__host__ __device__ ulonglong4 operator()(const RGBVoxel &a, const RGBVoxel &b)
	{ 
		ulonglong4 res;
		res.x = a.color.x + b.color.x;
		res.y = a.color.y + b.color.y;
		res.z = a.color.z + b.color.z;
		return res; 
	};
};

struct DistanceToWeight
{
	__host__ __device__ uint2 operator()(const uint2 &a)
	{ 
		float weight = a.x;
		weight += a.y;
		uint2 res = *reinterpret_cast<uint2*>(&weight);
		return res;
	};
};

void CUDACreateFromSlices(string anatomicalFolder, string segmentedFolder, vector<SegmentData>& segmentationTable, vector<unsigned char>& segmentationTransfer, int eps, int3& dim, vector<Voxel>& voxels, vector<uchar4>& palette)
{

	HANDLE hA;
	WIN32_FIND_DATAA fA;
	HANDLE hS;
	WIN32_FIND_DATAA fS;
	vector<string> filesA;
	vector<string> filesS;
	hA = FindFirstFileA((anatomicalFolder + "*").c_str(), &fA);  //Find "."
	FindNextFileA(hA, &fA); //Find ".."
	FindNextFileA(hA, &fA); //Find real filename
	Mat img = imread(anatomicalFolder + fA.cFileName);
	dim.x = img.size().width;
	dim.y = img.size().height;
	dim.z = 1;
	while (FindNextFileA(hA, &fA))
	{
		dim.z++;
	}
	hA = FindFirstFileA((anatomicalFolder + "*").c_str(), &fA);  //Find "."
	FindNextFileA(hA, &fA); //Find ".."
	hS = FindFirstFileA((segmentedFolder + "*").c_str(), &fS);  //Find "."
	FindNextFileA(hS, &fS); //Find ".."
	int curDepth = 0;
	int curNumber = 0;
	unsigned char* hDataA;
	unsigned char* hDataS;
	unsigned char* gDataA;
	unsigned char* gDataS;
	SegmentData* gSegmentTable;
	unsigned char* gSegmentationTransfer;
	thrust::device_vector<RGBVoxel> dVoxelsSlice(dim.x*dim.y);
	thrust::device_vector<RGBVoxel> dVoxels;
	int* gCount;
	int hCount;
	cudaMalloc((void**)&gDataA, sizeof(unsigned char)*(dim.x*dim.y) * 3);
	cudaMalloc((void**)&gDataS, sizeof(unsigned char)*(dim.x*dim.y) * 3);
	cudaMalloc((void**)&gSegmentTable, sizeof(SegmentData)*(segmentationTable.size()));
	cudaMemcpy(gSegmentTable, &segmentationTable[0], sizeof(SegmentData)*(segmentationTable.size()), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gSegmentationTransfer, sizeof(SegmentData)*(segmentationTransfer.size()));
	cudaMemcpy(gSegmentationTransfer, &segmentationTransfer[0], sizeof(SegmentData)*(segmentationTransfer.size()), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gCount, sizeof(int));
	Mat mA;
	Mat mS;
	while (FindNextFileA(hA, &fA) && FindNextFileA(hS, &fS))
	{
		mA = imread(anatomicalFolder + fA.cFileName);
		mS = imread(segmentedFolder + fS.cFileName);
		curDepth = atoi(fA.cFileName);
		cudaMemset(gCount, 0, sizeof(int));
		hDataA = mA.data;
		hDataS = mS.data;
		cudaMemcpy(gDataA, hDataA, sizeof(unsigned char)*(dim.x*dim.y) * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(gDataS, hDataS, sizeof(unsigned char)*(dim.x*dim.y) * 3, cudaMemcpyHostToDevice);
		GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps, thrust::raw_pointer_cast(dVoxelsSlice.data()), dim.x, dim.y, curDepth, curNumber, gCount);
		cudaMemcpy(&hCount, gCount, sizeof(int), cudaMemcpyDeviceToHost);
		if (hCount > 0)
		{
			int curSize = dVoxels.size();
			dVoxels.resize(curSize + hCount);
			thrust::copy(dVoxelsSlice.begin(), dVoxelsSlice.begin()+hCount, dVoxels.begin()+ curSize);
		}
		curNumber++;
	}
	cudaFree(gDataA);
	cudaFree(gDataS);
	cudaFree(gSegmentTable);
	cudaFree(gSegmentationTransfer);
	cudaFree(gCount);

	queue<PaletteElement> qPalette;
	qPalette.emplace(dVoxels.size());
	vector<PaletteElement> finalPaletteElements;
	RGBVoxel min;
	RGBVoxel max;
	int len = 0;

	while (!qPalette.empty())
	{
		PaletteElement cur = qPalette.front();
		min = *thrust::min_element(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsRed());
		max = *thrust::max_element(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsRed());
		len = max.color.x - min.color.x;
		cur.sortMode = PaletteElement::SORT_MODE::SORT_MODE_RED;
		min = *thrust::min_element(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsGreen());
		max = *thrust::max_element(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsGreen());
		if (len < (max.color.y - min.color.y))
		{
			len = max.color.y - min.color.y;
			cur.sortMode = PaletteElement::SORT_MODE::SORT_MODE_GREEN;
		}	
		min = *thrust::min_element(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsBlue());
		max = *thrust::max_element(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsBlue());
		if (len < (max.color.z - min.color.z))
		{
			len = max.color.z - min.color.z;
			cur.sortMode = PaletteElement::SORT_MODE::SORT_MODE_BLUE;
		}
		switch (cur.sortMode)
		{
		case PaletteElement::SORT_MODE::SORT_MODE_RED:
			thrust::sort(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsRed());
			break;
		case PaletteElement::SORT_MODE::SORT_MODE_GREEN:
			thrust::sort(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsGreen());
			break;
		case PaletteElement::SORT_MODE::SORT_MODE_BLUE:
			thrust::sort(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsBlue());
			break;
		}
		qPalette.pop();
		if (cur.level == 8)
		{
			finalPaletteElements.push_back(cur);
		}
		else
		{
			qPalette.emplace(cur, true);
			qPalette.emplace(cur, false);
		}
	}
	thrust::host_vector<RGBVoxel> hVoxels = dVoxels;
	vector<pair<uchar4, int>> tmpPalette;
	for (int i = 0; i < finalPaletteElements.size(); i++)
	{
		UINT64 r = 0;
		UINT64 g = 0;
		UINT64 b = 0;
		for (int j = finalPaletteElements[i].start; j < finalPaletteElements[i].start + finalPaletteElements[i].length; j++)
		{
			r += hVoxels[j].color.x;
			g += hVoxels[j].color.y;
			b += hVoxels[j].color.z;
		}
		uchar4 color = { r / finalPaletteElements[i].length,g / finalPaletteElements[i].length,b / finalPaletteElements[i].length,0 };
		tmpPalette.emplace_back(color, i);		
	}
	hVoxels.clear();
	hVoxels.shrink_to_fit();
	std::sort(tmpPalette.begin(), tmpPalette.end(), [](auto &a, auto&b) {return CompareColorsIntensity(a.first, b.first); });
	for (int i = 0; i < tmpPalette.size(); i++)
	{
		palette.push_back(tmpPalette[i].first);
		RGBToPalette curTransform(i);
		thrust::transform(dVoxels.begin() + finalPaletteElements[tmpPalette[i].second].start, dVoxels.begin() + finalPaletteElements[tmpPalette[i].second].start + finalPaletteElements[tmpPalette[i].second].length, dVoxels.begin() + finalPaletteElements[tmpPalette[i].second].start, curTransform);
	}
	thrust::sort(dVoxels.begin(), dVoxels.end(), CompareVoxelsIndex());
	voxels.resize(dVoxels.size());
	cudaMemcpy(&voxels[0], thrust::raw_pointer_cast(dVoxels.data()), sizeof(Voxel)*voxels.size(), cudaMemcpyDeviceToHost);
}

__device__ void SetMaskElement(uint index, int* mask)
{
	int pos = 1 << (index % 32);
	atomicOr(&mask[index / 32], pos);
}

__device__ bool GetMaskElement(uint index, int* mask)
{
	int pos = 1 << (index % 32);
	return (pos & mask[index / 32]);
}

__device__ int VoxelBinSearch(uint voxelIndex, Voxel* voxels, uint voxelsCount)
{
	uint left = 0;
	uint right = voxelsCount - 1;
	uint mid;
	uint curValue;
	while ((right - left) > 0)
	{
		mid = (left + right) / 2;
		curValue = voxels[mid].index;
		if (voxelIndex == curValue)
		{
			return mid;
		}
		if (voxelIndex < curValue)
		{
			right = mid;
		}
		else
		{
			left = mid + 1;
		}
	}
	curValue = voxels[right].index;
	if (voxelIndex == curValue)
	{
		return right;
	}
	return (-1);
}

__global__ void CalculateIntersectingVoxelsKernel(Voxel* voxels, int3 voxelsDim, uint voxelsCount, uint* dist02, uchar* bones02, int boneIndex, float3 invDir, float3 dirOrigin, int* mask, int* count)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	uint index = y * gridDim.x * blockDim.x + x;
	if (index >= voxelsCount)
	{
		return;
	}
	if (dist02[index] == 0)
	{
		return;
	}
	uint voxIndex = voxels[index].index;
	float3 voxPos;
	voxPos.z = voxIndex / (voxelsDim.y * voxelsDim.x);
	uint interm = voxIndex % (voxelsDim.y * voxelsDim.x);
	voxPos.y = interm / voxelsDim.x;
	voxPos.x = interm % voxelsDim.x;
	///Intersecting
	float tmin = (voxPos.x - dirOrigin.x) * invDir.x;
	float tmax = (voxPos.x + 1.0f - dirOrigin.x) * invDir.x;
	if (tmax < tmin)
	{
		float tmp = tmin;
		tmin = tmax;
		tmax = tmp;
	}
	float t1min = (voxPos.y - dirOrigin.y) * invDir.y;
	float t1max = (voxPos.y + 1.0f - dirOrigin.y) * invDir.y;
	if (t1max < t1min)
	{
		float tmp = t1min;
		t1min = t1max;
		t1max = tmp;
	}
	if ((tmin > t1max) || (t1min > tmax))
	{
		return;
	}
	if (t1min > tmin)
	{
		tmin = t1min;
	}
	if (t1max < tmax)
	{
		tmax = t1max;
	}
	t1min = (voxPos.z - dirOrigin.z) * invDir.z;
	t1max = (voxPos.z + 1.0f - dirOrigin.z) * invDir.z;
	if (t1max < t1min)
	{
		float tmp = t1min;
		t1min = t1max;
		t1max = tmp;
	}
	if ((tmin > t1max) || (t1min > tmax))
	{
		return;
	}
	if (t1min > tmin)
	{
		tmin = t1min;
	}
	if (t1max < tmax)
	{
		tmax = t1max;
	}
	//Tag
	if (((tmin >= 0.0f) && (tmin <= 1.0f)) || ((tmax >= 0.0f) && (tmax <= 1.0f)))
	{
		dist02[index] = 0;
		atomicAdd(count, 1);
		SetMaskElement(index, mask);
		bones02[index] = boneIndex;
	}
}

void CalculateIntersectingVoxels(Voxel* voxels, int3 voxelsDim, uint voxelsCount, uint* dist02, uchar* bones02, int boneIndex, float3 invDir, float3 dirOrigin, int* mask, int* count)
{
	dim3 blockSize(32, 32);
	int computeBlocksCount = ceil(sqrt(voxelsCount));
	computeBlocksCount = ceil(computeBlocksCount / 32.0);
	dim3 gridSize(computeBlocksCount, computeBlocksCount);
	CalculateIntersectingVoxelsKernel <<<gridSize, blockSize>>>(voxels, voxelsDim, voxelsCount, dist02, bones02, boneIndex, invDir, dirOrigin, mask, count);
}

__global__ void CalculateGeodesicDistancesKernel(Voxel* voxels, int3 voxelsDim, uint voxelsCount, uint* dist02, uchar* bones02, int borderSegment, int boneIndex, int* readMask, int* writeMask, int* count)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	uint index = y * gridDim.x * blockDim.x + x;
	if (index >= voxelsCount)
	{
		return;
	}
	if (!GetMaskElement(index, readMask))
	{
		return;
	}
	if (voxels[index].segment == borderSegment)
	{
		return;
	}
	uint nextDist = dist02[index] + 1;
	uint voxelIndex = voxels[index].index;

	int neighborArrayIndex = VoxelBinSearch(voxelIndex - 1, voxels, voxelsCount);
	if (neighborArrayIndex != -1)
	{
		if (atomicMin(&dist02[neighborArrayIndex], nextDist) > nextDist)
		{
			atomicAdd(count, 1);
			SetMaskElement(neighborArrayIndex, writeMask);
			bones02[neighborArrayIndex] = boneIndex;
		}
	}
	neighborArrayIndex = VoxelBinSearch(voxelIndex + 1, voxels, voxelsCount);
	if (neighborArrayIndex != -1)
	{
		if (atomicMin(&dist02[neighborArrayIndex], nextDist) > nextDist)
		{
			atomicAdd(count, 1);
			SetMaskElement(neighborArrayIndex, writeMask);
			bones02[neighborArrayIndex] = boneIndex;
		}
	}
	neighborArrayIndex = VoxelBinSearch(voxelIndex - voxelsDim.x, voxels, voxelsCount);
	if (neighborArrayIndex != -1)
	{
		if (atomicMin(&dist02[neighborArrayIndex], nextDist) > nextDist)
		{
			atomicAdd(count, 1);
			SetMaskElement(neighborArrayIndex, writeMask);
			bones02[neighborArrayIndex] = boneIndex;
		}
	}
	neighborArrayIndex = VoxelBinSearch(voxelIndex + voxelsDim.x, voxels, voxelsCount);
	if (neighborArrayIndex != -1)
	{
		if (atomicMin(&dist02[neighborArrayIndex], nextDist) > nextDist)
		{
			atomicAdd(count, 1);
			SetMaskElement(neighborArrayIndex, writeMask);
			bones02[neighborArrayIndex] = boneIndex;
		}
	}
	neighborArrayIndex = VoxelBinSearch(voxelIndex - voxelsDim.y * voxelsDim.x, voxels, voxelsCount);
	if (neighborArrayIndex != -1)
	{
		if (atomicMin(&dist02[neighborArrayIndex], nextDist) > nextDist)
		{
			atomicAdd(count, 1);
			SetMaskElement(neighborArrayIndex, writeMask);
			bones02[neighborArrayIndex] = boneIndex;
		}
	}
	neighborArrayIndex = VoxelBinSearch(voxelIndex + voxelsDim.y * voxelsDim.x, voxels, voxelsCount);
	if (neighborArrayIndex != -1)
	{
		if (atomicMin(&dist02[neighborArrayIndex], nextDist) > nextDist)
		{
			atomicAdd(count, 1);
			SetMaskElement(neighborArrayIndex, writeMask);
			bones02[neighborArrayIndex] = boneIndex;
		}
	}
}

void CalculateGeodesicDistances(Voxel* voxels, int3 voxelsDim, uint voxelsCount, uint* dist02, uchar* bones02, int borderSegment, int boneIndex, int* readMask, int* writeMask, int* count)
{
	dim3 blockSize(32, 32);
	int computeBlocksCount = ceil(sqrt(voxelsCount));
	computeBlocksCount = ceil(computeBlocksCount / 32.0);
	dim3 gridSize(computeBlocksCount, computeBlocksCount);
	CalculateGeodesicDistancesKernel << <gridSize, blockSize >> > (voxels, voxelsDim, voxelsCount, dist02, bones02, borderSegment, boneIndex, readMask, writeMask, count);
}

__global__ void SwapDistancesKernel(Voxel* voxels, uint voxelsCount, uint* dist00, uint* dist01, uint* dist02, uchar* bones02)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	uint index = y * gridDim.x * blockDim.x + x;
	if (index >= voxelsCount)
	{
		return;
	}
	if (dist02[index] < dist00[index])
	{
		uint tmpDist = dist02[index];
		uchar tmpBone = bones02[index];
		dist02[index] = dist01[index];
		bones02[index] = voxels[index].bone01;
		dist01[index] = dist00[index];
		voxels[index].bone01 = voxels[index].bone00;
		dist00[index] = tmpDist;
		voxels[index].bone00 = tmpBone;
	}
	else if (dist02[index] < dist01[index])
	{
		uint tmpDist = dist02[index];
		uchar tmpBone = bones02[index];
		dist02[index] = dist01[index];
		bones02[index] = voxels[index].bone01;
		dist01[index] = tmpDist;
		voxels[index].bone01 = tmpBone;
	}
}

void SwapDistances(Voxel* voxels, uint voxelsCount, uint* dist00, uint* dist01, uint* dist02, uchar* bones02)
{
	dim3 blockSize(32, 32);
	int computeBlocksCount = ceil(sqrt(voxelsCount));
	computeBlocksCount = ceil(computeBlocksCount / 32.0);
	dim3 gridSize(computeBlocksCount, computeBlocksCount);
	SwapDistancesKernel << <gridSize, blockSize >> > (voxels, voxelsCount, dist00, dist01, dist02, bones02);
}

__global__ void DistancesToWeightsKernel(uint voxelsCount, uint* dist00, uint* dist01, uint* dist02)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	uint index = y * gridDim.x * blockDim.x + x;
	if (index >= voxelsCount)
	{
		return;
	}

	float weight00 = (0.1f + dist00[index]);
	float weight01 = (0.1f + dist01[index]);
	float weight02 = (0.1f + dist02[index]);
	float sum = (weight00 + weight01 + weight02);
	weight00 = weight00 / sum;
	weight01 = weight01 / sum;
	weight02 = weight02 / sum;

	weight00 = 1.0f / weight00;
	weight00 *= weight00; //2
	weight00 *= weight00; //4
	weight00 *= weight00; //8

	weight01 = 1.0f / weight01;
	weight01 *= weight01; //2
	weight01 *= weight01; //4
	weight01 *= weight01; //8

	weight02 = 1.0f / weight02;
	weight02 *= weight02; //2
	weight02 *= weight02; //4
	weight02 *= weight02; //8
	sum = (weight00 + weight01 + weight02);
	weight00 = weight00 / sum;
	dist00[index] = *reinterpret_cast<uint*>(&weight00);
	weight01 = weight01 / sum;
	dist01[index] = *reinterpret_cast<uint*>(&weight01);
}

void DistancesToWeights(uint voxelsCount, uint* dist00, uint* dist01, uint* dist02)
{
	dim3 blockSize(32, 32);
	int computeBlocksCount = ceil(sqrt(voxelsCount));
	computeBlocksCount = ceil(computeBlocksCount / 32.0);
	dim3 gridSize(computeBlocksCount, computeBlocksCount);
	DistancesToWeightsKernel << <gridSize, blockSize >> > (voxelsCount, dist00, dist01, dist02);
}

void CUDACalculateWeights(vector<Voxel>& voxels, int3 voxelsDim, vector<float>& weights00, vector<float>& weights01, vector<uchar>& bones02, int borderSegment, vector<pair<Vector3,Vector3>>& bonesPoints)
{
	int hCount;
	int* dCount;
	cudaMalloc((void**)&dCount, sizeof(int));
	int* dMask00;
	int* dMask01;
	int bitsetSize = (voxels.size() - 1) / (sizeof(int) * 8) + 1;
	cudaMalloc((void**)&dMask00, sizeof(int) * bitsetSize);
	cudaMalloc((void**)&dMask01, sizeof(int) * bitsetSize);
	cudaMemset(dMask00, 0, sizeof(int) * bitsetSize);
	cudaMemset(dMask01, 0, sizeof(int) * bitsetSize);
	uint distInit = UINT_MAX;
	thrust::device_vector<uint> dDist00(voxels.size(), distInit);
	thrust::device_vector<uint> dDist01(voxels.size(), distInit);
	thrust::device_vector<uint> dDist02(voxels.size(), distInit);
	thrust::device_vector<uchar> dBones02(voxels.size());
	Voxel* dVoxels;
	cudaMalloc((void**)&dVoxels, sizeof(Voxel) * voxels.size());
	cudaMemcpy(dVoxels, &voxels[0], sizeof(Voxel) * voxels.size(), cudaMemcpyHostToDevice);
	for (int i = 1; i < bonesPoints.size(); i++)
	{
		cudaMemset(dCount, 0, sizeof(int));
		cudaMemset(dMask00, 0, sizeof(int) * bitsetSize);
		Vector3 invDir = bonesPoints[i].second - bonesPoints[i].first;
		invDir.x = 1.0f / invDir.x;
		invDir.y = 1.0f / invDir.y;
		invDir.z = 1.0f / invDir.z;
		float3 invDirF = { invDir.x, invDir.y, invDir.z };
		float3 dirOriginF = { bonesPoints[i].first.x, bonesPoints[i].first.y, bonesPoints[i].first.z };
		CalculateIntersectingVoxels(dVoxels, voxelsDim, voxels.size(), thrust::raw_pointer_cast(dDist02.data()), thrust::raw_pointer_cast(dBones02.data()), i, invDirF, dirOriginF, dMask00, dCount);
		cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);
		bool isFirstMask = true;
		while (hCount > 0)
		{
			cudaMemset(dCount, 0, sizeof(int));
			if (isFirstMask)
			{
				CalculateGeodesicDistances(dVoxels, voxelsDim, voxels.size(), thrust::raw_pointer_cast(dDist02.data()), thrust::raw_pointer_cast(dBones02.data()), borderSegment, i, dMask00, dMask01, dCount);
				isFirstMask = !isFirstMask;
				cudaMemset(dMask00, 0, sizeof(int) * bitsetSize);
			}
			else
			{
				CalculateGeodesicDistances(dVoxels, voxelsDim, voxels.size(), thrust::raw_pointer_cast(dDist02.data()), thrust::raw_pointer_cast(dBones02.data()), borderSegment, i, dMask01, dMask00, dCount);
				isFirstMask = !isFirstMask;
				cudaMemset(dMask01, 0, sizeof(int) * bitsetSize);
			}
			cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);
		}
		SwapDistances(dVoxels, voxels.size(), thrust::raw_pointer_cast(dDist00.data()), thrust::raw_pointer_cast(dDist01.data()), thrust::raw_pointer_cast(dDist02.data()), thrust::raw_pointer_cast(dBones02.data()));
	}
	DistancesToWeights(voxels.size(), thrust::raw_pointer_cast(dDist00.data()), thrust::raw_pointer_cast(dDist01.data()), thrust::raw_pointer_cast(dDist02.data()));
	cudaMemcpy(&voxels[0], dVoxels, sizeof(Voxel)*voxels.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(&weights00[0], thrust::raw_pointer_cast(dDist00.data()), sizeof(float)*voxels.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(&weights01[0], thrust::raw_pointer_cast(dDist01.data()), sizeof(float)*voxels.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bones02[0], thrust::raw_pointer_cast(dBones02.data()), sizeof(uchar)*voxels.size(), cudaMemcpyDeviceToHost);

	cudaFree(dCount);
	cudaFree(dMask00);
	cudaFree(dMask01);
	cudaFree(dVoxels);
}
