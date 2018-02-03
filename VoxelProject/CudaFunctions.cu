#include "stdafx.h"
#include "CudaFunctions.cuh"


__global__ void GetVoxelsAnatomicalSegmentationKernel(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, unsigned char* segmentationTransferTable, int eps, RGBVoxel* voxels, int width, int height, int curDepth, int depthMultiplier, int* count)
{
	//unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
	int curPos = (y*width + x) * 3;
	int multipliedDepth = curDepth * depthMultiplier;
	int4 pixel;
	pixel.z = segmentedImage[curPos];
	pixel.y = segmentedImage[curPos + 1];
	pixel.x = segmentedImage[curPos + 2];
	//pixel.w = 255;
	if ((pixel.x == segmentationTable[0].color.x) && (pixel.y == segmentationTable[0].color.y) && (pixel.z == segmentationTable[0].color.z))
	{
		return;
	}
	for (int i = 1; i < segmentsCount; i++)
	{
		if ((multipliedDepth >= segmentationTable[i].start) && (multipliedDepth <= segmentationTable[i].finish))
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
				voxels[curCount].index = width * height*curDepth + width * y + x;
				return;
			}
		}
	}
	/*
	{
		int curCount = atomicAdd(count, 1);
		voxels[curCount].color.w = 1;
		voxels[curCount].color.z = anatomicalImage[curPos];
		voxels[curCount].color.y = anatomicalImage[curPos + 1];
		voxels[curCount].color.x = anatomicalImage[curPos + 2];
		voxels[curCount].index = width * height*curDepth + width * y + x;
	}
	*/
}

void GetVoxelsAnatomicalSegmentation(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, unsigned char* segmentationTransferTable, int eps, RGBVoxel* voxels, int width, int height, int curDepth, int depthMultiplier, int* count)
{
	dim3 blockSize(32, 32);
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	GetVoxelsAnatomicalSegmentationKernel <<<gridSize, blockSize >>> (anatomicalImage, segmentedImage, segmentationTable, segmentsCount, segmentationTransferTable, eps, voxels, width, height, curDepth, depthMultiplier, count);
}

/*
__host__ __device__ bool CompareVoxelsRed(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.x < b.color.x);
}
__host__ __device__ bool CompareVoxelsGreen(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.y < b.color.y);
}
__host__ __device__ bool CompareVoxelsBlue(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.z < b.color.z);
}
*/

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

void CUDACreateFromSlices(string anatomicalFolder, string segmentedFolder, vector<SegmentData>& segmentationTable, vector<unsigned char>& segmentationTransfer, int depthMulptiplier, int eps, uint3& dim, vector<Voxel>& voxels, vector<uchar4>& palette)
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
	int Time = clock();
	Mat mA;
	Mat mS;
	while (FindNextFileA(hA, &fA) && FindNextFileA(hS, &fS))
	{
		mA = imread(anatomicalFolder + fA.cFileName);
		mS = imread(segmentedFolder + fS.cFileName);
		cudaMemset(gCount, 0, sizeof(int));
		hDataA = mA.data;
		hDataS = mS.data;
		cudaMemcpy(gDataA, hDataA, sizeof(unsigned char)*(dim.x*dim.y) * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(gDataS, hDataS, sizeof(unsigned char)*(dim.x*dim.y) * 3, cudaMemcpyHostToDevice);
		GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps, thrust::raw_pointer_cast(dVoxelsSlice.data()), dim.x, dim.y, curDepth, depthMulptiplier, gCount);
		cudaMemcpy(&hCount, gCount, sizeof(int), cudaMemcpyDeviceToHost);
		if (hCount > 0)
		{
			int curSize = dVoxels.size();
			dVoxels.resize(curSize + hCount);
			thrust::copy(dVoxelsSlice.begin(), dVoxelsSlice.begin()+hCount, dVoxels.begin()+ curSize);
		}
		curDepth++;
	}
	Time = clock() - Time;
	cudaFree(gDataA);
	cudaFree(gDataS);
	cudaFree(gSegmentTable);
	cudaFree(gSegmentationTransfer);
	cudaFree(gCount);

	queue<PaletteElement> qPalette;
	qPalette.emplace(dVoxels.size());
	vector<PaletteElement> finalPaletteElements;

	while (!qPalette.empty())
	{
		PaletteElement cur = qPalette.front();
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
