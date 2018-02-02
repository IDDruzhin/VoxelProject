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
				//int index = width * height*curDepth + width * y + x;
				//atomicOr(Mask + (Index / (sizeof(int) * 8)), (Index << Index % (sizeof(int) * 8)));
				voxels[curCount].segment = segmentationTransferTable[i];
				//pixel.z = anatomicalImage[curPos];
				//pixel.y = anatomicalImage[curPos + 1];
				//pixel.x = anatomicalImage[curPos + 2];
				//voxels[curCount].color = pixel;
				voxels[curCount].color.z = anatomicalImage[curPos];
				voxels[curCount].color.y = anatomicalImage[curPos + 1];
				voxels[curCount].color.x = anatomicalImage[curPos + 2];
				voxels[curCount].index = width * height*curDepth + width * y + x;
				return;
			}
		}
	}
	{
		int curCount = atomicAdd(count, 1);
		//int Index = width * Height*CurDepth + Width * y + x;
		//atomicOr(Mask + (Index / (sizeof(int) * 8)), (Index << Index % (sizeof(int) * 8)));
		voxels[curCount].segment = 1;
		//pixel.z = anatomicalImage[curPos];
		//pixel.y = anatomicalImage[curPos + 1];
		//pixel.x = anatomicalImage[curPos + 2];
		//voxels[curCount].color = pixel;
		voxels[curCount].color.z = anatomicalImage[curPos];
		voxels[curCount].color.y = anatomicalImage[curPos + 1];
		voxels[curCount].color.x = anatomicalImage[curPos + 2];
		voxels[curCount].index = width * height*curDepth + width * y + x;
	}
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

/*
//__host__ __device__ bool operator<(const RGBVoxel &a, const RGBVoxel &b) { return (a.color.x < b.color.x);};
__host__ __device__ bool operator<(const RGBVoxel &a, const RGBVoxel &b)
{
	return (a.color.z < b.color.z); 
}
*/

void CUDACreateFromSlices(string anatomicalFolder, string segmentedFolder, vector<SegmentData>& segmentationTable, vector<unsigned char>& segmentationTransfer, int depthMulptiplier, int eps, uint3& dim, vector<Voxel>& voxels, vector<uchar4>& palette)
{

	HANDLE hA;
	WIN32_FIND_DATAA fA;
	HANDLE hS;
	WIN32_FIND_DATAA fS;
	vector<string> filesA;
	vector<string> filesS;
	bool moreFilesA = true;
	bool moreFilesS = true;
	bool moreFiles = true;
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
	vector<RGBVoxel> hVoxels;
	//thrust::device_vector<int> dVoxelsSlice(10);
	//thrust::device_vector<RGBVoxel> dVoxelsSlice;
	thrust::device_vector<RGBVoxel> dVoxelsSlice(dim.x*dim.y);
	//dVoxelsSlice.resize(m_dim.x*m_dim.y);
	thrust::device_vector<RGBVoxel> dVoxels;
	//thrust::device_vector<RGBVoxel> dVoxels;
	//RGBVoxel* gVoxels;
	int* gCount;
	int hCount;
	cudaMalloc((void**)&gDataA, sizeof(unsigned char)*(dim.x*dim.y) * 3);
	cudaMalloc((void**)&gDataS, sizeof(unsigned char)*(dim.x*dim.y) * 3);
	cudaMalloc((void**)&gSegmentTable, sizeof(SegmentData)*(segmentationTable.size()));
	cudaMemcpy(gSegmentTable, &segmentationTable[0], sizeof(SegmentData)*(segmentationTable.size()), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gSegmentationTransfer, sizeof(SegmentData)*(segmentationTransfer.size()));
	cudaMemcpy(gSegmentationTransfer, &segmentationTransfer[0], sizeof(SegmentData)*(segmentationTransfer.size()), cudaMemcpyHostToDevice);
	//cudaMalloc((void**)&gVoxels, sizeof(RGBVoxel)*(dim.x*dim.y));
	cudaMalloc((void**)&gCount, sizeof(int));
	int Time = clock();
	Mat mA;
	Mat mS;
	while (FindNextFileA(hA, &fA) && FindNextFileA(hS, &fS))
	{
		//Mat mA = imread(anatomicalFolder + fA.cFileName);
		//Mat mS = imread(segmentedFolder + fS.cFileName);
		mA = imread(anatomicalFolder + fA.cFileName);
		mS = imread(segmentedFolder + fS.cFileName);
		cudaMemset(gCount, 0, sizeof(int));
		hDataA = mA.data;
		hDataS = mS.data;
		cudaMemcpy(gDataA, hDataA, sizeof(unsigned char)*(dim.x*dim.y) * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(gDataS, hDataS, sizeof(unsigned char)*(dim.x*dim.y) * 3, cudaMemcpyHostToDevice);
		//GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps, gVoxels, dim.x, dim.y, curDepth, depthMulptiplier, gCount);
		//cudaDeviceSynchronize();
		GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps, thrust::raw_pointer_cast(dVoxelsSlice.data()), dim.x, dim.y, curDepth, depthMulptiplier, gCount);
		cudaMemcpy(&hCount, gCount, sizeof(int), cudaMemcpyDeviceToHost);
		if (hCount > 0)
		{
			//int curSize = dVoxels.size();
			//dVoxels.resize(curSize + hCount);
			//thrust::copy(dVoxelsSlice.begin(), dVoxelsSlice.begin()+curSize, dVoxels.begin()+curSize);

			int curSize = dVoxels.size();
			dVoxels.resize(curSize + hCount);
			thrust::copy(dVoxelsSlice.begin(), dVoxelsSlice.begin()+hCount, dVoxels.begin()+ curSize);
			//cudaMemcpy(&hVoxels[curSize], gVoxels, sizeof(RGBVoxel)*hCount, cudaMemcpyDeviceToHost);
			//sort(hVoxels.begin() + curSize, hVoxels.end(), CompareVoxels);
		}
		curDepth++;
	}
	Time = clock() - Time;
	cudaFree(gDataA);
	cudaFree(gDataS);
	cudaFree(gSegmentTable);
	cudaFree(gSegmentationTransfer);
	//cudaFree(gVoxels);
	cudaFree(gCount);

	queue<PaletteElement> qPalette;
	qPalette.emplace(dVoxels.size());
	vector<PaletteElement> finalPaletteElements;
	//thrust::sort(dVoxels.begin(), dVoxels.end());
	//thrust::sort(dVoxels.begin(), dVoxels.end(), CompareVoxelsGreen);
	//thrust::sort(dVoxels.begin(), dVoxels.end(), CompareVoxelsBlue());

	//thrust::sort(dVoxels.begin()+qPalette.front().start, dVoxels.begin() + qPalette.front().start+qPalette.front().length, CompareVoxelsRed());
	//hVoxels.resize(dVoxels.size());
	//cudaMemcpy(&hVoxels[0], thrust::raw_pointer_cast(dVoxels.data()), sizeof(RGBVoxel)*dVoxels.size(), cudaMemcpyDeviceToHost);

	while (!qPalette.empty())
	{
		PaletteElement cur = qPalette.front();
		//thrust::sort(dVoxels.begin(), dVoxels.end());
		//thrust::sort(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length);
		//thrust::sort(dVoxels.begin(), dVoxels.begin() + 100);
		switch (cur.sortMode)
		{
		case PaletteElement::SORT_MODE::SORT_MODE_RED:
			thrust::sort(dVoxels.begin(), dVoxels.begin()+100, CompareVoxelsRed());
			//thrust::sort(dVoxels.begin() + cur.start, dVoxels.begin() + cur.start + cur.length, CompareVoxelsRed);
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
	hVoxels.resize(dVoxels.size());
	cudaMemcpy(&hVoxels[0], thrust::raw_pointer_cast(dVoxels.data()), sizeof(RGBVoxel)*dVoxels.size(), cudaMemcpyDeviceToHost);
	vector<pair<uchar4, int>> tmpPalette;
	for (int i = 0; i < finalPaletteElements.size(); i++)
	{
		/*
		UINT64 r = 0;
		UINT64 g = 0;
		UINT64 b = 0;
		int length = finalPaletteElements[i].length;
		auto cur = hVoxels.begin() + finalPaletteElements[i].start;
		auto finish = cur + length;
		for (; cur != finish; cur++)
		{
			r += cur->color.x;
			g += cur->color.y;
			b += cur->color.z;
		}
		uchar4 color = { r / length,g / length,b / length,0 };
		*/
		if (finalPaletteElements[i].length > 0)
		{
			/*
			ulonglong4 init = { 0,0,0,0 };
			ulonglong4 lColor = thrust::reduce(dVoxels.begin() + finalPaletteElements[i].start, dVoxels.begin() + finalPaletteElements[i].start + finalPaletteElements[i].length,init, ReduceColors());
			uchar4 color;
			color.x = lColor.x / finalPaletteElements[i].length;
			color.y = lColor.y / finalPaletteElements[i].length;
			color.z = lColor.z / finalPaletteElements[i].length;
			tmpPalette.emplace_back(color, i);
			*/
			/*
			uchar4 color;
			if (finalPaletteElements[i].length % 2 == 0)
			{
				auto median = dVoxels.begin() + finalPaletteElements[i].start + finalPaletteElements[i].length / 2;
				uint4 tmpColor = { median->color.x, median->color.y, median->color.z, 0 };
				median += 1;
				tmpColor.x += median->color.x;
				tmpColor.y += median->color.y;
				tmpColor.z += median->color.z;
				color.x = tmpColor.x / 2;
				color.y = tmpColor.y / 2;
				color.z = tmpColor.z / 2;
			}
			else
			{
				auto median = dVoxels.begin() + finalPaletteElements[i].start + finalPaletteElements[i].length / 2;
				color = median->color;
			}
			tmpPalette.emplace_back(color, i);
			*/
		}
		
	}
	std::sort(tmpPalette.begin(), tmpPalette.end(), [](auto &a, auto&b) {return CompareColorsIntensity(a.first, b.first); });
	for (int i = 0; i < tmpPalette.size(); i++)
	{
		palette.push_back(tmpPalette[i].first);
		int length = finalPaletteElements[tmpPalette[i].second].length;
		auto cur = hVoxels.begin() + finalPaletteElements[tmpPalette[i].second].start;
		auto finish = cur + length;
		for (; cur != finish; cur++)
		{
			voxels.emplace_back(cur->index, i, cur->segment);
		}	
	}
}
