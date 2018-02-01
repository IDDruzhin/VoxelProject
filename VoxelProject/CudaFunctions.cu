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
				voxels[curCount].segmentIndex = segmentationTransferTable[i];
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
		voxels[curCount].segmentIndex = 1;
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

void CUDACreateFromSlices(string anatomicalFolder, string segmentedFolder, vector<SegmentData>& segmentationTable, vector<unsigned char>& segmentationTransfer, int depthMulptiplier, int eps, uint3& dim, vector<Voxel>& voxels)
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
	RGBVoxel* gVoxels;
	int* gCount;
	int hCount;
	cudaMalloc((void**)&gDataA, sizeof(unsigned char)*(dim.x*dim.y) * 3);
	cudaMalloc((void**)&gDataS, sizeof(unsigned char)*(dim.x*dim.y) * 3);
	cudaMalloc((void**)&gSegmentTable, sizeof(SegmentData)*(segmentationTable.size()));
	cudaMemcpy(gSegmentTable, &segmentationTable[0], sizeof(SegmentData)*(segmentationTable.size()), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gSegmentationTransfer, sizeof(SegmentData)*(segmentationTransfer.size()));
	cudaMemcpy(gSegmentationTransfer, &segmentationTransfer[0], sizeof(SegmentData)*(segmentationTransfer.size()), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gVoxels, sizeof(RGBVoxel)*(dim.x*dim.y));
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
		GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps, gVoxels, dim.x, dim.y, curDepth, depthMulptiplier, gCount);
		//cudaDeviceSynchronize();
		//GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps, thrust::raw_pointer_cast(dVoxelsSlice.data()), m_dim.x, m_dim.y, curDepth, depthMulptiplier, gCount);
		cudaMemcpy(&hCount, gCount, sizeof(int), cudaMemcpyDeviceToHost);
		if (hCount > 0)
		{
			//int curSize = dVoxels.size();
			//dVoxels.resize(curSize + hCount);
			//thrust::copy(dVoxelsSlice.begin(), dVoxelsSlice.begin()+curSize, dVoxels.begin()+curSize);

			int curSize = hVoxels.size();
			hVoxels.resize(curSize + hCount);
			cudaMemcpy(&hVoxels[curSize], gVoxels, sizeof(RGBVoxel)*hCount, cudaMemcpyDeviceToHost);
			//sort(hVoxels.begin() + curSize, hVoxels.end(), CompareVoxels);
		}
		curDepth++;
	}
	Time = clock() - Time;
	cudaFree(gDataA);
	cudaFree(gDataS);
	cudaFree(gSegmentTable);
	cudaFree(gSegmentationTransfer);
	cudaFree(gVoxels);
	cudaFree(gCount);
}
