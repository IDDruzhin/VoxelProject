#include "stdafx.h"
#include "CudaFunctions.cuh"


__global__ void GetVoxelsAnatomicalSegmentationKernel(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, int* segmentationTransferTable, int eps, VoxelInfo* voxels, int width, int height, int curDepth, int depthMultiplier, int* count)
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

void GetVoxelsAnatomicalSegmentation(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, int* segmentationTransferTable, int eps, VoxelInfo* voxels, int width, int height, int curDepth, int depthMultiplier, int* count)
{
	dim3 blockSize(32, 32);
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	GetVoxelsAnatomicalSegmentationKernel <<<gridSize, blockSize >>> (anatomicalImage, segmentedImage, segmentationTable, segmentsCount, segmentationTransferTable, eps, voxels, width, height, curDepth, depthMultiplier, count);
}
