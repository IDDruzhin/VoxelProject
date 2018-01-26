#include "cuda_runtime.h"
#include "Structures.h"

void GetVoxelsAnatomicalSegmentation(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, int* segmentationTransferTable, int eps, VoxelInfo* voxels, int width, int height, int curDepth, int depthMultiplier, int* count);