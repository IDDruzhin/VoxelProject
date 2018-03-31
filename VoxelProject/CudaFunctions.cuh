#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Structures.h"

using namespace cv;

void GetVoxelsAnatomicalSegmentation(unsigned char* anatomicalImage, unsigned char* segmentedImage, SegmentData* segmentationTable, int segmentsCount, unsigned char* segmentationTransferTable, int eps, RGBVoxel* voxels, int width, int height, int curDepth, int curNumber, int* count);
void CUDACreateFromSlices(string anatomicalFolder, string segmentedFolder, vector<SegmentData>& segmentationTable, vector<uchar>& segmentationTransfer, int eps, int3& dim, vector<Voxel>& voxels, vector<uchar4>& palette);
void CUDACalculateWeights(vector<Voxel>& voxels, int3 voxelsDim, vector<float>& weights00, vector<float>& weights01, vector<uchar>& additionalBonesIndices, vector<pair<Vector3, Vector3>>& bonesPoints);