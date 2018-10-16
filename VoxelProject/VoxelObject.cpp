#include "stdafx.h"
#include "VoxelObject.h"


VoxelObject::VoxelObject(VoxelPipeline* voxPipeline)
{

}

VoxelObject::VoxelObject(string path, LOADING_MODE loadingMode, VoxelPipeline * voxPipeline)
{
	if (loadingMode == LOADING_MODE::LOADING_MODE_SLICES)
	{
		CreateFromSlices(path);
	}
	if (loadingMode == LOADING_MODE::LOADING_MODE_BIN)
	{
		LoadBin(path);
	}
	m_paletteRes = voxPipeline->RegisterPalette(m_palette);
	m_segmentsOpacityRes = voxPipeline->RegisterSegmentsOpacity(m_segmentsOpacity);
}

VoxelObject::~VoxelObject()
{
}

void VoxelObject::CreateFromSlices(string path)
{
	ifstream file;
	file.open(path);
	if (file.is_open())
	{
		string anatomicalFolder;
		string segmentedFolder;
		string segmentationTablePath;
		string segmentationTransferPath;
		string segmentationNamesPath;
		string s_depthMultiplier;
		getline(file, m_name);
		getline(file, anatomicalFolder);
		getline(file, segmentedFolder);
		getline(file, segmentationTablePath);
		getline(file, segmentationTransferPath);
		getline(file, segmentationNamesPath);
		getline(file, s_depthMultiplier);
		int depthMulptiplier = atoi(s_depthMultiplier.c_str());
		vector<SegmentData> segmentationTable;
		ifstream segmentationTableFile;
		segmentationTableFile.open(segmentationTablePath);
		if (segmentationTableFile.is_open())
		{
			SegmentData cur;
			string line;
			string word;
			istringstream sS;
			size_t sep;
			while (getline(segmentationTableFile, line))
			{
				sep = line.find_first_of("0123456789");
				line = line.substr(sep, line.length());
				sS.str(line);
				sS >> word;
				cur.color.x = atoi(word.c_str());
				sS >> word;
				cur.color.y = atoi(word.c_str());
				sS >> word;
				cur.color.z = atoi(word.c_str());
				sS >> word;
				cur.start = atoi(word.c_str());
				sS >> word;
				cur.finish = atoi(word.c_str());
				segmentationTable.push_back(cur);
				sS.clear();
			}
			segmentationTableFile.close();
		}
		else
		{
			throw std::exception("Can`t open segmentation table file");
		}
		vector<uchar> segmentationTransfer;
		ifstream segmentationTransferFile;
		segmentationTransferFile.open(segmentationTransferPath);
		if (segmentationTransferFile.is_open())
		{
			string line;
			uchar cur;
			while (getline(segmentationTransferFile, line))
			{
				cur = atoi(line.c_str());
				segmentationTransfer.push_back(cur);
			}
			segmentationTransferFile.close();
		}
		else
		{
			throw std::exception("Can`t open segmentation transfer file");
		}
		if (segmentationTable.size() != segmentationTransfer.size())
		{
			throw std::exception("Incorrect segmentation transfer");
		}
		ifstream segmentationNamesFile;
		segmentationNamesFile.open(segmentationNamesPath);
		if (segmentationNamesFile.is_open())
		{
			string line;
			while (getline(segmentationNamesFile, line))
			{
				m_segmentationTableNames.push_back(line);
			}
			m_segmentsOpacity.resize(m_segmentationTableNames.size(),1.0f);
			m_segmentsOpacity[0] = 0.0f;
			segmentationNamesFile.close();
		}
		else
		{
			throw std::exception("Can`t open segmentation names file");
		}
		int eps = 2;
		CUDACreateFromSlices(anatomicalFolder, segmentedFolder, segmentationTable, segmentationTransfer, eps, m_dim, m_voxels, m_palette);

		float maxSide = (m_dim.x > m_dim.y) ? m_dim.x : m_dim.y;
		maxSide = (maxSide > m_dim.z) ? maxSide : m_dim.z;
		m_skeleton.SetBonesThickness(0.02f * maxSide);
		m_skeleton.SetRootPos(Vector3(m_dim.x, m_dim.y, m_dim.z) / 2.0f);
		maxSide = 1.0f / maxSide;
		m_s = Vector3(maxSide, maxSide, maxSide);
		m_t = (Vector3(m_dim.x, m_dim.y, m_dim.z) * m_s) / -2.0f;
		m_isSkeletonBinded = false;
	}
	else
	{
		throw std::exception("Can`t open input file");
	}
}

void VoxelObject::SaveBin(string path, string name)
{
	ofstream f;
	f.open((path + name + ".bin").c_str(), ios::binary);
	if (f.is_open())
	{
		int count;
		f.write((char*)(&m_dim), sizeof(uint3));
		count = m_palette.size();
		f.write((char*)(&count), sizeof(int));
		f.write((char*)(&m_palette[0]), sizeof(uchar4)*count);
		count = m_voxels.size();
		f.write((char*)(&count), sizeof(int));
		f.write((char*)(&m_voxels[0]), sizeof(Voxel)*count);
		count = m_segmentationTableNames.size();
		f.write((char*)(&count), sizeof(int));
		int stringSize;
		string tmp;
		for (int i = 0; i < count; i++)
		{
			stringSize = m_segmentationTableNames[i].size();
			f.write((char*)(&stringSize), sizeof(int));
			tmp = m_segmentationTableNames[i];
			f.write((char*)(&tmp[0]), stringSize);
		}
		f.write((char*)(&m_segmentsOpacity[0]), sizeof(float)*count);
		f.write((char*)(&m_s), sizeof(Vector3));
		f.write((char*)(&m_t), sizeof(Vector3));
		f.write((char*)(&m_isSkeletonBinded), sizeof(bool));
		if (m_isSkeletonBinded)
		{
			f.write((char*)(&m_weights00[0]), sizeof(float)*m_weights00.size());
			f.write((char*)(&m_weights01[0]), sizeof(float)*m_weights00.size());
			f.write((char*)(&m_bones02[0]), sizeof(uchar)*m_weights00.size());
		}
		m_skeleton.SaveBin(f);
		f.close();
	}
	else
	{
		throw std::exception("Can`t open file");
	}
}

void VoxelObject::LoadBin(string path)
{
	ifstream f;
	f.open(path.c_str(), ios::binary);
	if (f.is_open())
	{
		int count;
		f.read((char*)(&m_dim), sizeof(uint3));
		f.read((char*)(&count), sizeof(int));
		m_palette.resize(count);
		f.read((char*)(&m_palette[0]), sizeof(uchar4)*count);
		f.read((char*)(&count), sizeof(int));
		m_voxels.resize(count);
		f.read((char*)(&m_voxels[0]), sizeof(Voxel)*count);
		f.read((char*)(&count), sizeof(int));
		int stringSize;
		string tmp;
		for (int i = 0; i < count; i++)
		{
			f.read((char*)(&stringSize), sizeof(int));
			tmp.resize(stringSize);
			f.read((char*)(&tmp[0]), stringSize);
			m_segmentationTableNames.push_back(tmp);
		}
		m_segmentsOpacity.resize(count);
		f.read((char*)(&m_segmentsOpacity[0]), sizeof(float)*count);
		f.read((char*)(&m_s), sizeof(Vector3));
		f.read((char*)(&m_t), sizeof(Vector3));
		f.read((char*)(&m_isSkeletonBinded), sizeof(bool));
		if (m_isSkeletonBinded)
		{
			m_weights00.resize(m_voxels.size());
			m_weights01.resize(m_voxels.size());
			m_bones02.resize(m_voxels.size());
			f.read((char*)(&m_weights00[0]), sizeof(float)*m_weights00.size());
			f.read((char*)(&m_weights01[0]), sizeof(float)*m_weights00.size());
			f.read((char*)(&m_bones02[0]), sizeof(uchar)*m_weights00.size());
		}
		m_skeleton.LoadBin(f);
		f.close();
	}
	else
	{
		throw std::exception("Can`t open file");
	}
}

void VoxelObject::BlocksDecomposition(VoxelPipeline* voxPipeline, int blockSize, int overlap)
{
	int3 min;
	int3 max;
	if (m_isSkeletonBinded)
	{
		int maxSide = (m_dim.x > m_dim.y) ? m_dim.x : m_dim.y;
		maxSide = (maxSide > m_dim.z) ? maxSide : m_dim.z;
		min = { -2 * maxSide, -2 * maxSide, -2 * maxSide };
		max = { 2 * maxSide, 2 * maxSide, 2 * maxSide };
	}
	else
	{
		min = { 0, 0, 0 };
		max = m_dim;
	}
	m_blockSize = blockSize;
	vector<BlockInfo> blocksInfo;
	int3 dimBlocks = {ceil((float)(max.x-min.x)/blockSize), ceil((float)(max.y - min.y) / blockSize), ceil((float)(max.z - min.z) / blockSize) };
	for (int k = 0; k < dimBlocks.z; k++)
	{
		for (int j = 0; j < dimBlocks.y; j++)
		{
			for (int i = 0; i < dimBlocks.x; i++)
			{
				BlockInfo cur = { { min.x + (i + 1)*blockSize - 1, min.y + (j + 1)*blockSize - 1, min.z + (k + 1)*blockSize - 1 },{ min.x + i*blockSize, min.y + j*blockSize, min.z + k*blockSize } };
				blocksInfo.push_back(cur);
			}
		}
	}
	ComPtr<ID3D12Resource> voxelsRes = voxPipeline->RegisterVoxels(m_voxels);
	ComPtr<ID3D12Resource> blocksInfoRes = voxPipeline->RegisterBlocksInfo(blocksInfo);
	ComPtr<ID3D12Resource> weights00Res;
	ComPtr<ID3D12Resource> weights01Res;
	ComPtr<ID3D12Resource> bones02Res;
	if (m_isSkeletonBinded)
	{
		weights00Res = voxPipeline->RegisterBonesWeights(m_weights00, 0);
		weights01Res = voxPipeline->RegisterBonesWeights(m_weights01, 1);
		bones02Res = voxPipeline->RegisterAdditionalBonesIndices(m_bones02);
	}
	if (m_isSkeletonBinded)
	{
		voxPipeline->ComputePoseDetectBlocks(m_voxels.size(), m_dim, m_blockSize, dimBlocks, min, max, blocksInfo, blocksInfoRes, m_skeleton);
	}
	else
	{
		voxPipeline->ComputeDetectBlocks(m_voxels.size(), m_dim, m_blockSize, dimBlocks, min, max, blocksInfo, blocksInfoRes);
	}
	
	ComPtr<ID3D12Resource> blocksIndexesRes;
	voxPipeline->RegisterBlocks(overlap, dimBlocks, m_blockSize, min, blocksInfo, m_blocksRes, m_texturesRes, blocksIndexesRes, m_blocksPosInfo, m_blocksPriorInfo);
	m_blocksVis.resize(m_blocksPriorInfo.size());
	UpdateBlocksVis();

	m_blocksBufferView.BufferLocation = m_blocksRes->GetGPUVirtualAddress();
	m_blocksBufferView.StrideInBytes = sizeof(Vertex);
	m_blocksBufferView.SizeInBytes = sizeof(Block)*m_texturesRes.size();
	if (m_isSkeletonBinded)
	{
		voxPipeline->ComputePoseFillBlocks(m_voxels.size(), m_texturesRes.size(), m_dim, m_blockSize, dimBlocks, min, max, overlap, m_texturesRes, m_skeleton);
	}
	else
	{
		voxPipeline->ComputeFillBlocks(m_voxels.size(), m_texturesRes.size(), m_dim, m_blockSize, dimBlocks, min, max, overlap, m_texturesRes);
	}
}

vector<BlockPriorityInfo> VoxelObject::CalculatePriorities(Vector3 cameraPos)
{
	for (int i = 0; i < m_blocksPosInfo.size(); i++)
	{
		m_blocksPosInfo[i].distance = Vector3::DistanceSquared(m_blocksPosInfo[i].position, cameraPos);
	}
	auto first = min_element(m_blocksPosInfo.begin(), m_blocksPosInfo.end(), [](BlockPositionInfo &a, BlockPositionInfo &b) { return (a.distance < b.distance); });
	for (int i = 0; i < m_blocksPriorInfo.size(); i++)
	{
		m_blocksPriorInfo[i].priority = abs(m_blocksPriorInfo[i].block3dIndex.x - first->block3dIndex.x) + abs(m_blocksPriorInfo[i].block3dIndex.y - first->block3dIndex.y) + abs(m_blocksPriorInfo[i].block3dIndex.z - first->block3dIndex.z);
	}
	sort(m_blocksPriorInfo.begin(), m_blocksPriorInfo.end(), [](BlockPriorityInfo &a, BlockPriorityInfo &b) { return (a.priority < b.priority); });
	return m_blocksPriorInfo;
}

D3D12_VERTEX_BUFFER_VIEW VoxelObject::GetBlocksVBV()
{
	return m_blocksBufferView;
}

float VoxelObject::GetVoxelSize()
{
	float maxSide = (m_dim.x > m_dim.y) ? m_dim.x : m_dim.y;
	maxSide = (maxSide > m_dim.z) ? maxSide : m_dim.z;
	return (1.0f / maxSide);
}

vector<string> VoxelObject::GetSegmentsNames()
{
	return m_segmentationTableNames;
}

void VoxelObject::SetSegmentsOpacity(VoxelPipeline * voxPipeline, vector<float> &segmentsOpacity)
{
	m_segmentsOpacity = segmentsOpacity;
	voxPipeline->SetSegmentsOpacity(m_segmentsOpacity, m_segmentsOpacityRes);
}

vector<float> VoxelObject::GetSegmentsOpacity()
{
	return m_segmentsOpacity;
}

int VoxelObject::GetBonesCount()
{
	return m_skeleton.GetBonesCount();
}

int VoxelObject::PickBone(float x, float y, float eps, Matrix viewProj)
{
	return m_skeleton.PickBone(x, y, eps, (GetWorld() * viewProj));
}

void VoxelObject::SetSkeletonMatricesForDraw(Matrix viewProj, Matrix* matricesForDraw)
{
	Matrix worldViewProj = GetWorld() * viewProj;
	m_skeleton.SetMatricesForDraw(worldViewProj, matricesForDraw);
}

int VoxelObject::AddBone(int selectedIndex)
{
	return m_skeleton.AddBone(selectedIndex);
}

void VoxelObject::SetBoneLength(int selectedIndex, float length)
{
	m_skeleton.SetBoneLength(selectedIndex, length);
}

void VoxelObject::TranslateSkeleton(Vector3 dt)
{
	m_skeleton.Translate(dt);
}

void VoxelObject::RotateBone(Vector3 dr, int index)
{
	m_skeleton.RotateBone(dr, index);
}

void VoxelObject::DeleteBone(int index)
{
	m_skeleton.DeleteBone(index);
}

int VoxelObject::CopyBones(int index)
{
	return m_skeleton.CopyBones(index);
}

void VoxelObject::MirrorRotation(int index, Vector3 axis)
{
	m_skeleton.MirrorRotation(index, axis);
}


void VoxelObject::BindBones(int borderSegment)
{
	m_weights00.resize(m_voxels.size());
	m_weights01.resize(m_voxels.size());
	m_bones02.resize(m_voxels.size());
	m_skeleton.SetOffsets();
	vector<pair<Vector3, Vector3>> bonesPoints = m_skeleton.GetBonesPoints();
	CUDACalculateWeights(m_voxels, m_dim, m_weights00, m_weights01, m_bones02, borderSegment, bonesPoints);
	m_isSkeletonBinded = true;
}

bool VoxelObject::IsSkeletonBinded()
{
	return m_isSkeletonBinded;
}

void VoxelObject::UnbindSkeleton()
{
	m_isSkeletonBinded = false;
	m_weights00.clear();
	m_weights00.shrink_to_fit();
	m_weights01.clear();
	m_weights01.shrink_to_fit();
	m_bones02.clear();
	m_bones02.shrink_to_fit();
}

void VoxelObject::SetBonesThickness(float thickness)
{
	m_skeleton.SetBonesThickness(thickness);
}

void VoxelObject::UpdateBlocksVis()
{
	int act = 0;
	for (int i = 0; i < m_blocksVis.size(); i++)
	{
		act = rand() % 3;
		m_blocksVis[i] = act;
	}
}

vector<int> VoxelObject::GetBlocksVis()
{
	return m_blocksVis;
}

