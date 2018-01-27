#include "stdafx.h"
#include "VoxelObject.h"


VoxelObject::VoxelObject(VoxelPipeline* voxPipeline)
{
	/*
	m_dim.x = 608;
	m_dim.y = 346;
	m_dim.z = 1702;
	float maxSide = max(max(m_dim.x, m_dim.y), m_dim.z);
	m_s = Vector3(1 / maxSide);
	//m_t = -m_dim / (2.0f*maxSide);
	//m_size = m_dim / maxSide;
	m_startPos = -m_size/2.0f;
	m_blockDim = 32;
	//m_blockDim = 64;
	m_blockSize = m_blockDim/maxSide;
	int x_blockDim = ceil(m_dim.x / m_blockDim);
	int y_blockDim = ceil(m_dim.y / m_blockDim);
	int z_blockDim = ceil(m_dim.z / m_blockDim);
	Vector3 blockIndex(0, 0, 0);
	for (blockIndex.x=0; blockIndex.x < x_blockDim; blockIndex.x++)
	{
		for (blockIndex.y=0; blockIndex.y < y_blockDim; blockIndex.y++)
		{
			for (blockIndex.z=0; blockIndex.z < z_blockDim; blockIndex.z++)
			{
				m_blocks.emplace_back(m_dim, blockIndex, m_startPos, m_blockDim, m_blockSize);
				m_blocksInfo.emplace_back(blockIndex, m_blockDim);
			}
		}
	}
	m_blocksRes = voxPipeline->CreateBlocksViews(&m_blocks[0], m_blocks.size());
	m_blocksBufferView.BufferLocation = m_blocksRes->GetGPUVirtualAddress();
	m_blocksBufferView.StrideInBytes = sizeof(Vertex);
	m_blocksBufferView.SizeInBytes = sizeof(Vertex)*sizeof(Block)*m_blocks.size();
	*/
}

VoxelObject::VoxelObject(string path, LOADING_MODE loadingMode, VoxelPipeline * voxPipeline)
{
	if (loadingMode == LOADING_MODE::LOADING_MODE_SLICES)
	{
		CreateFromSlices(path, voxPipeline);
	}
}


VoxelObject::~VoxelObject()
{
}

ID3D12Resource * VoxelObject::GetBlocksRes()
{
	return m_blocksRes.Get();
}

D3D12_VERTEX_BUFFER_VIEW VoxelObject::GetBlocksVertexBufferView()
{
	return m_blocksBufferView;
}

int VoxelObject::GetBlocksCount()
{
	return m_blocks.size();
}

void VoxelObject::CreateFromSlices(string path, VoxelPipeline * voxPipeline)
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
				//segmentationTableNames.push_back(line.substr(0, sep - 1));
				line = line.substr(sep, line.length());
				sS.str(line);
				sS >> word;
				//Cur.Color[0] = atoi(Word.c_str());
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
			//throw std::exception("Incorrect segmentation transfer");
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
			segmentationNamesFile.close();
		}
		else
		{
			throw std::exception("Can`t open segmentation names file");
		}

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
		m_dim.x = img.size().width;
		m_dim.y = img.size().height;
		m_dim.z = 1;
		while (FindNextFileA(hA, &fA))
		{
			m_dim.z++;
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
		thrust::device_vector<int> dVoxelsSlice(10);
		//thrust::device_vector<RGBVoxel> dVoxelsSlice;
		//dVoxelsSlice.resize(m_dim.x*m_dim.y);
		thrust::device_vector<RGBVoxel> dVoxels;
		//thrust::device_vector<RGBVoxel> dVoxels;
		RGBVoxel* gVoxels;
		int* gCount;
		int hCount;
		cudaMalloc((void**)&gDataA, sizeof(unsigned char)*(m_dim.x*m_dim.y) * 3);
		cudaMalloc((void**)&gDataS, sizeof(unsigned char)*(m_dim.x*m_dim.y) * 3);
		cudaMalloc((void**)&gSegmentTable, sizeof(SegmentData)*(segmentationTable.size()));
		cudaMemcpy(gSegmentTable, &segmentationTable[0], sizeof(SegmentData)*(segmentationTable.size()), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&gSegmentationTransfer, sizeof(SegmentData)*(segmentationTransfer.size()));
		cudaMemcpy(gSegmentationTransfer, &segmentationTransfer[0], sizeof(SegmentData)*(segmentationTransfer.size()), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&gVoxels, sizeof(RGBVoxel)*(m_dim.x*m_dim.y));
		cudaMalloc((void**)&gCount, sizeof(int));
		int eps = 2;
		int Time = clock();
		while (FindNextFileA(hA, &fA) && FindNextFileA(hS, &fS))
		{
			Mat mA = imread(anatomicalFolder + fA.cFileName);
			Mat mS = imread(segmentedFolder + fS.cFileName);
			cudaMemset(gCount, 0, sizeof(int));
			hDataA = mA.data;
			hDataS = mS.data;
			cudaMemcpy(gDataA, hDataA, sizeof(unsigned char)*(m_dim.x*m_dim.y) * 3, cudaMemcpyHostToDevice);
			cudaMemcpy(gDataS, hDataS, sizeof(unsigned char)*(m_dim.x*m_dim.y) * 3, cudaMemcpyHostToDevice);
			GetVoxelsAnatomicalSegmentation(gDataA, gDataS, gSegmentTable, segmentationTable.size(), gSegmentationTransfer, eps,gVoxels, m_dim.x, m_dim.y, curDepth,depthMulptiplier, gCount);
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
				//sort(Voxels.begin() + CurSize, Voxels.end(), CompareVoxels);
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
	else
	{
		throw std::exception("Can`t open input file");
	}
}

