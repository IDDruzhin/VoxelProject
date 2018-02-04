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
			segmentationNamesFile.close();
		}
		else
		{
			throw std::exception("Can`t open segmentation names file");
		}

		int eps = 2;
		CUDACreateFromSlices(anatomicalFolder, segmentedFolder, segmentationTable, segmentationTransfer, depthMulptiplier, eps, m_dim, m_voxels, m_palette);
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
		f.close();
	}
	else
	{
		throw std::exception("Can`t open file");
	}
}

