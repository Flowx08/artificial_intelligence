////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "IOData.hpp"
#include "ensure.hpp"
#include <stdio.h>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name, const char* data, unsigned int size)
	{
		_name = name;	
		setData(data, size);
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name, const int data)
	{
		_name = name;
		int size = sizeof(int);
		const char* bytes = reinterpret_cast<const char*>(&data);
		setData(bytes, size);
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name, const std::string data)
	{
		_name = name;
		int size = data.size();
		setData(&data[0], size);
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name, const float* data, unsigned int size)
	{
		_name = name;
		const char* bytes = reinterpret_cast<const char*>(data);
		setData(bytes, sizeof(float) * size);
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name, const float data)
	{
		_name = name;
		int size = sizeof(float);
		const char* bytes = reinterpret_cast<const char*>(&data);
		setData(bytes, size);
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name, const bool data)
	{
		_name = name;
		int size = sizeof(bool);
		const char* bytes = reinterpret_cast<const char*>(&data);
		setData(bytes, size);
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::string name)
	{
		_name = name;
		_data_size = 0;
	}
	
	////////////////////////////////////////////////////////////
	IOData::IOData(std::ifstream& filestream)
	{
		loadFromStream(filestream);
	}

	////////////////////////////////////////////////////////////
	IOData::~IOData() {}

	
	////////////////////////////////////////////////////////////
	bool IOData::loadFromFile(std::string filename)
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file) {
			printf("Error in IOData, can't load file %s\n", filename.c_str());
			return false;
		}
		loadFromStream(file);
		return true;
	}
	
	////////////////////////////////////////////////////////////
	void IOData::loadFromStream(std::ifstream& filestream)
	{
		_name = readString(filestream);
		filestream.read(reinterpret_cast<char*>(&_data_size), sizeof(_data_size));
		_data = std::shared_ptr<char>(new char[_data_size]);
		filestream.read(reinterpret_cast<char*>(&_data.get()[0]), sizeof(char) * _data_size);
		int subnodes_size;
		filestream.read(reinterpret_cast<char*>(&subnodes_size), sizeof(subnodes_size));
		for (int i = 0; i < subnodes_size; i++)
			_subnodes.push_back(IOData(filestream));	
	}

	////////////////////////////////////////////////////////////
	bool IOData::writeToFile(std::string filename)
	{
		std::ofstream file(filename, std::ios::binary);
		if (!file) {
			printf("Error in IOData, can't save data to file %s\n", filename.c_str());
			return false;
		}
		writeToStream(file);
		return true;
	}

	////////////////////////////////////////////////////////////
	void IOData::writeToStream(std::ofstream& filestream)
	{
		writeString(filestream, _name);
		filestream.write(reinterpret_cast<char*>(&_data_size), sizeof(_data_size));
		filestream.write(reinterpret_cast<char*>(&_data.get()[0]), sizeof(char) * _data_size);
		int subnodes_size = _subnodes.size();
		filestream.write(reinterpret_cast<char*>(&subnodes_size), sizeof(subnodes_size));
		for (int i = 0; i < (int)_subnodes.size(); i++)
			_subnodes[i].writeToStream(filestream);
	}
	
	////////////////////////////////////////////////////////////
	void IOData::get(char* data) const
	{
		ensure(_data_size > 0);
		char* rawdata = _data.get();
		for (int i = 0; i < _data_size; i++)
			data[i] = rawdata[i];
	}
	
	////////////////////////////////////////////////////////////
	void IOData::get(int& data) const
	{
		ensure_print(_data_size == sizeof(int), "%d != %d\n", _data_size, (int)sizeof(int));
		get(reinterpret_cast<char*>(&data));
	}
	
	////////////////////////////////////////////////////////////
	void IOData::get(std::string& data) const
	{
		ensure(_data_size > 0);
		data = std::string(_data_size, ' ');
		get(reinterpret_cast<char*>(&data[0]));
	}

	////////////////////////////////////////////////////////////
	void IOData::get(float* data) const
	{
		ensure(_data_size > 0);
		get(reinterpret_cast<char*>(data));
	}
	
	////////////////////////////////////////////////////////////
	void IOData::get(float& data) const
	{
		ensure_print(_data_size == sizeof(float), "%d != %d\n", _data_size, (int)sizeof(float));
		get(reinterpret_cast<char*>(&data));
	}
	
	////////////////////////////////////////////////////////////
	void IOData::get(bool& data) const
	{
		ensure_print(_data_size == sizeof(bool), "%d != %d\n", _data_size, (int)sizeof(bool));
		get(reinterpret_cast<char*>(&data));
	}

	////////////////////////////////////////////////////////////
	const std::string IOData::getName() const
	{
		return _name;
	}
	
	////////////////////////////////////////////////////////////
	void IOData::setName(std::string name)
	{
		_name = name;
	}
	
	////////////////////////////////////////////////////////////
	void IOData::removeNode(std::string label)
	{
		int id = findNodeIndex(label);
		_subnodes.erase(_subnodes.begin() + id);
	}
	
	////////////////////////////////////////////////////////////
	IOData* IOData::findNode(std::string label)
	{
		int id = findNodeIndex(label);
		if (id != -1) return &_subnodes[id];
		else return NULL;
	}
	
	////////////////////////////////////////////////////////////
	int IOData::findNodeIndex(std::string label)
	{
		for (int i = 0; i < (int)_subnodes.size(); i++)
			if (_subnodes[i].getName() == label) return i;
		return -1;
	}
	
	////////////////////////////////////////////////////////////
	bool IOData::existsNode(std::string label)
	{
		return (findNodeIndex(label) != -1);
	}
	
	////////////////////////////////////////////////////////////
	std::string IOData::readString(std::ifstream &filestream)
	{
		int string_size;
		filestream.read(reinterpret_cast<char*>(&string_size), sizeof(string_size));
		std::string s(string_size, ' ');
		filestream.read(&s[0], sizeof(char) * string_size);
		return s;
	}

	////////////////////////////////////////////////////////////
	void IOData::writeString(std::ofstream &filestream, const std::string s)
	{
		int string_size = s.size();
		filestream.write(reinterpret_cast<char*>(&string_size), sizeof(string_size));
		filestream.write(&s[0], sizeof(char) * s.size());
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name)
	{
		_subnodes.push_back(IOData(name));
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name, const char* data, unsigned int size)
	{
		_subnodes.push_back(IOData(name, data, size));	
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name, const int data)
	{
		_subnodes.push_back(IOData(name, data));	
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name, const std::string data)
	{
		_subnodes.push_back(IOData(name, data));	
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name, const float* data, unsigned int size)
	{
		_subnodes.push_back(IOData(name, data, size));	
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name, const float data)
	{
		_subnodes.push_back(IOData(name, data));	
	}
	
	////////////////////////////////////////////////////////////
	void IOData::pushNode(std::string name, const bool data)
	{
		_subnodes.push_back(IOData(name, data));	
	}

	////////////////////////////////////////////////////////////
	void IOData::setData(const char* data, unsigned int size)
	{
		_data_size = size;
		_data = std::shared_ptr<char>(new char[size]);
		char* rawdata = _data.get();
		for (unsigned int i = 0; i < size; i++)
			rawdata[i] = data[i];
	}
	
	////////////////////////////////////////////////////////////
	const std::vector<IOData> IOData::getSubNodes() const
	{
		return _subnodes;
	}

} /* namespace ai */
