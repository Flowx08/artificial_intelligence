#ifndef IODATA_HPP
#define IODATA_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <fstream>
#include <memory>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	class IOData
	{
		public:
			
			IOData(std::string name);
			IOData(std::string name, const char* data, unsigned int size);
			IOData(std::string name, const int data);
			IOData(std::string name, const std::string data);
			IOData(std::string name, const float* data, unsigned int size);
			IOData(std::string name, const float data);
			IOData(std::string name, const bool data);
			IOData(std::ifstream& filestream);
			~IOData();	
			
			bool loadFromFile(std::string filename);
			void loadFromStream(std::ifstream& filestream);
			bool writeToFile(std::string filename);
			void writeToStream(std::ofstream& filestream);
			
			void get(char* data) const;
			void get(int& data) const;
			void get(std::string& data) const;
			void get(float* data) const;
			void get(float& data) const;
			void get(bool& data) const;
			void setName(std::string name);
			const std::string getName() const;

			void pushNode(std::string name);
			void pushNode(std::string name, const char* data, unsigned int size);
			void pushNode(std::string name, const int data);
			void pushNode(std::string name, const std::string data);
			void pushNode(std::string name, const float* data, unsigned int size);
			void pushNode(std::string name, const float data);
			void pushNode(std::string name, const bool data);
			void removeNode(std::string label);
			IOData* findNode(std::string label);
			int findNodeIndex(std::string label);
			bool existsNode(std::string label);
			const std::vector<IOData> getSubNodes() const;

		private:
			void setData(const char* data, unsigned int size);
			std::string readString(std::ifstream &filestream);
			void writeString(std::ofstream &filestream, const std::string s);
			
			std::string _name;
			int _data_size;
			std::shared_ptr<char> _data;
			std::vector<IOData> _subnodes;
	};

} /* namespace ai */

#endif /* end of include guard: IODATA_HPP */

