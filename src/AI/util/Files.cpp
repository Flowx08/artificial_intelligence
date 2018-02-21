////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Files.hpp"
#include <fstream>
#if defined(_WIN32) || defined(_WIN64)
#include "../util/dirent_win.h"
#else
#include <dirent.h>
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	///	FILES
	////////////////////////////////////////////////////////////
	namespace files
	{
		
		////////////////////////////////////////////////////////////
		std::vector<std::string> listdir(std::string folderpath)
		{
			//Open directory
			DIR *dir_count;
			struct dirent *ent_count;
			dir_count = opendir(folderpath.c_str());
			
			//Check for errrors
			if (dir_count == NULL)
				printf("Error, can't open directory of path %s\n", folderpath.c_str());

			//Store all file names
			std::vector<std::string> list_files;
			while ((ent_count = readdir(dir_count)) != NULL) {
				if (std::string(ent_count->d_name) == "." || std::string(ent_count->d_name) == "..") continue;
				list_files.push_back(std::string(ent_count->d_name));
			}

			return list_files;
		}
		
		////////////////////////////////////////////////////////////
		std::string get_extension(std::string path)
		{
			size_t  lastdot = path.find_last_of(".");
			if (lastdot == std::string::npos) return "";
			return path.substr(lastdot, path.size());
		}
		
		////////////////////////////////////////////////////////////
		std::string remove_extension(std::string path)
		{
			size_t  lastdot = path.find_last_of(".");
			if (lastdot == std::string::npos) return path;
			return path.substr(0, lastdot);
		}
		
		////////////////////////////////////////////////////////////
		bool exists(std::string filepath)
		{
			std::ifstream file(filepath);
			if (file) return true;
			return false;
		}

	} /* namespace files */

} /* namespace ai */
