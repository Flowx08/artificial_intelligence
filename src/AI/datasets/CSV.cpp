////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CSV.hpp"
#include <fstream>
#include <sstream>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
		
	////////////////////////////////////////////////////////////
	std::vector< std::vector<std::string> > loadCSV(std::string csv_filepath)
	{
		//Create output vector
		std::vector< std::vector<std::string> > data;

		//Open file
		std::ifstream file(csv_filepath);

		//Check for errors
		if (!file) {
			printf("Error, can't load CSV file of path %s\n", csv_filepath.c_str());
			return data;
		}

		//Read file line by line
		std::string line;
		std::string cell;
		while (std::getline(file, line)) {
			//Add dimnesion to data
			data.push_back(std::vector<std::string>());

			//Read all the cells in the line
			std::stringstream lineStream(line);
			while (std::getline(lineStream, cell, ',')) {

				//Store the content of every cells
				data.back().push_back(cell);
			}
		}

		return data;
	}
	
} /* namespace ai */
