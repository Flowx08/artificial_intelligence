////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "mnist_binary_loader.hpp"
#include <fstream>
#include <assert.h>
#include <stdio.h>

////////////////////////////////////////////////////////////
mnist_binary_loader::mnist_binary_loader(const std::string train_images_path, const std::string test_images_path,
		const std::string train_labels_path, const std::string test_labels_path)
{
	std::ifstream file(train_images_path, std::ios::binary);
	assert(file && "Unable to open filepath while loading mnist dataset");
	load_images(file, _train_images);
	file = std::ifstream(test_images_path, std::ios::binary);
	assert(file && "Unable to open filepath while loading mnist dataset");
	load_images(file, _test_images);
	file = std::ifstream(train_labels_path, std::ios::binary);
	assert(file && "Unable to open filepath while loading mnist dataset");
	load_labels(file, _train_labels);
	file = std::ifstream(test_labels_path, std::ios::binary);
	assert(file && "Unable to open filepath while loading mnist dataset");
	load_labels(file, _test_labels);
}

////////////////////////////////////////////////////////////
int big_endian_to_small(int big_endian)
{
	return		((big_endian >> 24) &0xff) |				// move byte 3 to byte 0
            ((big_endian << 8)  &0xff0000) |		// move byte 1 to byte 2
            ((big_endian >> 8)  &0xff00) |			// move byte 2 to byte 1
            ((big_endian << 24) &0xff000000);		// byte 0 to byte 3
}

////////////////////////////////////////////////////////////
void mnist_binary_loader::load_images(std::ifstream& file, std::vector< std::vector<unsigned char> >& images)
{
	//Check first 32 bit magic number
	int MSB_check;
	file.read(reinterpret_cast<char*>(&MSB_check), sizeof(int));
	MSB_check = big_endian_to_small(MSB_check);
	assert(MSB_check == 2051);
	
	//Get number of images
	int images_count;
	file.read(reinterpret_cast<char*>(&images_count), sizeof(int));
	images_count = big_endian_to_small(images_count);
	
	//Get images with and height
	int images_width, images_height;
	file.read(reinterpret_cast<char*>(&images_width), sizeof(int));
	file.read(reinterpret_cast<char*>(&images_height), sizeof(int));
	images_width = big_endian_to_small(images_width);
	images_height = big_endian_to_small(images_height);
	assert(images_width == 28 && images_height == 28);

	//Allocate memory for all the images
	images = std::vector< std::vector< unsigned char> >(images_count);
	for (int i = 0; i < images_count; i++)
		images[i] = std::vector< unsigned char >(images_width * images_height);	
	
	//Read data for each image
	for (int i = 0; i < images_count; i++)
		file.read(reinterpret_cast<char*>(&images[i][0]), sizeof(unsigned char) * images_width * images_height);
}

////////////////////////////////////////////////////////////
void mnist_binary_loader::load_labels(std::ifstream& file, std::vector<unsigned char>& labels)
{
	//Check first 32 bit magic number
	int MSB_check;
	file.read(reinterpret_cast<char*>(&MSB_check), sizeof(int));
	MSB_check = big_endian_to_small(MSB_check);
	assert(MSB_check == 2049);
	
	//Get number of images
	int images_count;
	file.read(reinterpret_cast<char*>(&images_count), sizeof(int));
	images_count = big_endian_to_small(images_count);
	
	//Allocate memory for all labels
	labels = std::vector< unsigned char >(images_count);

	//Read all labels values into out buffer
	file.read(reinterpret_cast<char*>(&labels[0]), sizeof(unsigned char) * images_count);
}

////////////////////////////////////////////////////////////
const std::vector< std::vector< unsigned char > >& mnist_binary_loader::get_train_images() const
{
	return _train_images;	
}

////////////////////////////////////////////////////////////
const std::vector< std::vector< unsigned char > >& mnist_binary_loader::get_test_images() const
{
	return _test_images;	
}

////////////////////////////////////////////////////////////
const std::vector< unsigned char >& mnist_binary_loader::get_train_labels() const
{
	return _train_labels;	
}

////////////////////////////////////////////////////////////
const std::vector< unsigned char >& mnist_binary_loader::get_test_labels() const
{
	return _test_labels;	
}

