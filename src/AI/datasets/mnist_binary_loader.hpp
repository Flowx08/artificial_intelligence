#ifndef MNIST_BINARY_LOADER_HPP
#define MNIST_BINARY_LOADER_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>

class mnist_binary_loader
{
public:
	mnist_binary_loader(const std::string train_images_path, const std::string test_images_path,
											const std::string train_labels_path, const std::string test_labels_path);
	const std::vector< std::vector< unsigned char > >& get_train_images() const;
	const std::vector< std::vector< unsigned char > >& get_test_images() const;
	const std::vector< unsigned char >& get_train_labels() const;
	const std::vector< unsigned char >& get_test_labels() const;

private:
	void load_images(std::ifstream& file, std::vector< std::vector<unsigned char> >& images);
	void load_labels(std::ifstream& file, std::vector<unsigned char>& labels);

	//Data
	std::vector< std::vector< unsigned char > > _train_images;
	std::vector< std::vector< unsigned char > > _test_images;
	std::vector< unsigned char > _train_labels;
	std::vector< unsigned char > _test_labels;
};

#endif /* end of include guard: MNIST_BINARY_LOADER_HPP */

