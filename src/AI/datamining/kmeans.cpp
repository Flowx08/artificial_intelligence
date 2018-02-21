////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "kmeans.hpp"
#include <stdlib.h>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai 
{

	////////////////////////////////////////////////////////////
	kmeans::kmeans(int k, int dimensions, float init_mean, float init_dev)
	{
		//Set number of clusters
		_k = k;
		_dimensions = dimensions;
		_init_mean = init_mean;
		_init_dev = init_dev;

		//Randomly set centroids
		_centroids = std::vector< vec >(k);
		for (int i = 0; i < k; i++) {
			_centroids[i] = vec(dimensions);
			for (int j = 0; j < _dimensions; j++)
				_centroids[i][j] = init_mean - init_dev + ((double)rand() / RAND_MAX) * 2 * init_dev;
		}
	}

	////////////////////////////////////////////////////////////
	void kmeans::fit(const std::vector< vec > &data, int max_iterations)
	{
		//vec-cluster buffers
		std::vector< int > datavec_cluster(data.size());
		std::vector< int > datavec_cluster_old(data.size());
		bool finished = false;
		bool force_iteration = false;

		//Limit the number of iterations
		for (int q = 0; q < max_iterations; q++) {

			datavec_cluster_old = datavec_cluster;

			//Find best cluster for each datavec
			for (int d = 0; d < (int)data.size(); d++)
				datavec_cluster[d] = getcluster(data[d]);

			//Check if we have converget to optimal solution
			if (q != 0 && !force_iteration) {
				finished = true;
				for (int d = 0; d < (int)data.size(); d++) {
					if (datavec_cluster[d] != datavec_cluster_old[d]) {
						finished = false;
						break;
					}
				}

				//We have converged!
				if (finished) {
					_fit_iterations = q;
					return;
				}
			}
			force_iteration = false;

			//For each cluster, update centroid position
			int cluster_size = 0;
			vec newcentroid(_dimensions);
			for (int i = 0; i < _k; i ++) {

				//Reset new centroid position
				std::fill(newcentroid.begin(), newcentroid.end(), 0);

				//For each datavec
				for (int d = 0; d < (int)data.size(); d++) {

					//Check if it's in this cluster
					if (datavec_cluster[d] != i) continue;
					else cluster_size++; //New vec in this cluster!

					//Summatory
					for (int j = 0; j < _dimensions; j++) {
						newcentroid[j] += data[d][j];
					}
				}

				//Calculate mean position of vecs in cluster
				if (cluster_size != 0) 
				{
					for (int j = 0; j < _dimensions; j++)
						newcentroid[j] /= (double)cluster_size;
				}
				else //Ops, empty cluster... we must deal with it!
				{
					for (int j = 0; j < _dimensions; j++)
						newcentroid[j] = _init_mean - _init_dev + ((double)rand() / RAND_MAX) * 2 * _init_dev;
					force_iteration = true;
				}

				//Update centroid[i] with newcentroid position
				_centroids[i] = newcentroid;
			}
		}

		_fit_iterations = max_iterations;
	}
	
	////////////////////////////////////////////////////////////
	void kmeans::fit_continuous(const std::vector< vec > &data, float learningrate, int iterations)
	{
		//vec-cluster buffers
		std::vector< int > datavec_cluster(data.size());
		
		//Limit the number of iterations
		for (int q = 0; q < iterations; q++) {
			
			//Find best cluster for each datavec
			for (int d = 0; d < (int)data.size(); d++)
				datavec_cluster[d] = getcluster(data[d]);

			//For each cluster, update centroid position
			int cluster_size = 0;
			vec newcentroid(_dimensions);
			for (int i = 0; i < _k; i ++) {

				//Reset new centroid position
				std::fill(newcentroid.begin(), newcentroid.end(), 0);

				//For each datavec
				for (int d = 0; d < (int)data.size(); d++) {

					//Check if it's in this cluster
					if (datavec_cluster[d] != i) continue;
					else cluster_size++; //New vec in this cluster!

					//Summatory
					for (int j = 0; j < _dimensions; j++) {
						newcentroid[j] += data[d][j];
					}
				}

				//Calculate mean position of vecs in cluster
				if (cluster_size != 0) 
				{
					for (int j = 0; j < _dimensions; j++)
						newcentroid[j] /= (double)cluster_size;
				}
				else //Ops, empty cluster... we must deal with it!
				{
					//Probably we only have to wait...
				}

				//Update centroid[i] with newcentroid position
				_centroids[i] = newcentroid;
			}
		}
	}

	////////////////////////////////////////////////////////////
	int kmeans::getiterations()
	{
		return _fit_iterations;
	}

	////////////////////////////////////////////////////////////
	int kmeans::getcluster(const vec &data)
	{
		double dist;
		double best_dist = 0xFFFFFF;
		int best_cluster = 0;

		//For each cluster
		for (int i = 0; i < _k; i++) {

			//Reset distance
			dist = 0.0;

			//Calculate euclidean distance between datavec and centroid[i]
			for (int j = 0; j < _dimensions; j++)
				dist += pow(data[j] - _centroids[i][j], 2);
			dist = sqrt(dist);

			//Update best cluster
			if (dist < best_dist) {
				best_dist = dist;
				best_cluster = i;
			}
		}

		return best_cluster;
	}
	
	////////////////////////////////////////////////////////////
	const std::vector< vec > &kmeans::getcentroids()
	{
		return _centroids;
	}
	
	////////////////////////////////////////////////////////////
	void kmeans::setcentroid(int id, const vec& centroid)
	{
		_centroids[id] = centroid;
	}

} //namespace ai
