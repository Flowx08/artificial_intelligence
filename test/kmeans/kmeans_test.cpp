////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../../src/AI/datamining/kmeans.hpp"
#include <time.h>
#include <random>

//#define STANDARD
#define CONTINUOUS

#if defined(STANDARD)

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));

	ai::kmeans test(2, 2);
	std::vector< ai::vec > vecs(6);
	vecs[0] = ai::vec(2);
	vecs[0][0] = 1.0;
	vecs[0][1] = 1.0;
	
	vecs[1] = ai::vec(2);
	vecs[1][0] = 2.0;
	vecs[1][1] = 1.0;
	
	vecs[2] = ai::vec(2);
	vecs[2][0] = 1.0;
	vecs[2][1] = 2.0;
	
	vecs[3] = ai::vec(2);
	vecs[3][0] = 3+ 1.0;
	vecs[3][1] = 3+ 1.0;
	
	vecs[4] = ai::vec(2);
	vecs[4][0] = 3+ 2.0;
	vecs[4][1] = 3+ 1.0;
	
	vecs[5] = ai::vec(2);
	vecs[5][0] = 3+ 1.0;
	vecs[5][1] = 3+ 2.0;
	
	test.fit(vecs, 100);
	printf("Fitted in:%d\n", test.getiterations());

	for (int i = 0; i < (int)vecs.size(); i++) {
		printf("Point:%d Cluster:%d\n", i, test.getcluster(vecs[i]));
	}

	return 0;
}

#elif defined(CONTINUOUS)

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));

	ai::kmeans test(2, 2);
	std::vector< ai::vec > vecs(6);
	vecs[0] = ai::vec(2);
	vecs[0][0] = 1.0;
	vecs[0][1] = 1.0;
	
	vecs[1] = ai::vec(2);
	vecs[1][0] = 2.0;
	vecs[1][1] = 1.0;
	
	vecs[2] = ai::vec(2);
	vecs[2][0] = 1.0;
	vecs[2][1] = 2.0;
	
	vecs[3] = ai::vec(2);
	vecs[3][0] = 3+ 1.0;
	vecs[3][1] = 3+ 1.0;
	
	vecs[4] = ai::vec(2);
	vecs[4][0] = 3+ 2.0;
	vecs[4][1] = 3+ 1.0;
	
	vecs[5] = ai::vec(2);
	vecs[5][0] = 3+ 1.0;
	vecs[5][1] = 3+ 2.0;
	
	const int cicles = 3;
	for (int i = 0; i < cicles; i++) {
		for (int k = 0; k < (int)test.getcentroids().size(); k++) {
			printf("Centroid: %d X:%f Y:%f\n", k, test.getcentroids()[k][0], test.getcentroids()[k][1]);
		}
		std::vector< ai::vec > ps(1);
		ps[0] = vecs[rand() % (int)vecs.size()];
		test.fit_continuous(vecs, 0.1, 1);
	}
	printf("Fitted in:%d\n", cicles);

	for (int i = 0; i < (int)vecs.size(); i++) {
		printf("Point:%d Cluster:%d\n", i, test.getcluster(vecs[i]));
	}

	return 0;
}

#endif
