#ifndef CLOCK_H
#define CLOCK_H

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <sys/time.h>

//Vars
static struct timeval tv1, tv2;
static double total_elapsed = 0;

////////////////////////////////////////////////////////////
void measure_start()
{
	gettimeofday(&tv1, NULL);
}

////////////////////////////////////////////////////////////
double measure_stop()
{
	gettimeofday(&tv2, NULL);
	return (double) (tv2.tv_usec - tv1.tv_usec) / 1000000.f + (double) (tv2.tv_sec - tv1.tv_sec);
}

#endif /* end of include guard: CLOCK_H */
