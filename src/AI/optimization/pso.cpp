////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "pso.hpp"
#include <stdlib.h>

////////////////////////////////////////////////////////////
///	AI
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	double randuniform()
	{
		return (double)rand() / RAND_MAX;
	}	

	////////////////////////////////////////////////////////////
	pso::pso(int particles_count, int params_count, double(*eval)(std::vector<double>&), double learning_rate_local, double learning_rate_global)
	{
		//Set population parameters
		_particles_count = particles_count;
		_params_count = params_count;
		_learning_rate_local = learning_rate_local;
		_learning_rate_global = learning_rate_global;

		//Set evaluation function
		_eval = eval;

		//Set global best score to a very low number and create random best positon
		_global_bestscore = -0xFFFFFF;
		_global_bestpos = std::vector<double>(_params_count);
		for (int i = 0; i < _params_count; i++) 
			_global_bestpos[i] = randuniform();

		//Randomly generate particles
		for (int i = 0; i < _particles_count; i++) {
			particle p;

			//Create random velicity vector
			p.v = std::vector<double>(_params_count);
			for (int i = 0; i < _params_count; i++) 
				p.v[i] = randuniform();

			//Place the particle in a random position
			p.pos = std::vector<double>(_params_count);
			for (int i = 0; i < _params_count; i++) 
				p.pos[i] = randuniform(); 

			//Initialize best position
			p.bestpos = p.pos;
			p.bestscore = -0xFFFFFF;

			//Add particle
			_particles.push_back(p);
		}
	}

	////////////////////////////////////////////////////////////
	void pso::run(unsigned long steps)
	{
		double score;
		for (unsigned int s = 0; s < steps; s++) {

			//Test each particle
			for (int p = 0; p < _particles_count; p++) {

				//Evaluate particle position
				score = _eval(_particles[p].pos);

				//Update best particle score
				if (score > _particles[p].bestscore) {
					_particles[p].bestscore = score;
					_particles[p].bestpos = _particles[p].pos;
				}

				//Update global best
				if (score > _global_bestscore) {
					_global_bestscore = score;
					_global_bestpos = _particles[p].pos;
				}
			}

			//Move each particle
			for (int p = 0; p < _particles_count; p++) {
				for (int j = 0; j < _params_count; j++) {
					//Accelerate in the direction of the particle best pos and the global best pos
					//with randomized velocities
					_particles[p].v[j] += _learning_rate_local * randuniform() * (_particles[p].bestpos[j] - _particles[p].pos[j]) + 
						_learning_rate_global * randuniform() * (_global_bestpos[j] - _particles[p].pos[j]);
					if (_particles[p].v[j] > 20) _particles[p].v[j] = 20;
					if (_particles[p].v[j] < -20) _particles[p].v[j] = -20;
					_particles[p].pos[j] += _particles[p].v[j];
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	std::vector<double> pso::getBestParams()
	{
		return _global_bestpos;
	}

	////////////////////////////////////////////////////////////
	double pso::getBestScore()
	{
		return _global_bestscore;
	}

	////////////////////////////////////////////////////////////
	void pso::setLocalLearningrate(double val)
	{
		_learning_rate_local = val;
	}

	////////////////////////////////////////////////////////////
	void pso::setGlobalLearningrate(double val)
	{
		_learning_rate_global = val;
	}

	////////////////////////////////////////////////////////////
	double pso::getLocalLearningrate()
	{
		return _learning_rate_local;
	}

	////////////////////////////////////////////////////////////
	double pso::getGlobalLearningrate()
	{
		return _learning_rate_global;
	}

	////////////////////////////////////////////////////////////
	const std::vector<particle>& pso::getParticles()
	{
		return _particles;
	}

} //namespace ai

