#ifndef GA_HPP
#define GA_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <assert.h>

////////////////////////////////////////////////////////////
///	AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	///	Genetic Algorithms
	////////////////////////////////////////////////////////////
	template <typename T> class genetic_optimizer
	{
		public:

			////////////////////////////////////////////////////////////
			/// \brief	Initialize genetic_optimizer
			///
			////////////////////////////////////////////////////////////
			genetic_optimizer(std::vector<T> population_start, int round_count, double(*eval)(T&), T(*generate)(T&, T&), void(*destroy)(T&))
			{	
				//Store starting genes
				_population_data = population_start;

				//Get the population size
				_population_size = (int)population_start.size();

				//Create the fitness vector for storing fitness values
				_population_fitness = std::vector<double>(_population_size, 0.0);

				//Store parameters values
				_eval = eval;
				_generate = generate;
				_destroy = destroy;
				_round_count = round_count;

				//Best genome initialize
				_best_data_id = 0;

				//Set the best gene's fitness to a very low value
				_best_fitness = -0xFFFF;
			}

			////////////////////////////////////////////////////////////
			/// \brief	Get best genome data	
			///
			////////////////////////////////////////////////////////////
			T getBestGenomeData()
			{
				return _population_data[_best_data_id];
			}

			////////////////////////////////////////////////////////////
			/// \brief	Get best genome fitness value	
			///
			////////////////////////////////////////////////////////////
			double getBestGenomeFitness()
			{
				return _best_fitness;
			}

			////////////////////////////////////////////////////////////
			/// \brief Evolve the data for X steps using Y cores
			///
			////////////////////////////////////////////////////////////
			void run(unsigned long steps, unsigned int cores)
			{
				if (cores <= 1) _run_single(steps);
				else
				{
					//Create threads
					std::vector< std::thread > workers(cores);
					for (int i = 0; i < cores; i++)
						workers[i] = std::thread(&genetic_optimizer::_run_multiple, this, steps/cores);
					
					//Run in parallel!
					for (int i = 0; i < cores; i++)
						workers[i].join();
				}
			}

			////////////////////////////////////////////////////////////
			/// \brief	Get the whole population	
			///
			////////////////////////////////////////////////////////////
			const std::vector<T>& getPopulation()
			{
				return _population_data;
			}
		
		private:
			
			////////////////////////////////////////////////////////////
			/// \brief	Run function for one thread	
			///
			////////////////////////////////////////////////////////////
			void _run_single(int steps) 
			{
				for (int i = 0; i < steps; i++) {
					
					//Get a badgenome
					int victim = getBadGenome();
					assert(victim != _best_data_id);

					//Get father and mother id
					int father_id = getGoodGenome();
					do { father_id = getGoodGenome(); } while(father_id == victim);
					int mother_id = rand() % _population_size;
					do { mother_id = rand() % _population_size; } while (mother_id == victim);
					T& father = _population_data[father_id];
					T& mother = _population_data[mother_id];

					//Generate son from mother and father and evaluate him
					_destroy(_population_data[victim]);
					_population_data[victim] = _generate(mother, father);
					_population_fitness[victim] = _eval(_population_data[victim]);

					//Update best genome
					if (_population_fitness[victim] > _best_fitness) {
						_best_data_id = victim;
						_best_fitness = _population_fitness[victim];
					}
				}
			}
			
			////////////////////////////////////////////////////////////
			/// \brief	Run function for multiple threads	
			///
			////////////////////////////////////////////////////////////
			void _run_multiple(int steps) 
			{
				for (int i = 0; i < steps; i++) {

					//Get father and mother id
					g_lock.lock();
					T father = _population_data[getGoodGenome()];
					T mother = _population_data[rand() % _population_size];
					g_lock.unlock();

					//Generate son from mother and father and evaluate him
					T son_data = _generate(mother, father);
					double son_fitness = _eval(son_data);
					
					g_lock.lock();
					
					//Get a badgenome
					int victim = getBadGenome();
					_destroy(_population_data[victim]);
					_population_data[victim] = son_data;
					_population_fitness[victim] = son_fitness;

					//Update best genome
					if (_population_fitness[victim] > _best_fitness) {
						_best_data_id = victim;
						_best_fitness = _population_fitness[victim];
					}
					
					g_lock.unlock();
				}
				printf("Done\n");
			}

			////////////////////////////////////////////////////////////
			/// \brief	Get good genome id from population	
			///
			////////////////////////////////////////////////////////////
			int getGoodGenome()
			{
				int search_id = rand() % _population_size;
				double best_fitness = _population_fitness[search_id];
				int best_id = search_id;
				for (int r = 0; r < _round_count + 1; r++) {
					search_id = rand() % _population_size;
					if (_population_fitness[search_id] > best_fitness) {
						best_id = search_id;
						best_fitness = _population_fitness[search_id];
					}
				}
				return best_id;
			}

			////////////////////////////////////////////////////////////
			/// \brief	Get bad genome id from population	
			///
			////////////////////////////////////////////////////////////
			int getBadGenome()
			{
				int search_id = rand() % _population_size;
				while (search_id == _best_data_id) { search_id = rand() % _population_size; };
				double worst_fitness = _population_fitness[search_id];
				int worst_id = search_id;
				for (int r = 0; r < _round_count + 1; r++) {
					search_id = rand() % _population_size;
					if (search_id == _best_data_id) { r--; continue; }
					if (_population_fitness[search_id] < worst_fitness) {
						worst_id = search_id;
						worst_fitness = _population_fitness[search_id];
					}
				}
				return worst_id;
			}

			//Population data
			int _population_size;
			std::vector< T > _population_data;
			std::vector< double > _population_fitness;

			//Best gene data
			int _best_data_id;
			double _best_fitness;

			//Parameters
			int _round_count;
			double(*_eval)(T& gene);
			void(*_destroy)(T& gene);
			T(*_generate)(T& mother, T& father);
			
			//For multithreading control
			std::mutex g_lock;
	};

} //namespace ai

#endif /* end of include guard: GA_HPP */

