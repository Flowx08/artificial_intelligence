/*
MIT License

Copyright (c) 2023 Carlo Meroni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GENETICALGORITHM_HPP
#define GENETICALGORITHM_HPP

#include <mutex>
#include <chrono>
#include <vector>

typedef void* Chromosome;

class GeneticAlgorithm {
 public:

   enum MigrationMode {
    ELITIST,
    RANDOM,
   };

   enum StopConditionType {
    ITERATIONS,
    SECONDS,
   };

  GeneticAlgorithm();

  Chromosome getBestChromosome();
  float getBestChromosomeFitness();
  const std::vector<Chromosome>& getPopulation();
  void optimize(unsigned int populationSize, double iterations,
           unsigned int cores = 1, float migrationInterval = 5, bool verbose = true);
  const std::vector<float>& getFitnessLog();

  // functions to overload
  virtual Chromosome initSolution(unsigned int wid);
  virtual float computeFitness(Chromosome& c, unsigned int wid);
  virtual Chromosome crossingOver(const Chromosome& mother,
                                  const Chromosome& father, 
                                  unsigned int wid);
  virtual void mutate(Chromosome& c, float mutationRate, unsigned int wid);
  virtual void freeChromosome(Chromosome& c);
  virtual void printChromosome(Chromosome& c);
  virtual void copyChromosome(Chromosome& src, Chromosome& dst);

  virtual unsigned int roundSchedule(const float time);
  virtual float mutationSchedule(const float time);
  virtual unsigned int getSolutionSize();

  void setMigrationMode(MigrationMode mode);
  void setMigrationSize(const unsigned int size);
  void setSelectionProbability(const float p);
  void setStopConditionType(StopConditionType type);
  
  int randSafe(unsigned int wid);
  float randfSafe(unsigned int wid);

 private:
  void _initializing_multiple(unsigned int workers, unsigned int wid);
  void _run_multiple(unsigned int populationSize, unsigned int wid);
  void _clearPopulation();
  void _migrateIsland_Elitist(unsigned int wid);
  void _migrateIsland_Random(unsigned int wid, unsigned int count);

  unsigned int getGoodChromosomeID(unsigned int wid);
  unsigned int getBadChromosomeID(unsigned int wid);
  float getMeanFitness();

  void setupRandomSeeds(unsigned int workers);
  
  unsigned int populationSize;
  int islandId;
  int islandsCount;
  std::vector<int> islandsState;
  unsigned int migrationInterval;
  std::chrono::high_resolution_clock::time_point lastMigrationTime;
  std::vector<Chromosome> population;
  std::vector<float> populationFitness;
  std::vector<unsigned int> seeds;

  std::vector<float> fitnessLog;

  int bestChromosomeID;
  float bestFitness;
  
  MigrationMode migrationMode;
  unsigned int migrationSize;
  bool verbose;
  double currentIteration;
  double totalIterations;

  unsigned int roundCount;
  float selectionProbability;
  StopConditionType stopConditionType;

  // FOr multithreadhing control
  std::mutex g_lock;
};

#endif  // !GENETICALGORITHM_HPP
