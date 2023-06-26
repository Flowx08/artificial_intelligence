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

#include "GeneticAlgorithm.hpp"

#include <assert.h>
#include <random>
#include <chrono>
#include <thread>

#ifdef GA_DISTRIBUTED
#include <mpi.h>
#endif

using namespace std::chrono;

GeneticAlgorithm::GeneticAlgorithm() {
  populationSize = 0;
  bestChromosomeID = -1;
  bestFitness = 0;
  islandId = 0;
  islandsCount = 1;
  migrationInterval = 5;
  selectionProbability = 1.0;
  migrationMode = GeneticAlgorithm::ELITIST;
  stopConditionType = GeneticAlgorithm::ITERATIONS;
  migrationSize = 1;
}

Chromosome GeneticAlgorithm::getBestChromosome() {
  return population[bestChromosomeID];
}

float GeneticAlgorithm::getBestChromosomeFitness() { return bestFitness; }

const std::vector<Chromosome>& GeneticAlgorithm::getPopulation() {
  return population;
}

void GeneticAlgorithm::optimize(unsigned int populationSize, double iterations,
    unsigned int cores, float migrationInterval, bool verbose) {
  assert(cores > 0);
  assert(migrationInterval > 0);
  assert(populationSize > 0);
  assert(iterations > 0);
  assert(populationSize >= cores * 2);

  //Setup MPI if used
#ifdef GA_DISTRIBUTED

  // initialize MPI environment
  MPI_Init(NULL, NULL);      

  // number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &islandsCount);

  // the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &islandId);

  //Keep track of islands states 
  islandsState = std::vector<int>(islandsCount, true);

  assert(populationSize > islandsCount);
#endif

  // generate starting population
  _clearPopulation();
  this->migrationInterval = migrationInterval;
  this->populationSize = populationSize;
  population = std::vector<Chromosome>(populationSize);
  populationFitness = std::vector<float>(populationSize);
  roundCount = roundSchedule(0);
  this->verbose = verbose;
  fitnessLog.clear();

  setupRandomSeeds(cores);
  std::vector<std::thread> workers(cores);

  if (verbose)
    printf("Generating and evaluating initial population of size %d...\n",
        populationSize);
  // initialize population
  for (int i = 0; i < cores; i++)
    workers[i] = std::thread(&GeneticAlgorithm::_initializing_multiple, this,
        cores, i);

  for (int i = 0; i < cores; i++) workers[i].join();

  // find current best solution
  bestFitness = populationFitness[0];
  bestChromosomeID = 0;
  for (int i = 1; i < populationSize; i++) {
    if (populationFitness[i] > bestFitness) {
      bestFitness = populationFitness[i];
      bestChromosomeID = i;
    }
  }
  if (verbose) printf("Population generated!\n");

  // setup iterations info
  totalIterations = iterations;
  currentIteration = 0;

  // optimize
  for (int i = 0; i < cores; i++)
    workers[i] = std::thread(&GeneticAlgorithm::_run_multiple, this,
        populationSize, i);

  // Run in parallel!
  for (int i = 0; i < cores; i++) workers[i].join();

#ifdef GA_DISTRIBUTED
  MPI_Finalize();
#endif
}

void GeneticAlgorithm::setupRandomSeeds(unsigned int workers) {
  seeds = std::vector<unsigned int>(workers);
  for (int i = 0; i < workers; i++) seeds[i] = rand();
}

int GeneticAlgorithm::randSafe(unsigned int wid) { return rand_r(&seeds[wid]); }

float GeneticAlgorithm::randfSafe(unsigned int wid) { return (double)rand_r(&seeds[wid]) / (double)RAND_MAX; }

float GeneticAlgorithm::getMeanFitness() {
  double mean = 0;
  for (float f : populationFitness) mean += f;
  return mean / (double)populationSize;
}

unsigned int GeneticAlgorithm::getGoodChromosomeID(unsigned int wid) {
  unsigned int searchId = randSafe(wid) % populationSize;
  unsigned int bestId = searchId;
  float currentFitness = populationFitness[searchId];
  for (int r = 0; r <= roundCount; r++) {
    searchId = randSafe(wid) % populationSize;
    float rand_norm = (double)randSafe(wid) / (double)RAND_MAX;
    if (populationFitness[searchId] > currentFitness) {
      if (rand_norm <= selectionProbability) {
        bestId = searchId;
        currentFitness = populationFitness[searchId];
      }
    }
    else
    {
      if (rand_norm > selectionProbability) {
        bestId = searchId;
        currentFitness = populationFitness[searchId];
      }
    }
  }
  return bestId;
}

unsigned int GeneticAlgorithm::getBadChromosomeID(unsigned int wid) {
  unsigned int searchId;
  double worstFitness;
  unsigned int worstId;

  // select random starting point that it is not the best chromosome
  do {
    searchId = randSafe(wid) % populationSize;
  } while (searchId == bestChromosomeID);
  worstFitness = populationFitness[searchId];
  worstId = searchId;

  for (int r = 0; r < roundCount + 1; r++) {
    searchId = randSafe(wid) % populationSize;

    // do not select best chromosome
    if (searchId == bestChromosomeID) {
      r--;
      continue;
    }

    // found a worse chromosome
    if (populationFitness[searchId] < worstFitness) {
      worstId = searchId;
      worstFitness = populationFitness[searchId];
    }
  }
  return worstId;
}

void GeneticAlgorithm::_initializing_multiple(unsigned int workers,
    unsigned int wid) {
  for (int i = wid; i < populationSize; i += workers) {
    if (verbose && (i % (populationSize / 10) == 0)) {
#ifdef GA_DISTRIBUTED
      if (wid == 0)
        printf("Generating population: %-2d %%\n", (int) (((float)i / (float)populationSize) * 100.f));
#else 
      printf("Generating population: %-2d %%\n", (int) (((float)i / (float)populationSize) * 100.f));
#endif
    }
    population[i] = initSolution(wid);
    populationFitness[i] = computeFitness(population[i], wid);
  }
}

void GeneticAlgorithm::_run_multiple(unsigned int populationSize, unsigned int wid) {
  Chromosome father, mother;
  unsigned int victimId, fatherId, motherId;
  float mutationRate;
  float workDone;
  float sonFitness;
  float logTargetTime = 0.0;
  const float logTargetTimeStep = 0.025;

  // Get father and mother id
  father = initSolution(wid);
  mother = initSolution(wid);
  g_lock.lock();
  fatherId = getGoodChromosomeID(wid);
  motherId = getGoodChromosomeID(wid);
  copyChromosome(population[fatherId], father);
  copyChromosome(population[motherId], mother);
  g_lock.unlock();

  auto startTime = std::chrono::high_resolution_clock::now();

  //reset migration time
#ifdef GA_DISTRIBUTED
  g_lock.lock();
  lastMigrationTime = startTime;
  g_lock.unlock();
#endif

  while (true) {

    if (stopConditionType == StopConditionType::ITERATIONS) {
      workDone = (float)currentIteration / (float)totalIterations;

    } else if (stopConditionType == StopConditionType::SECONDS) {
      double elapsedMicro = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
      workDone = elapsedMicro / (double)(totalIterations * (double)1000000.0);
    }
    
    //Stop condition
    if (workDone >= 1.0)
      break;

    // logging
    if (verbose && (workDone >= logTargetTime) && (islandId == 0)) {
      if (currentIteration != 0 && wid == 0) {
        printf(
            "island: %-2d \titeration: %-8d (%-3d%%) \tbestFitness: %0.6f \tmeanFitness: %0.6f "
            "\tmutationRate: %0.6f \troundCount: %d\n",
            islandId, (int)currentIteration, (int)(workDone * 100), getBestChromosomeFitness(),
            getMeanFitness(), mutationRate, roundCount);
        logTargetTime += logTargetTimeStep;
      }
    }

    mutationRate = mutationSchedule(workDone);
    roundCount = roundSchedule(workDone);


    // generate son and evaluate fitness
    Chromosome son = crossingOver(mother, father, wid);
    mutate(son, mutationRate, wid);
    sonFitness = computeFitness(son, wid);

    g_lock.lock();

    victimId = getBadChromosomeID(wid);
    freeChromosome(population[victimId]);
    population[victimId] = son;
    populationFitness[victimId] = sonFitness;

    // Update best solution
    if (populationFitness[victimId] > bestFitness) {
      bestChromosomeID = victimId;
      bestFitness = populationFitness[victimId];
      //if (verbose) printf("new best solution found!\n");
    }
    fitnessLog.push_back(getBestChromosomeFitness());

    fatherId = getGoodChromosomeID(wid);
    motherId = getGoodChromosomeID(wid);
    copyChromosome(population[fatherId], father);
    copyChromosome(population[motherId], mother);

    // update iterations
    currentIteration++;

    //migrate best solutions from islands if it is time
#ifdef GA_DISTRIBUTED
    auto now = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(now - lastMigrationTime).count() / (double)1000000.0;
    if (elapsedSeconds >= migrationInterval) {
      switch (migrationMode) {
        case MigrationMode::ELITIST:
          _migrateIsland_Elitist(wid);
          break;
        case MigrationMode::RANDOM:
          _migrateIsland_Random(wid, migrationSize);
          break;
      }
      lastMigrationTime = std::chrono::high_resolution_clock::now();
      //lastMigrationTime = clock_type::now();
    }
#endif

    g_lock.unlock();
  }

  //migrate best solutions
#ifdef GA_DISTRIBUTED
  if (wid == 0) {
    islandsState[islandId] = 0;
    switch (migrationMode) {
      case MigrationMode::ELITIST:
        _migrateIsland_Elitist(wid);
        break;
      case MigrationMode::RANDOM:
        _migrateIsland_Random(wid, migrationSize);
        break;
    }
  }
#endif

  freeChromosome(mother);
  freeChromosome(father);
}

void GeneticAlgorithm::_migrateIsland_Elitist(unsigned int wid)
{
#ifdef GA_DISTRIBUTED
  if (verbose) printf("Migrating...\n");
  for (int i = 0; i < islandsCount; i++) {
    if (i == islandId) continue;
    if (islandsState[i] == 0) continue;
    if (i < islandId)
    {
      //Send my own state
      MPI_Send(
          &islandsState[islandId],
          1,
          MPI_INT,
          i,
          0,
          MPI_COMM_WORLD);

      //Send my elitist
      MPI_Send(
          getBestChromosome(),
          getSolutionSize(),
          MPI_CHAR,
          i,
          0,
          MPI_COMM_WORLD);

      //Receive island state
      MPI_Recv(
          &islandsState[i],
          1,
          MPI_INT,
          i,
          0,
          MPI_COMM_WORLD,
          MPI_STATUS_IGNORE);

      //Receive island elitist
      unsigned int victimId = getBadChromosomeID(wid);
      MPI_Recv(
          population[victimId],
          getSolutionSize(),
          MPI_CHAR,
          i,
          0,
          MPI_COMM_WORLD,
          MPI_STATUS_IGNORE);
      float sonFitness = computeFitness(population[victimId], wid);
      populationFitness[victimId] = sonFitness;
      if (populationFitness[victimId] > bestFitness) {
        bestChromosomeID = victimId;
        bestFitness = populationFitness[victimId];
      }
    }
    else
    {
      //Receive island state
      MPI_Recv(
          &islandsState[i],
          1,
          MPI_INT,
          i,
          0,
          MPI_COMM_WORLD,
          MPI_STATUS_IGNORE);

      //Receive island elitist
      unsigned int victimId = getBadChromosomeID(wid);
      MPI_Recv(
          population[victimId],
          getSolutionSize(),
          MPI_CHAR,
          i,
          0,
          MPI_COMM_WORLD,
          MPI_STATUS_IGNORE);
      float sonFitness = computeFitness(population[victimId], wid);
      populationFitness[victimId] = sonFitness;
      if (populationFitness[victimId] > bestFitness) {
        bestChromosomeID = victimId;
        bestFitness = populationFitness[victimId];
      }

      //Send my own state
      MPI_Send(
          &islandsState[islandId],
          1,
          MPI_INT,
          i,
          0,
          MPI_COMM_WORLD);

      //Send my elitist
      MPI_Send(
          getBestChromosome(),
          getSolutionSize(),
          MPI_CHAR,
          i,
          0,
          MPI_COMM_WORLD);
    }
  }
  if (verbose) printf("Elitist migrated!\n");
#endif
}

void GeneticAlgorithm::_migrateIsland_Random(unsigned int wid, unsigned int count)
{
#ifdef GA_DISTRIBUTED
  if (verbose) printf("Migrating...\n");
  for (int i = 0; i < islandsCount; i++) {
    if (i == islandId) continue;
    if (islandsState[i] == 0) continue;
    if (i < islandId)
    {
      for (int k = 0; k < count; k++)
      {
        //Send my own state
        MPI_Send(
            &islandsState[islandId],
            1,
            MPI_INT,
            i,
            0,
            MPI_COMM_WORLD);

        //Send my elitist
        int migrantId = rand() % populationSize;
        Chromosome migrant = population[migrantId];
        MPI_Send(
            migrant,
            getSolutionSize(),
            MPI_CHAR,
            i,
            0,
            MPI_COMM_WORLD);

        //Receive island state
        MPI_Recv(
            &islandsState[i],
            1,
            MPI_INT,
            i,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        MPI_Recv(
            population[migrantId],
            getSolutionSize(),
            MPI_CHAR,
            i,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        float sonFitness = computeFitness(population[migrantId], wid);
        populationFitness[migrantId] = sonFitness;
        if (populationFitness[migrantId] > bestFitness) {
          bestChromosomeID = migrantId;
          bestFitness = populationFitness[migrantId];
        }
      }
    }
    else
    {
      for (int k = 0; k < count; k++)
      {
        //Receive island state
        MPI_Recv(
            &islandsState[i],
            1,
            MPI_INT,
            i,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        //Receive island elitist
        int migrantId = rand() % populationSize;

        //copy solution to send
        Chromosome migrant = initSolution(wid);
        copyChromosome(population[migrantId], migrant); 

        MPI_Recv(
            population[migrantId],
            getSolutionSize(),
            MPI_CHAR,
            i,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        float sonFitness = computeFitness(population[migrantId], wid);
        populationFitness[migrantId] = sonFitness;
        if (populationFitness[migrantId] > bestFitness) {
          bestChromosomeID = migrantId;
          bestFitness = populationFitness[migrantId];
        }

        //Send my own state
        MPI_Send(
            &islandsState[islandId],
            1,
            MPI_INT,
            i,
            0,
            MPI_COMM_WORLD);

        //Send my elitist
        MPI_Send(
            migrant,
            getSolutionSize(),
            MPI_CHAR,
            i,
            0,
            MPI_COMM_WORLD);

        //free memory of copied solution
        freeChromosome(migrant);
      }
    }
  }
  if (verbose) printf("%d solutions migrated!\n", count);
#endif
}

void GeneticAlgorithm::_clearPopulation() {
  if (population.size() == 0) return;
  for (int i = 0; i < population.size(); i++) {
    freeChromosome(population[i]);
  }
  population.clear();
  populationFitness.clear();
  populationSize = 0;
}

const std::vector<float>& GeneticAlgorithm::getFitnessLog()
{
  return fitnessLog;
}

void GeneticAlgorithm::setSelectionProbability(const float p)
{
  selectionProbability = p;
}

void GeneticAlgorithm::setMigrationSize(const unsigned int size)
{
  migrationSize = size;
}

void GeneticAlgorithm::setMigrationMode(MigrationMode mode)
{
  migrationMode = mode;
}

void GeneticAlgorithm::setStopConditionType(StopConditionType type)
{
  stopConditionType = type;
}

Chromosome GeneticAlgorithm::initSolution(unsigned int wid) { return Chromosome(); }

float GeneticAlgorithm::computeFitness(Chromosome& c, unsigned int wid) { return -1; }

unsigned int GeneticAlgorithm::getSolutionSize()
{
  return 0;
}

Chromosome GeneticAlgorithm::crossingOver(const Chromosome& mother,
    const Chromosome& father, unsigned int wid) {
  return Chromosome();
}

void GeneticAlgorithm::mutate(Chromosome& c, float mutationRate, unsigned int wid) {}

unsigned int GeneticAlgorithm::roundSchedule(const float time) {
  return (int)(2 + (4 - 2 + 1) * time);
}

float GeneticAlgorithm::mutationSchedule(const float time) {
  return (1 - time);
}

void GeneticAlgorithm::freeChromosome(Chromosome& c) {}

void GeneticAlgorithm::copyChromosome(Chromosome& src, Chromosome& dst) {}

void GeneticAlgorithm::printChromosome(Chromosome& c) {}
