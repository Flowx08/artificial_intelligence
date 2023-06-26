#include "../../src/AI/optimization/GeneticAlgorithm.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <time.h>

struct Solution {
  unsigned char data[512 * 512 * 4];
};

class ImageOptimizer : public GeneticAlgorithm {
  private:
    std::string m_filepath;
    Bitmap *m_bm;

  public:
    ImageOptimizer(const std::string filepath) {
      m_filepath = filepath;
      printf("Loading image...\n");
      m_bm = new Bitmap(filepath, Bitmap::RGB);
      printf("Image loaded: %dx%dx%d\n", m_bm->m_width, m_bm->m_height, m_bm->m_channels);
      assert(m_bm->m_width <= 512 && m_bm->m_height <= 512);
    }

    ~ImageOptimizer() {
      delete m_bm;
    }

    const Bitmap* getBitmap() const {
      return m_bm;
    }

    Chromosome initSolution(unsigned int wid)
    {
      assert(m_bm != NULL);

      Solution* s = new Solution();
      const int img_size = m_bm->m_width * m_bm->m_height * m_bm->m_channels;
      assert(img_size < 512 * 512 * 4);
      for (int i = 0; i < img_size; i++)
        s->data[i] = rand() % 256;
      return (Chromosome)s;    
    }

    float computeFitness(Chromosome& c, unsigned int wid)
    {
      assert(m_bm != NULL);

      Solution* s = (Solution*)c;
      float score = 0.0;
      const int img_size = m_bm->m_width * m_bm->m_height * m_bm->m_channels;
      for (int i = 0; i < img_size; i++)
        score += 255 - abs(m_bm->m_data[i] - s->data[i]);
      return score;
    }

    Chromosome crossingOver(const Chromosome& mother,
        const Chromosome& father, 
        unsigned int wid)
    {
      Solution* m = (Solution*)mother;
      Solution* f = (Solution*)father;
      Solution* s = new Solution();
      const int img_size = m_bm->m_width * m_bm->m_height * m_bm->m_channels;
      for (int i = 0; i < img_size; i++) {
        s->data[i] = rand() % 2 ? m->data[i] : f->data[i];
      }
      return (Chromosome)s; 
    }

    void mutate(Chromosome& c, float mutationRate, unsigned int wid)
    {
      Solution* ch = (Solution*)c;
      const int img_size = m_bm->m_width * m_bm->m_height * m_bm->m_channels;
      ch->data[rand() % img_size] = rand() % 256;
    }

    void copyChromosome(Chromosome& src, Chromosome& dst)
    {
      Solution* s = (Solution*)src;
      Solution* d = (Solution*)dst;
      memcpy(d->data, s->data, sizeof(Solution));
    }

    void freeChromosome(Chromosome& c)
    {
      Solution* s = (Solution*)c;
      delete s;
    }

    unsigned int getSolutionSize()
    {
      return sizeof(Solution);
    }
};

int main(int argc, const char *argv[])
{
  srand((int)time(NULL));
  
  printf("Optimizing image...\n");
  ImageOptimizer opt("./data.png");
  opt.optimize(30, 200000, 8, 5, true);
  printf("Final score: %f\n", opt.getBestChromosomeFitness());
  
  printf("Saving result...\n");
  const Bitmap* bm = opt.getBitmap();
  Bitmap result(bm->m_width, bm->m_height, Bitmap::RGB, 0x000000);
  Solution* s = (Solution*)opt.getBestChromosome();
  const int img_size = bm->m_width * bm->m_height * bm->m_channels;
  for (int i = 0; i < img_size; i++)
    result.m_data[i] = s->data[i];
  result.save("result.png");
  printf("Done.\n");
  return 0;
}
