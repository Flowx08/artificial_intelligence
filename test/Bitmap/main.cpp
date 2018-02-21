#include <stdio.h>
#include "../../src/AI/visualization/Bitmap.hpp"

int main(int argc, const char *argv[])
{
	Bitmap bm(100, 100, Bitmap::RGB, 0xFFFFFF);
	bm.drawRect(10, 10, 64, 64, 0x00FF00, 24);
	bm.drawRect(10, 10, 64, 64, 0xFF0000, 3);
	bm.rotate(bm.getWidth()/2, bm.getHeight()/2, 30, 0x000000);
	bm.drawCircle(32, 32, 32, 0x0000FF, 3);
	bm.save("test.png");
	return 0;
}
