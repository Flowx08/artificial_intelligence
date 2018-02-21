#include "TextRendering.hpp"
#include "font8x8_basic.h"
#include <string.h>

void render_char(float* image_data, const uint image_width, const uint image_height, const uint image_channels,
		char *char_bitmap, const uint x, const uint y, const uint size, long color)
{
	int set;
	int mask;
	float scale = 8.f / (float)size;
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF
	for (uint i = 0; i < size; i++) {
		if (x + i >= image_width) continue;
		for (uint j = 0; j < size; j++) {
			if (y + j >= image_height) continue;
			set = char_bitmap[(int)(j * scale)] & 1 << (int)(i * scale);
			if (set) {
				for (int c = 0; c < image_channels; c++) {
					image_data[((y + j) * image_width + x + i) * image_channels + c] = GETBYTE(color, (image_channels -1 -c));
				}
			}
		}
	}
}

void text_draw(float* image_data, const uint image_width, const uint image_height, const uint image_channels,
	const char* text, const uint x, const uint y, const uint char_size, const long color)
{
	const int padding = char_size / 4.f;
	const int text_lengt = strlen(text);
	for (int i = 0; i < text_lengt; i++)
		render_char(image_data, image_width, image_height, image_channels, font8x8_basic[(int)text[i]],
				x + i * (char_size + padding), y, char_size, color);		
}
