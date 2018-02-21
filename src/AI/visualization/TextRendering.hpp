#ifndef TEXTRENDERING_HPP
#define TEXTRENDERING_HPP

typedef unsigned int uint;

void text_draw(float* image_data, const uint image_width, const uint image_height, const uint image_channels,
	const char* text, const uint x, const uint y, const uint char_size, const long color);

#endif /* end of include guard: TEXTRENDERING_HPP */

