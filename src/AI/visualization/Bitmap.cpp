////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Bitmap.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>

////////////////////////////////////////////////////////////
///	CONSTANTS
////////////////////////////////////////////////////////////
#define LOG(msg) printf("[OK] in %s l%d: %s\n", __FILE__, __LINE__, msg)
#define ERROR(msg) printf("[ERROR] in %s l%d: %s\n", __FILE__, __LINE__, msg)

////////////////////////////////////////////////////////////
///	TOOLS
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// \brief	Get file fomat from filepath string	
///
////////////////////////////////////////////////////////////
std::string getfileformat(std::string filepath)
{
	int dot_pos = -1;
	std::string fmt = "";

	//Find last dot position
	for (int i = (int)filepath.size()-1; i >= 0; i--) {
		if (filepath[i] == '.') {
			dot_pos = i;
			break;
		}
	}

	//Check for errors
	if (dot_pos == -1 || dot_pos == (int)filepath.size()-1) return "";

	//Get format
	for (int i = dot_pos+1; i < (int)filepath.size(); i++)
		fmt += filepath[i];

	return fmt;
}

////////////////////////////////////////////////////////////
///	BITMAP
////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
Bitmap::Bitmap(std::string filepath, Channels channels)
{
	//Variables
	int stbi_fmt = 0;
	int fsize = 0;
	std::string fmt =  getfileformat(filepath);

	//Check file format
	if (fmt == "jpg" || fmt == "jpeg" || fmt == "png" || fmt == "bmp" || fmt == "tga")
	{

		switch (channels)
		{
			case Bitmap::MONO:	stbi_fmt = STBI_grey; break;
			case Bitmap::RGB:	stbi_fmt = STBI_rgb; break;
			case Bitmap::RGBA:	stbi_fmt = STBI_rgb_alpha; break;
			default:
								ERROR("number of channels not supported");
								return;
		}

		//Open file
		FILE* file = fopen(filepath.c_str(), "rb");
		if (!file) {
			ERROR("can't open the file");
			return;	
		}

		//Get file size
		fseek(file, 0L, SEEK_END);
		fsize = (int)ftell(file);
		fseek(file, 0L, SEEK_SET);

		//Read all the file into filedata
		stbi_uc* fileData = (stbi_uc*)malloc(sizeof(stbi_uc)* fsize);
		fread(fileData, sizeof(stbi_uc)* fsize, 1, file);
		fclose(file);

		//Get the image data
		m_data = stbi_load_from_memory(fileData, fsize, &m_width, &m_height, &m_channels, stbi_fmt);

		//Free the filedata
		if (fileData) free(fileData);

		//Check for stb_image fail
		if (m_data == NULL) {
			ERROR("can't get the image data");
			printf("Details : %s\n", stbi_failure_reason());
			return;
		}
	}
	//Check file format
	else if (fmt == "pgm")
	{
		/*
		   PGM file structure:
		   2 bytes		P5
		   1 byte			whitespace(blanks, TAB, CRs, LFs)
		   ? bytes		width in ASCII
		   1 byte			whitespace
		   ? bytes		height in ACII
		   1 byte			whitespace
		   ? bytes		maximum gray value in ASCII (0 < x < 65536)
		   1 byte			whitespace (usually newline)
		   w*h*k bytes	data (k = 1 if maximum gray is <= 255 else k = 2)
		   */

		//Open file
		FILE* file = fopen(filepath.c_str(), "rb");
		if (!file) {
			ERROR("can't open the file");
			return;	
		}

		//First let's check the header P5 string
		std::string p5; 
		p5 += getc(file); 
		p5 += getc(file);
		if (p5 != "P5") {
			ERROR("can't find P5 string in the header");
			return;
		}

		//Jump whitespace
		getc(file);

		//Read width and height
		m_width = m_height = -1;
		fscanf(file, "%d %d", &m_width, &m_height);
		if (m_width <= 0 || m_width > 4096 || m_height <= 0 || m_height > 4096) {
			ERROR("invalid image size");
			return;
		}

		//Jump whitespace
		getc(file);

		//Read maximum grey value
		int maxgrey = -1;
		fscanf(file, "%d", &maxgrey);

		//Jump whitespace
		getc(file);

		//Free old data and allocate space for new data
		m_data = (unsigned char*)malloc(sizeof(char) * (m_width * m_height));

		//Set channel info, PGM format support only mono channel
		m_channels = 1;

		//Load data
		int k = maxgrey > 255 ? 2 : 1;
		double scale = 255.f / (double)maxgrey;
		unsigned char cbuf[2]; 
		cbuf[0] = cbuf[1] = 0;
		int ibuf;
		for (int y = 0; y < m_height; y++) {
			for (int x = 0; x < m_width; x++) {
				fread(&cbuf, sizeof(char), k, file);
				ibuf = (double)(cbuf[0] + (cbuf[1] >> 6)) * scale;
				m_data[y * m_width + x] = ibuf;
			}
		}

		//Close file
		fclose(file);
	}
	else
	{
		ERROR("can't load image, unknown file format");
		printf("Details: format:%s\n", fmt.c_str());
	}
}

///////////////////////////////////////////////////////////
Bitmap::Bitmap(Bitmap& bm_source, int x, int y, int w, int h)
{
	//Check for errors
	if (bm_source.getData() == NULL) {
		ERROR("can't load from region");
		return;
	}

	//Set new info
	m_channels = bm_source.getChannels();
	m_width = w;
	m_height = h;

	//Allocate memory
	m_data = (unsigned char*)malloc(w * h * m_channels);

	//Copy each pixel line from one bitmap to the other
	unsigned char* fromdata = bm_source.getData();
	int fromwidth = bm_source.getWidth();
	for (int from_y = 0; from_y < h; from_y++)
		memcpy(	&m_data[from_y * w  * m_channels],
				&fromdata[((from_y + y) * fromwidth + x) * m_channels],
				m_channels * w);
}

////////////////////////////////////////////////////////////
Bitmap::Bitmap(int w, int h, Channels channels, long color)
{
	//Set new info
	m_channels = channels;
	m_width = w;
	m_height = h;

	//Allocate memory
	m_data = (unsigned char*)malloc(w * h * m_channels);

	//Get byte shortcut
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF

	//Get color per channel
	unsigned char c[4];
	if (channels == Bitmap::RGBA)
	{
		c[0] = GETBYTE(color, 3);
		c[1] = GETBYTE(color, 2);
		c[2] = GETBYTE(color, 1);
		c[3] = GETBYTE(color, 0);
	}
	else
	{
		c[0] = GETBYTE(color, 2);
		c[1] = GETBYTE(color, 1);
		c[2] = GETBYTE(color, 0);
	}

	//Set mono color if needed
	if (channels == 1) c[0] = (c[0] + c[1] + c[2])/3;

	//Fill the data
	for (int i = 0; i < w * h; i++)
		for (int j = 0; j < channels; j++)
			m_data[i * channels + j] = c[j];	

}

////////////////////////////////////////////////////////////
Bitmap::~Bitmap()
{
	if (m_data) free(m_data);
	m_data = NULL;
}

////////////////////////////////////////////////////////////
bool Bitmap::save(std::string filepath)
{
	//Get file format
	std::string fileformat = getfileformat(filepath);
	Formats fmt = Formats::PNG;
	if (fileformat == "png") fmt = Formats::PNG;
	else if (fileformat == "bmp") fmt = Formats::BMP;
	else if (fileformat == "tga") fmt = Formats::TGA;
	else //check for erros
	{
		ERROR("Unknown fileformat, can't save the Bitmap");
		return 0;
	}
	
	switch (fmt)
	{
		case Bitmap::PNG:
			return stbi_write_png(filepath.c_str(), m_width, m_height, m_channels,
					(void*)m_data, m_width * m_channels);	

		case Bitmap::BMP:
			return stbi_write_bmp(filepath.c_str(), m_width, m_height, m_channels, (void*)m_data);

		case Bitmap::TGA:
			return stbi_write_tga(filepath.c_str(), m_width, m_height, m_channels, (void*)m_data);

		default: return 0;
	}
}

////////////////////////////////////////////////////////////
void Bitmap::fill(long color)
{
	//Get byte shortcut
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF

	//Get color per channel
	unsigned char c[4];
	if (m_channels == Bitmap::RGBA)
	{
		c[0] = GETBYTE(color, 3);
		c[1] = GETBYTE(color, 2);
		c[2] = GETBYTE(color, 1);
		c[3] = GETBYTE(color, 0);
	}
	else
	{
		c[0] = GETBYTE(color, 2);
		c[1] = GETBYTE(color, 1);
		c[2] = GETBYTE(color, 0);
	}

	//Set mono color if needed
	if (m_channels == 1) c[0] = (c[0] + c[1] + c[2])/3;

	//Fill the data
	for (int i = 0; i < m_width * m_height; i++)
		for (int j = 0; j < m_channels; j++)
			m_data[i * m_channels + j] = c[j];	

}

////////////////////////////////////////////////////////////
void Bitmap::convertToMono()
{
	if (m_channels == 1) return;

	//New bitmap data
	unsigned char* ndata = (unsigned char*)malloc(m_width * m_height);

	int color = 0;
	for (int x = 0; x < m_width; x++) {
		for (int y = 0; y < m_height; y++) {

			//Get mono color
			color = 0;
			for (int i = 0; i < m_channels; i++)
				color += m_data[(y * m_width + x) * m_channels + i];
			color /= m_channels;

			//Set in the new bitmap data
			ndata[y * m_width + x] = color;
		}
	}

	//Free old data
	free(m_data);

	//Set new data
	m_data = ndata;
	m_channels = 1;
}

////////////////////////////////////////////////////////////
void Bitmap::adjustContrast(float contrast)
{
	for (int x = 0; x < m_width; x++) {
		for (int y = 0; y < m_height; y++) {

			static float c[3];
			static int pos = 0;
			for (int i = 0; i < m_channels; i++) {
				pos = (y * m_width + x) * m_channels + i;
				c[i] = m_data[pos] / 255.0f;
				c[i] = (((c[i] - 0.5f) * contrast) + 0.5f) * 255.0f;
				if (c[i] > 255) c[i] = 255;
				else if (c[i] < 0) c[i] = 0;
				m_data[pos] = (int)c[i];
			}
		}
	}
}

///////////////////////////////////////////////////////////
void Bitmap::resize(int w, int h)
{
	if (w == m_width && h == m_height) return;

	//New bitmap data
	unsigned char* ndata = (unsigned char*)malloc(w * h * m_channels);

	//Proportions
	const double WRate = (double)m_width / w;
	const double HRate = (double)m_height / h;

	//Copy resized data into the new array
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			memcpy(	&ndata[(y * w + x) * m_channels],
					&m_data[((int)(y * HRate) * m_width + (int)(x * WRate)) * m_channels],
					m_channels);

	//Unload old data
	if (m_data) free(m_data);
	m_data = NULL;

	//Update informations
	m_data = ndata;
	m_width = w;
	m_height = h;
}

////////////////////////////////////////////////////////////
void Bitmap::rotate(const unsigned int center_x, const unsigned int center_y, const float angle, const long fill_color)
{
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF
	
	//Allocate temporary buffer
	float* temp = new float[m_width * m_height * m_channels];
	for (int x = 0; x < m_width; x++)
		for (int y = 0; y < m_height; y++)
			for (int c = 0; c < m_channels; c++)
				temp[(y * m_width + x) * m_channels + c] = GETBYTE(fill_color, m_channels -1 -c);

	//Compute constants
	const float radians = angle * 3.14159f/180.0;
	const float a = cos(radians);
	const float b = sin(radians);
	const int xoffset = center_x - (center_x * a - center_y * b);
	const int yoffset = center_y - (center_x * b + center_y * a);

	//Compute rotation in temporary buffer
	// x' = x * cos(a) + y * sin(a) + xoffset
	// y' = y * cos(a) - x * sin(a) + yoffset
	int nx, ny;
	for (int x = 0; x < m_width; x++) {
		for (int y = 0; y < m_height; y++) {
			nx = x * a - y * b + xoffset;
			ny = x * b + y * a + yoffset;
			if (nx < 0 || nx >= m_width) continue;
			if (ny < 0 || ny >= m_height) continue;
			for (int c = 0; c < m_channels; c++) {
				temp[(y * m_width + x) * m_channels + c] = m_data[(ny * m_width + nx) * m_channels + c];
			}
		}
	}

	//Copy result to final tensor
	for (int i = 0; i < m_width * m_height * m_channels; i++) m_data[i] = temp[i];

	//Free temporary buffer
	delete[] temp;
}

////////////////////////////////////////////////////////////
void Bitmap::copyToRegion(Bitmap& bm_dest, int from_x, int from_y, int from_w,
		int from_h, int to_x, int to_y, int to_w, int to_h)
{
	//Check for errrors
	if (bm_dest.getChannels() != m_channels) {
		ERROR("The two regions must have the same color channels count");
		return;
	}

	//Proportions
	const double WRate = (double)from_w / to_w;
	const double HRate = (double)from_h / to_h;

	//Copy data
	unsigned char *data = (unsigned char*)malloc(from_w * from_h * m_channels);
	for (int x = 0; x < from_w; x++) {
		for (int y = 0; y < from_h; y++) {
			memcpy(&data[(y * from_w + x) * m_channels],
					&m_data[((y + from_y) * m_width + (x + from_x)) * m_channels],
					m_channels);
		}
	}

	//Paste resized data
	int destwidth = bm_dest.getWidth();
	int destheight = bm_dest.getHeight();
	unsigned char* destdata = bm_dest.getData();
	int destchannels = bm_dest.getChannels();
	for (int x = 0; x < to_w; x++) {
		for (int y = 0; y < to_h; y++) {
			int fy = (double)(y * HRate);
			int fx = (double)(x * WRate);
			if (to_x + x >= destwidth || to_x + x < 0 ||
					to_y + y >= destheight || to_y + y < 0)
				continue;
			memcpy( &destdata[((to_y + y) * destwidth + (to_x + x)) * destchannels],
					&data[(fy * from_w + fx) * m_channels],
					m_channels);
		}
	}

	//Free
	free(data);
}

////////////////////////////////////////////////////////////
void Bitmap::filterThreshold(int threshold)
{
	for (int i = 0; i < m_width * m_height * m_channels; i++) {
		if (m_data[i] < threshold) m_data[i] = 0;
		else m_data[i] = 255;
	}
}

////////////////////////////////////////////////////////////
void Bitmap::drawRect(const unsigned int x, const unsigned int y, const unsigned int width,
		const unsigned int height, const long color, const unsigned int border_thickness)
{
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF
	
	if (border_thickness <= 0) //fill the rectangle with color
	{
		for (unsigned int i = x; i < x + width; i++) {
			if (i >= m_width) continue;
			for (unsigned int j = y; j < y + height; j++) {
				if (j >= m_height) continue;
				for (unsigned int c = 0; c < m_channels; c++) {
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c)); 
				}
			}
		}
	}
	else //just draw the borders of the rectangle
	{
		const int offset_l = floor((float)border_thickness / 2.0 + 0.5 - border_thickness % 2);
		const int offset_h = floor((float)border_thickness / 2.0 + 0.5);
		
		//Draw y axis borders
		for (int j = (int)y - offset_l; j < (int)y + (int)height + offset_h; j++) {
			if (j < 0 || j >= m_height) continue;
			for (int i = (int)x -offset_l; i < (int)x + offset_h; i++) {
				if (i < 0 || i >= m_width) continue;
				for (int c = 0; c < m_channels; c++) {
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c)); 
				}
			}
			for (int i = (int)x + (int)width -offset_l; i < (int)x + (int)width + offset_h; i++) {
				if (i < 0 || i >= m_width) continue;
				for (int c = 0; c < m_channels; c++) {
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c)); 
				}
			}
		}

		//Draw x axis borders
		for (int i = (int)x - offset_l; i < (int)x + (int)width + offset_h; i++) {
			if (i < 0 || i >= m_width) continue;
			for (int j = (int)y - offset_l; j < (int)y + offset_h; j++) {
				if (j < 0 || j >= m_height) continue;
				for (int c = 0; c < m_channels; c++) {
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c)); 
				}
			}
			for (int j = (int)y + (int)height -offset_l; j < (int)y + (int)height + offset_h; j++) {
				if (j < 0 || j >= m_height) continue;
				for (int c = 0; c < m_channels; c++) {
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c)); 
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////
void Bitmap::drawCircle(const unsigned int x, const unsigned int y, const unsigned int radius,
		const long color, const unsigned int border_tickness)
{
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF
	
	if (border_tickness <= 0)
	{
		for (int i = (int)x - (int)radius; i < (int)x + (int)radius; i++) {
			if (i < 0 || i >= m_width) continue;
			for (int j = (int)y - (int)radius; j < (int)y + (int)radius; j++) {
				if (j < 0 || j >= m_height) continue;
				if (sqrt(pow(i - (int)x, 2) + pow(j - (int)y, 2)) > radius) continue;
				for (int c = 0; c < m_channels; c++)
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c));
			}
		}
	}
	else
	{
		for (int i = (int)x - (int)radius; i < (int)x + radius; i++) {
			if (i < 0 || i >= m_width) continue;
			for (int j = (int)y - (int)radius; j < (int)y + radius; j++) {
				if (j < 0 || j >= m_height) continue;
				const unsigned int distance = sqrt(pow(i - (int)x, 2) + pow(j - (int)y, 2));
				if (distance > radius || distance < (int)radius - (int)border_tickness) continue;
				for (int c = 0; c < m_channels; c++)
					m_data[(j * m_width + i) * m_channels + c] = GETBYTE(color, (m_channels -1 -c));
			}
		}
	}
}

///////////////////////////////////////////////////////////
unsigned char* Bitmap::getData()
{
	return m_data;
}

///////////////////////////////////////////////////////////
int Bitmap::getWidth()
{
	return m_width;
}

///////////////////////////////////////////////////////////
int Bitmap::getHeight()
{
	return m_height;
}

///////////////////////////////////////////////////////////
int Bitmap::getChannels()
{
	return m_channels;
}
