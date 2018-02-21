#ifndef ENSURE_HPP
#define ENSURE_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <stdio.h>

#ifdef __unix__
void __print_trace();
#endif

////////////////////////////////////////////////////////////
void __error_exit(const char* str_condition, const char* file, const int line);

////////////////////////////////////////////////////////////
#define ensure(x) if(!(x)) __error_exit(#x, __FILE__, __LINE__)
#define ensure_print(x, fmt, ...) if(!(x)) { printf(fmt, ##__VA_ARGS__); __error_exit(#x, __FILE__, __LINE__); } else

#endif /* end of include guard: ENSURE_HPP */

