////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "ensure.hpp"
#include <stdio.h>
#include <stdlib.h>
#ifdef __unix__
#include <execinfo.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

////////////////////////////////////////////////////////////
#ifdef __unix__
void __print_trace() {
	char pid_buf[30];
	sprintf(pid_buf, "%d", getpid());
	char name_buf[512];
	name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;
	int child_pid = fork();
	if (!child_pid) {           
		dup2(2,1); // redirect output to stderr
		fprintf(stdout,"stack trace for %s pid=%s\n",name_buf,pid_buf);
		execlp("gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt", name_buf, pid_buf, NULL);
		abort(); /* If gdb failed to start */
	} else {
		waitpid(child_pid,NULL,0);
	}
}
#endif

////////////////////////////////////////////////////////////
void __error_exit(const char* str_condition, const char* file, const int line)
{
	//Show errors
	printf("Assertion failed FILE: %s LINE: %d CONDITION: %s\n", file, line, str_condition);

	//Show backtrace
	#ifdef __unix__
	__print_trace();
	#endif

	//terminate program
	exit(-1);
}
