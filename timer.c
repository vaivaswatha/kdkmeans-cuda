#ifndef TIMER_C
#define TIMER_C
/* Stolen from the internet somewhere. */

#include <sys/time.h>
static struct timeval _tstart, _tend;
static struct timezone tz;

static void tstart(void)
{
    gettimeofday(&_tstart, &tz);
}
static void tend(void)
{
    gettimeofday(&_tend,&tz);
}

static double tval()
{
    double t1, t2;
    
    t1 =  (double)_tstart.tv_sec + (double)_tstart.tv_usec/(1000*1000);
    t2 =  (double)_tend.tv_sec + (double)_tend.tv_usec/(1000*1000);
    return t2-t1;
}

#endif
