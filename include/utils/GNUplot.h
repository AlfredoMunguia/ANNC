
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include <string>
#include <iostream>

//#ifdef WITH_GNUPLOT

#ifndef _GNUPLOT_HPP_
#define _GNUPLOT_HPP_

class GNUplot {
public:
    GNUplot() {
#ifdef WIN32
       gnuplotpipe = _popen("gnuplot","w");
#else
       gnuplotpipe = popen("gnuplot","w");
#endif
	if (!gnuplotpipe) {
	    throw("Gnuplot not found !");
	}
    }

    ~GNUplot() {
	fprintf(gnuplotpipe,"exit\n");
#ifdef WIN32
       _pclose(gnuplotpipe);
#else
        pclose(gnuplotpipe);
#endif
    }

    void operator()(const std::string& command) {
	fprintf(gnuplotpipe,"%s\n",command.c_str());
	fflush(gnuplotpipe);	
    }

protected:
    FILE *gnuplotpipe;
};

#endif // _GNUPLOT_H_

//#endif // WITH_GNUPLOT
