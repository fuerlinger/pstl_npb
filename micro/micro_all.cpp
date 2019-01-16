#include "pstl/execution"
#include "pstl/algorithm"
#include "pstl/numeric"

#include <mutex>
#include <vector>
#include <numeric>
#include <tbb/task_scheduler_init.h>
#include <thread>
#include <chrono>

#include <iostream>

/* parameters */
#define	N		1000

void reset_vector(std::vector<double> &in) {
	
	/*std::generate(pstl::execution::par, in.begin(), in.end(), []() ->double{
		
		return rand() % 1000;
	});*/

	int g_seed = 343434;
	g_seed = (214013*g_seed+2531011);
	for(int i = 0; i < in.size(); i++) in[i] = (g_seed>>16) & 0x7FFF;
}

int main(int argc, char **argv) {
	
	double end_res = 0.0;
	int num_workers;
    if(const char * nw = std::getenv("TBB_NUM_THREADS")) {
        num_workers = atoi(nw);
    } else {
        num_workers = 1;
    }
	
	tbb::task_scheduler_init init(num_workers);
	
	printf("\n");
	printf(" Microbenchmark Parallel STL\n");
	printf(" Hardware Threads: %d\n", std::thread::hardware_concurrency());
	printf(" TBB Threads:      %d\n", num_workers);
	printf(" Test Size:        %ld\n", N);
	printf("\n\n");
	
	std::vector<double> v(N);
	
	srand (time(NULL));
	
	reset_vector(v);
	
	
	
	
	//////////////SORT
	auto begin = std::chrono::high_resolution_clock::now();
	
	std::sort(pstl::execution::seq, v.begin(), v.end());
	
	auto end = std::chrono::high_resolution_clock::now();

	printf(" std::sort::seq():              %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	
	
	
	
	
	printf("\n");
	
	
	/*
	
	
	
	
	/////////////STD::FILL
	begin = std::chrono::high_resolution_clock::now();
	
	std::fill(pstl::execution::par, v.begin(), v.end(), 1.0);
	
	end = std::chrono::high_resolution_clock::now();

	printf(" std::fill():              %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	///////////FOR_EACH FILL
	begin = std::chrono::high_resolution_clock::now();
	
	auto iotabegin = std::chrono::high_resolution_clock::now();
	std::vector<int> w0(N);
	std::iota(w0.begin(), w0.end(), 0);
	auto iotaend = std::chrono::high_resolution_clock::now();
	
	std::for_each(pstl::execution::par, w0.begin(), w0.end(), [&v](int k) {v[k] = 2.0;});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" for_eachfill():           %llu.%06llu seconds, ", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	printf("%llu.%06llu seconds spent in std:iota\n", 
	std::chrono::duration_cast<std::chrono::seconds>(iotaend - iotabegin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(iotaend - iotabegin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	///////////FOR_EACH FILL WITH array
	
	begin = std::chrono::high_resolution_clock::now();
	
	iotabegin = std::chrono::high_resolution_clock::now();
	int w9[N];
	std::iota(&w9[0], &w9[N], 0);
	iotaend = std::chrono::high_resolution_clock::now();
	
	std::for_each(pstl::execution::par, &w9[0], &w9[N], [&v](int k) {v[k] = 2.0;});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" for_eachfill with array():%llu.%06llu seconds, ", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	printf("%llu.%06llu seconds spent in std:iota\n", 
	std::chrono::duration_cast<std::chrono::seconds>(iotaend - iotabegin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(iotaend - iotabegin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	/////////SERIAL FILL
	begin = std::chrono::high_resolution_clock::now();
	
	for(int i = 0; i < N; i++) v[i] = 3.0;
	
	end = std::chrono::high_resolution_clock::now();

	printf(" serial fill():            %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	printf("\n");
	
	
	std::vector<double> x(N);
	reset_vector(x);
	
	
	/////////////STD::COPY
	begin = std::chrono::high_resolution_clock::now();
	
	std::copy(pstl::execution::par, v.begin(), v.end(), x.begin());
	
	end = std::chrono::high_resolution_clock::now();

	printf(" std::copy():              %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, x.begin(), x.end(), end_res);
	reset_vector(v);
	
	///////////FOR_EACH COPY
	begin = std::chrono::high_resolution_clock::now();
	
	iotabegin = std::chrono::high_resolution_clock::now();
	std::vector<int> w1(N);
	std::iota(w1.begin(), w1.end(), 0);
	iotaend = std::chrono::high_resolution_clock::now();
	
	std::for_each(pstl::execution::par, w1.begin(), w1.end(), [&v, &x](int k) {x[k] = v[k];});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" for_eachcopy():           %llu.%06llu seconds, ", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	printf("%llu.%06llu seconds spent in std:iota\n", 
	std::chrono::duration_cast<std::chrono::seconds>(iotaend - iotabegin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(iotaend - iotabegin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, x.begin(), x.end(), end_res);
	reset_vector(v);
	
	///////////FOR_EACH COPY with value
	
	begin = std::chrono::high_resolution_clock::now();
	
	iotabegin = std::chrono::high_resolution_clock::now();
	std::vector<int> w2(N);
	std::iota(w2.begin(), w2.end(), 0);
	iotaend = std::chrono::high_resolution_clock::now();
	
	std::for_each(pstl::execution::par, w2.begin(), w2.end(), [v, &x](int k) {x[k] = v[k];});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" for_eachcopy2():          %llu.%06llu seconds, ", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	printf("%llu.%06llu seconds spent in std:iota\n", 
	std::chrono::duration_cast<std::chrono::seconds>(iotaend - iotabegin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(iotaend - iotabegin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, x.begin(), x.end(), end_res);
	reset_vector(v);
	
	/////////SERIAL COPY
	begin = std::chrono::high_resolution_clock::now();
	
	for(int i = 0; i < N; i++) x[i] = v[i];
	
	end = std::chrono::high_resolution_clock::now();

	printf(" serial copy():            %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, x.begin(), x.end(), end_res);
	reset_vector(v);
	reset_vector(x);
	
	
	
	
	printf("\n");
	
	
	
	
	
	
	
	////////////////STD::TRANSFORM
	begin = std::chrono::high_resolution_clock::now();
	
	std::transform(pstl::execution::par, v.begin(), v.end(), v.begin(), [](double a) ->double{
	
		return exp(sin(a));
	
	});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" std::transform():         %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	///////////FOR_EACH TRANSFORM
	begin = std::chrono::high_resolution_clock::now();
	
	iotabegin = std::chrono::high_resolution_clock::now();
	std::vector<int> w3(N);
	std::iota(w3.begin(), w3.end(), 0);
	iotaend = std::chrono::high_resolution_clock::now();
	
	std::for_each(pstl::execution::par, w3.begin(), w3.end(), [&v](int k) {v[k] = exp(sin(v[k]));});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" for_eachtransform():      %llu.%06llu seconds, ", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	printf("%llu.%06llu seconds spent in std:iota\n", 
	std::chrono::duration_cast<std::chrono::seconds>(iotaend - iotabegin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(iotaend - iotabegin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	///////SERIAL TRANSFORM
	begin = std::chrono::high_resolution_clock::now();
	
	for(int i = 0; i < N; i++) v[i] = exp(sin(v[i]));
	
	end = std::chrono::high_resolution_clock::now();

	printf(" serial transform():       %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	
	
	
	
	printf("\n");
	
	
	
	
	
	
	
	
	////////////////STD::REDUCE
	begin = std::chrono::high_resolution_clock::now();
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	
	end = std::chrono::high_resolution_clock::now();

	printf(" std::reduce():            %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	///////SERIAL REDUCE
	begin = std::chrono::high_resolution_clock::now();
	
	for(int i = 0; i < N; i++) end_res+= v[i];
	
	end = std::chrono::high_resolution_clock::now();

	printf(" serial reduce():          %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	
	
	
	
	
	
	printf("\n");
	
	
	
	
	
	
	
	
	////////////////STD::TRANSFORM_REDUCE
	begin = std::chrono::high_resolution_clock::now();
	
	end_res = std::transform_reduce(pstl::execution::par, v.begin(), v.end(), end_res, std::plus<double>(), [](double a) ->double{
	
		return exp(sin(a));
	
	});
	
	end = std::chrono::high_resolution_clock::now();

	printf(" std::trans_red():         %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	///////SERIAL TRANSFORM_REDUCE
	begin = std::chrono::high_resolution_clock::now();
	
	for(int i = 0; i < N; i++) end_res+= exp(sin(v[i]));
	
	end = std::chrono::high_resolution_clock::now();

	printf(" serial trans_red():       %llu.%06llu seconds\n", 
	std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(), 
	std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() % 1000000);
	
	end_res = std::reduce(pstl::execution::par, v.begin(), v.end(), end_res);
	reset_vector(v);
	
	
	
	*/
	
	
	printf("\n\n Final output: %f\n\n", end_res);
	
    return 0;
}
