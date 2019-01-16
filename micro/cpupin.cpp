#include "pstl/execution"
#include "pstl/algorithm"
#include "pstl/numeric"

#include <thread>
#include <mutex>

#include <tbb/task_scheduler_init.h>

int main(int argc, char **argv) {
	
	double end_res = 0.0;
	int num_workers;
    if(const char * nw = std::getenv("TBB_NUM_THREADS")) {
        num_workers = atoi(nw);
    } else {
        num_workers = 1;
    }
	
	//int n = tbb::task_scheduler_init::default_num_threads();
	
	tbb::task_scheduler_init init(num_workers);

	int n = std::thread::hardware_concurrency();
	
	printf("\n");
	printf(" Microbenchmark Parallel STL\n");
	printf(" Hardware Threads: %d\n", n);
	printf(" TBB Threads:      %d\n", num_workers);
	printf("\n\n");
	
	std::mutex my_m;

	int res[n];

	for(int i = 0; i < n; i++) res[i]=0;
	
	int np = 1000;
	
	int v[np];
	std::iota(&v[0], &v[np], 1);
	
	std::for_each(pstl::execution::par, &v[0], &v[np], [&my_m, &res](int k)
	{
		my_m.lock();
		printf("Iteration %10d on core %3d\n", k, sched_getcpu());
		res[sched_getcpu()]++;
		my_m.unlock();
	});

	int sum = 0;

	for(int i = 0; i < n; i++) sum += res[i];

	for(int i = 0; i < n; i++)
	{
		if(res[i] != 0) printf("Core %3d used in %10d cases (%f percent)\n", i, res[i], 100 * (double) res[i]/ (double) sum);
	}
	
	
    return 0;
}
