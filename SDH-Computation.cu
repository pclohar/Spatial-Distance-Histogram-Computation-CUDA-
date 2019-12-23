/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long int d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket * prl_histogram;	
bucket * gpu_res_histogram;	
long long PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
atom * prl_atom_list;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/

__device__ double p2p_distance_gpu(int ind1, int ind2, atom *pa) {
	double x1 = pa[ind1].x_pos;
	double x2 = pa[ind2].x_pos;
	double y1 = pa[ind1].y_pos;
	double y2 = pa[ind2].y_pos;
	double z1 = pa[ind1].z_pos;
	double z2 = pa[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

__global__ void PDH_baseline_gpu(bucket *ph, atom *pa, long long pdh_ac, double res) {
	int i, h_pos;
	double dist;

	register int tid = blockIdx.x * blockDim.x + threadIdx.x;

	register int t =tid + 1;
	for(i = t; i < pdh_ac; i++) {
		dist = p2p_distance_gpu(tid,i,pa);
		h_pos = (int) (dist / res);
		atomicAdd(&(ph[h_pos].d_cnt), 1);
		//ph[h_pos].d_cnt++;
	}
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(char t[]) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for %s version: %ld.%06ld\n", t,sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/


void output_histogram(bucket *histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void compute_print_hist_diff(bucket *cpu, bucket *gpu){
	int i; 
	long long int diff = 0; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		diff = abs((long long int)(cpu[i].d_cnt - gpu[i].d_cnt));
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", diff);
		total_cnt += diff;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	
	/*CPU Execution Starts */
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time("CPU");
	
	/* print out the histogram */
	output_histogram(histogram);
	
	//free(histogram);
	/* CPU Execution Ends*/
	
	/* GPU Execution Starts */

	gpu_res_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);


	cudaMalloc((void**)&prl_atom_list, sizeof(atom) * PDH_acnt);
	cudaMemcpy(prl_atom_list, atom_list, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&prl_histogram, sizeof(bucket) * num_buckets);
	cudaMemcpy(prl_histogram, gpu_res_histogram, sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);	

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);	

	/* call GPU with 512 threads to compute the histogram */
	PDH_baseline_gpu<<<PDH_acnt/128, 128>>>(prl_histogram, prl_atom_list, PDH_acnt, PDH_res);
	
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	float elapsedTime; 
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf( "Time to generate: %0.5f ms\n", elapsedTime );
	cudaEventDestroy(start); 
	cudaEventDestroy(stop); 

	/* check the total running time */ 

	
	cudaMemcpy(gpu_res_histogram, prl_histogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);	
	
	
	/* print out the histogram */

	output_histogram(gpu_res_histogram);

	/* print out the difference between cpu and gpu results */
	printf("\nDifference between CPU and GPU execution results - \n");
	compute_print_hist_diff(histogram, gpu_res_histogram);

	/* print out the histogram ends*/

	/*freeing cuda allocated memory*/
	cudaFree(prl_atom_list);
	cudaFree(prl_histogram);
	cudaFree(gpu_res_histogram);

	free(histogram);
	/* GPU Execution Ends */
	return 0;
}


