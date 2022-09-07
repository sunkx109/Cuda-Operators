#include <cuda_runtime.h>
#include <stdio.h>
#define THREAD_PER_BLOCK 256

//reduce baseline
__global__ void reduce0(float *d_in,float *d_out)
{
    __shared__ float sdata[THREAD_PER_BLOCK];
    //each thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    //do reduction in shared mem
    for(unsigned int s=1;s<blockDim.x;s*=2)
    {
        if(tid%(2*s)==0)
        {
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    //write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];

}

int main()
{
    const int N=32*1024*1024; //
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int block_num=N/THREAD_PER_BLOCK;
    float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK,1);
    dim3 Block( THREAD_PER_BLOCK,1);

    reduce0<<<Grid,Block>>>(d_a,d_out);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);

}