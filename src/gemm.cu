#include <cuda_runtime.h>
#include <stdio.h>



void matrixmul(const float *A,const float *B,float *C,int M,int N,int K)
{
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            int tmp_sum=0;
            for(int k=0;k<K;k++)
            {
               tmp_sum+= A[i*M+k]*B[k*K+j];
            }
            C[i*M+j]=tmp_sum;
        }
    }
}
__global__ void matrixMul(const float *A,const float *B,float *C,int M,int N,int K)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(ty<M && tx<N)
    {
        float c = 0;
        for(int i=0;i<K;i++)
        {
            c += A[ty*K+i]*B[i*N+tx];
        }
        C[ty*N+tx]=c;
    }

}