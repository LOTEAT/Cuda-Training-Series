<!--
 * @Author: LOTEAT
 * @Date: 2024-09-14 19:00:17
-->

## Solution

### 1. Max Reduction
In hw5, we do not have to write any codes, but we need to learn `reduction` from this homework. 
```cpp
__global__ void reduce(float *gdata, float *out, size_t n){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < n) {  // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += gridDim.x*blockDim.x;  
        }
    // for this loop, suppose that blockDim.x = 4
    // then the dimension of sdata is 4
    // At the beginning of the loop, the value of sdata is
    // s[0], s[1], s[2], s[3]
    // __syncthreads makes all threads synchronization
    // In the first loop, the value of sdata should be
    // s[0] + s[2], s[1] + s[3], s[2], s[3]
    // then __syncthreads makes all threads synchronization
    // In the second loop, the value of sdata should be
    // s[0] + s[2] + s[1] + s[3], s[1] + s[3], s[2], s[3]
    // loop ends
     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // parallel sweep reduction
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) out[blockIdx.x] = sdata[0];
  }
```

### 2. Reductions
I am confused about warp-shuffle.