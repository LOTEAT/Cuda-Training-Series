<!--
 * @Author: LOTEAT
 * @Date: 2024-08-10 15:44:51
-->
## Solution

### 1. Stencil 1D
This problem actually is the 1D sum pooling operation. For the input vector `in`, we assume that the dimension of `in` is `N`. And we calculate the sum of `2r + 1` elements at once. We hope that the dimension of the output vector `out` is still `N`, so we need to pad `in` with value `v`, which is
$$
in_{pad} = 
\begin{array}{ccc}
& \underbrace{v, ... ,v}_r & ,in, & \underbrace{v, ... ,v}_r \\
\end{array}
$$
And we perform the sum pooling operation
$$
out[i] = \Sigma_{k=i}^{i+n-1}in_{pad}[k]
$$
To make this calculation more faster, we use the shared memory. We first claim it. `temp` is shared by all the threads.
```C++
__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
```
Next, we calculate the index.
```C++
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
```
`gindex` is the index of the value we want to calculate in `out`. `lindex` is the index of the value we want to assign to `temp`.
```C++
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
      temp[lindex - RADIUS] = in[gindex - RADIUS];
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();
``` 
This part is used to copy values from `in`.
```C++
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[lindex + offset];

    // Store the result
    out[gindex] = result;
```
Then we calculate the sum.

### 2. Matrix Multiplication Shared
For matrix $A_{N\times N}$ and matrix $B_{N\times N}$, and the output $C_{N \times N} = A \times B$, where $C[i, j] = \Sigma_{k=1}^NA[i,k]B[k,j]$.

We first claim these matrices.
```C++
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];
```
Note that the matrix is divided into many blocks, the dimension of each block is $block\_size \times block\_size$. 
```C++
  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index
```
`idx`, `idy` is the x-y index of output matrix $C$. 
```C++
  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds/block_size; i++) {

      // We load data into shared memory block by block.
      As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

      // Synchronize
      __syncthreads();

      // Calculate the sum in parallel.
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();

    }

    // Write to global memory
    C[idy*ds+idx] = temp;
  }
```
Explain in the comments.