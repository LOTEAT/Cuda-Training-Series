nvcc -o build/stencil_1d stencil_1d.cu
echo "Executing stencil_1d..."
./build/stencil_1d

nvcc -o build/matrix_mul_shared matrix_mul_shared.cu
echo "Executing matrix_mul_shared..."
./build/matrix_mul_shared