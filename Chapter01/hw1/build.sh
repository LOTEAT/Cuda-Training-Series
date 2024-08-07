nvcc -o build/hello hello.cu
echo "Executing hello..."
./build/hello

nvcc -o build/vector_add vector_add.cu
echo "Executing vector_add..."
./build/vector_add

nvcc -o build/matrix_mul matrix_mul.cu
echo "Executing matrix_mul..."
./build/matrix_mul