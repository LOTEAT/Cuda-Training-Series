nvcc -o build/max_reduction max_reduction.cu
echo "Executing max_reduction..."
./build/max_reduction

nvcc -o build/reductions reductions.cu
echo "Executing reductions..."
./build/reductions