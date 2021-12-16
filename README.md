# GPUKruskal
# Language: CUDA
# Input: TXT
# Output: TXT
# Tested with: PluMA 1.0, CUDA 10

Run Kruskal's algorithm on the GPU, compute a minimum spanning tree (MST)

Original authors: Manuel Garcia-Cruz, Brian LaRusso, Joseph Gonzalez, Shahinaz Elmahmoudi

The plugin accepts as input a tab-delimited file of keyword-value pairs:
matrix: TSV file of values representing the matrix (tab-delimited)
N: Matrix size (assumed N X N)

MST will be output to a TXT file
