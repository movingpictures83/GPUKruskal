#ifndef GPUKRUSKALPLUGIN_H
#define GPUKRUSKALPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>
#include <emmintrin.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <fstream>


int find(int*,int);

int getWeight(int array[],int row, int col, int n);
int setWeight(int* array[],int row, int col, int n, int value);

class GPUKruskalPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
		void convert_array_to_three_way(unsigned short *original_array, unsigned short* vert_out,
                                                                        unsigned short* weights, unsigned short* vert_in, int numVert);
                std::string inputfile;
		std::string outputfile;

	int numVert; 
	unsigned short* v_out;
	unsigned short* v_in;
	unsigned short* weights;
	
	unsigned short* theGraph; //pointer to a 1D array where the matrix NxN is gonna be
std::map<std::string, std::string> parameters;
unsigned short* gpuResult;
};



#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

		
/****************************************************************************************
* Global variables declaration
****************************************************************************************/
__device__ int devFound; //global variable to get result of the check method

__device__ int dev_totalCost; //cumulative minimun weight



/****************************************************************************************
* Union find implementation section
****************************************************************************************/


/*
*	This method initialize a union-find-matrix  with the diagonal set to 1.
*	This matrix keeps track of sets of vertices connected to and specific vertice.
*	Each column means the set of vertices that are connected to the vertices denoted by the column number.
*	
*	Ex. 3 x 3 adjecency matrix initialized where each node belongs to its set of connected vertices. 0 connected to 0
*	1 connected to 1 and so on.
*	
*	indexes | 0 | 1 | 2 |
*	--------------------
*		0   | 1 | 0 | 0 | 
*		----------------
*		1   | 0	| 1 | 0 |
*		----------------
*		2   | 0	| 0 | 1 |
*		
*		
*	Ex. 3 x 3 adjecency matrix after some insertions. Here in column 0 we have row 0 to mark as 1 and row 1 mark as 1. This
*	means that  vertices 0 and 1 are connected to vertice 0 (column number)
*	
*	indexes | 0 | 1 | 2 |
*	--------------------
*		0   | 1 | 1 | 0 | 
*		----------------
*		1   | 1	| 1 | 0 |
*		----------------
*		2   | 0	| 0 | 1 |	
*/
__global__ void initializeUnionMatrix(char * matrix,size_t pitch, int numVert){
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	//printf("working on col:%d row:%d \n",col,row);
	
	if(row < numVert && col < numVert){
		
		if(row == col){ //if is t he diagonal make mark it as 1. Each element belongs to its set.
			
			char* row_ptr= (char*)((char*)matrix + row * pitch);
			
			row_ptr[col] = '1';
			
			//printf("working on col:%d row:%d VALUE:%d\n",col,row,1);
			
		}else{
			
			//printf("working on col:%d row:%d VALUE:%d\n",col,row,0);
			char* row_ptr= (char*)((char*)matrix + row * pitch);
			row_ptr[col] = '0';
			
		}
	}
	
	
	__syncthreads();
	
}// end initializeUnionMatrix

/*
*	This method makes the first update in the union-find-matrix
*	reflecting the fact that the new inserted vertices  are connected 
*	between them
*
*	Arguments:
*
*	int* matrix: pointer to the union-find-matrix
*	size_t pitch: size (in bytes) of each row
*	int* list: Mainly a size 2 array with the 2 vertice numbers to be updated.
*/
__global__ void setValue(char* matrix,int* list,size_t pitch){
		//inserting values in the union-find-matrix
		char* ptr = (char*)((char*)matrix + (*(list+0)) * pitch);
		ptr[(*(list+1))] = '1';
		
		ptr = (char*)((char*)matrix +(*(list+1)) * pitch);
		ptr[(*(list+0))] = '1';
} //end of setValue


/*
*	This methods or 2 columns in the union-find-matrix. This 
*	garantees that both sets have the same values.
*	Arguments:
*	
*	int* matrix: pointer to the union-find-matrix
*	size_t pitch: size (in bytes) of each row
*	int numVert: number of vertices of the adjecency matrix
*	int* list: Mainly a size 2 array with the 2 vertices number to be checked
*/
__global__ void orCol(char *matrix,size_t pitch,int numVert,int * list){
	int firstVert = (*(list + 0));
	int secondVert = (*(list + 1));
	
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < numVert){ //checking not to go out of boundries
		
		char* row_ptr = (char*)((char*)matrix + row * pitch);
		
		char firstValue = row_ptr[firstVert];
		char secondValue = row_ptr[secondVert];
		
		//TODO: Work on a better way to or.
		//not a fancy or :(
		if(firstValue == '1')
			row_ptr[secondVert] = '1';
		if(secondValue == '1')
			row_ptr[firstVert] = '1';		
	}
		
	__syncthreads();
}//end orCol



/*
*	This method updates the whole union-find-matrix forcing all the vertices 
*	connected to the new added vertices to be updated to reflec the new connections created.
*	Arguments:
*	
*	int* matrix: pointer to the union-find-matrix
*	size_t pitch: size (in bytes) of each row
*	int numVert: number of vertices of the adjecency matrix
*	int* list: Mainly a size 1 array with the vertice number to be used as reference for the update
*/
__global__ void update(char *matrix, size_t pitch,int numVert,int * list){
	int reference = (*(list+0));
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	char* row_ptr;
	if((col < numVert) && (row < numVert) ){ // checking not to go out of boundries
		if(col != reference){ //Avoiding writing to the reference column
			row_ptr= (char*)((char*)matrix + row * pitch);
			char value = row_ptr[reference];
			
			if(value == '1'){//if the current row exists in the reference column set
				row_ptr= (char*)((char*)matrix + col * pitch);
				char exist = row_ptr[reference];
				
				if(exist == '1'){
					row_ptr= (char*)((char*)matrix + row * pitch);
					row_ptr[col] = '1';
				}
			}
		}
	}
	
	 __syncthreads();
}//end of update


/****************************************************************************************
* End of Union find implementation section
****************************************************************************************/

/****************************************************************************************
* Avoiding Loops section
****************************************************************************************/

/*
*	This method checks the union-find-matrix to see if 2 vertices are already connected. 
*	It has n threads, being n the number of vertices, each checking a single column.
*
*	Arguments:
*	
*	int* matrix: pointer to the union-find-matrix
*	int* checkArray: pointer to the array where the result of the checking will be
*	size_t pitch: size (in bytes) of each row
*	int numVert: number of vertices of the adjecency matrix
*	int* list: Mainly a size 2 array with the 2 vertices number to be checked
*/
__global__ void checkSet(char* matrix,char* checkArray, size_t pitch, int numVert,int * list){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if((col < numVert)){ //checking not to go out of boundries in the union-find-matrix
		char* rowFirstVert = (char*)((char*)matrix + (*(list+0)) * pitch);
		char firstValue = rowFirstVert[col];
		
		char* rowSecondVert = (char*)((char*)matrix + (*(list+1)) * pitch);
		char secondValue = rowSecondVert[col];
		
		if((firstValue == '1') && (secondValue == '1')){//if both elements were found in the same set it means that they are connected already
			checkArray[col] = '1'; // marking that correspondent spot of the current thread as 1, meaning, found
		}else{
			//printf("Not found from thread:%d!\n",threadIdx.x);
		}
			
	}
	
	__syncthreads();
	
}//end CheckSet


/*
*	This methods check if any thread from the checkSet method found the two vertices candidates for insertion. 
*	
*	Arguments:
*	
*	int* array: pointer to the array where the result of the checking are
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void arrayCheck(char * array,int numVert){
	
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(pos < numVert){ //checking not to go out of boundries
		char result = *(array + pos);
		
		if(result == '1'){ //if some thread reported 1. It means that they belong to the same set, hence, they are connected.
			devFound = 1;	
		}
		
	}
			
	__syncthreads();
}//end arrayCheck


/*
*	Reseting the value of devFound to 0
*/
__global__ void resetGlobalFound(){
	devFound = 0;	
}//end resetGlobalFound


/*
*	Resets the check Array to all 0s.
*	
*	Arguments:
*
*	int* array: pointer to the array where the result of the checking are
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void resetArray(char * array,int numVert){
	
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(pos < numVert){
		*(array + pos) = '0';
	}
	
}//end resetArray

/****************************************************************************************
* End of Avoiding Loops section
****************************************************************************************/


/****************************************************************************************
* Extra tool methods section
****************************************************************************************/

/*
* 	Fill out the array of in an ordered way.
*	
*	Arguments:
*	
*	int* order: array to be filled
*	int size: max lenght of the array
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void fillOrder(int* order, int size, int numVert){

	int index = blockIdx.x * blockDim.x + threadIdx.x; //getting the actual position in the array.
	
	if(index < size)//checking not to go out of boundries
		*(order + index) = index;
	
	__syncthreads();
	
}//end of fillOrder


/*
* This method creates a copy of the array.
*
* 	Arguments:
*
*	unsigned char* source: Array to be copied
*	unsigned char* dest: Array to copy to
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void copyingGraphs(unsigned short* source, int* dest, int size){

	int index = blockIdx.x * blockDim.x + threadIdx.x; //getting the actual position in the array.
	
	if(index < size) //checking not to go out of boundries
		*(dest + index) = (int)*(source + index);
	
	__syncthreads();
	
}//end copyingGraphs


/*
*	This method increments the total minimun weight of the new minimun spamming tree
*
*	Arguments:
*
*	unsigned char* original: Original array with the orignial positions for the weights
*	int* list: Mainly a size 2 array with the 2 vertice numbers to get the weight of the edge between them.
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void addToMinWeight(int* original,int pos){
	
	int weight = *(original + pos);
	
	dev_totalCost += weight;
}//end addToMinWeight


/*
*	This method retrieves the column and row of 
*/
__global__ void getValue(int* ordered, int * list,int pos, int numVert){
	int index = *(ordered + pos);
	int row = index / numVert;
	int col = index % numVert;
	
	//this is how we retrieve the data back in the CPU
	*(list + 0) = row;
	*(list + 1) = col;

}//end getValue


/*
*	Prints an array in the GPU
*/
__global__ void printA(int* array, int size){
	int i;
	
	for(i = 0; i < size; i++){
		printf("%d ",array[i]);
		
	}
	
	printf("\n");
	
}
/*
* 	Inserts edge back into the result graph
*/
__global__ void insertToResult(int* origin, unsigned short* result, int* list, int numVert,int pos){
	if(threadIdx.x == 0){
		*(result + list[0]*numVert + list[1]) = (unsigned short)*(origin + pos); 
	}else if(threadIdx.x == 1){
		*(result + list[1]*numVert + list[0]) = (unsigned short)*(origin + pos);
	}
}
/*
*
*	Resets the and unsigned short array to all 0s. It is an overrided implementation of te resetArray Method
*	
*	Arguments:
*
*	int* array: pointer to the array where the result of the checking are
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void resetResult(unsigned short * array,int numVert){
	
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(pos < numVert){
		*(array + pos) = 0;
	}
	
}//end resetArray
/****************************************************************************************/

#endif
