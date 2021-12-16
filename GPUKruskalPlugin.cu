#include "GPUKruskalPlugin.h"

void GPUKruskalPlugin::convert_array_to_three_way(unsigned short *original_array, unsigned short* vert_out,
									unsigned short* weights, unsigned short* vert_in, int numVert){
	
	unsigned short i,j;
	int pos = 0;
	for(i = 0;i < numVert; i++){
		for(j = i; j < numVert; j++){
			*(vert_out + pos) = i;
			*(vert_in + pos) = j;
			*(weights + pos) = *(original_array + i * numVert + j);
			
			pos++;
		}
	}
		
	printf("pos: %d\n",pos);
}

void GPUKruskalPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 std::string matrixfile = std::string(PluginManager::prefix())+"/"+parameters["matrix"];
 numVert = atoi(parameters["N"].c_str());
	/***************************************************************************************/
	v_out = (unsigned short*)calloc((numVert  * (numVert - 1))/2  + numVert, sizeof(unsigned short));
	v_in = (unsigned short*)calloc((numVert  * (numVert - 1))/2  + numVert, sizeof(unsigned short));
	weights = (unsigned short*)calloc((numVert  * (numVert - 1))/2  + numVert, sizeof(unsigned short));
	
	gpuResult = (unsigned short*)calloc(numVert * numVert, sizeof(unsigned short)); //creating a graph size NxN with zero connection between vertices

        theGraph = (unsigned short*) malloc(numVert*numVert*sizeof(unsigned short));
	std::ifstream myinput(matrixfile.c_str(), std::ios::in);
 int i;
 int M = numVert*numVert;
 for (i = 0; i < M; ++i) {
        short k;
        myinput >> k;
        theGraph[i] = k;
}
convert_array_to_three_way(theGraph,v_out, weights, v_in,numVert);
}

void GPUKruskalPlugin::run() {
	int size = numVert * numVert;
	char* unionMatrix;	
	
	char* checkArray; //it is where each thread reports after checking the union-find-matrix
	
	size_t pitch;	
	
	unsigned short* d_weights_original; //this is where the original graph is gonna be copied in the device
	unsigned short* d_result; //where the resulting spanning tree is gonna be placed in the device
	
	int* d_weights_copy;
	
	int* d_order;
	
	int* vertList; //it is gonna be only size 2.
	
	/****************************************************************************************
	* Alocating memory in the device
	*****************************************************************************************/
	
	cudaMalloc(&d_weights_original, size * sizeof(unsigned short)); //graph in the device
	cudaMalloc(&d_weights_copy, size * sizeof(int)); //Array that is gonna be used in the sort
	
	cudaMalloc(&d_order, size * sizeof(int));//would store a sorted array of number to keep track of the indexes to move
	
	cudaMalloc(&vertList,2 * sizeof(int));
	
	cudaMallocPitch(&unionMatrix, &pitch,
                (numVert) * sizeof(char), numVert); //allocating memory for the union-find-matrix
				
	cudaMalloc(&checkArray, (numVert)*sizeof(char)); //allocating memory for the checkArray

	
	cudaMemcpy(d_weights_original, theGraph, size * sizeof(unsigned short), cudaMemcpyHostToDevice); //Transfering the 1D array from the CPU's DRAM into the Device's DRAM
	
	cudaCheckErrors("cudaMalloc fail");
	/****************************************************************************************
	* End allocating Memory in the device
	*****************************************************************************************/
	int numThreads = 1024;
	int numBlocks = numVert / numThreads + 1;
	int numBlocks_d = (numVert*numVert) / numThreads + 1;
	
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks2D(numVert/threadsPerBlock.x + 1,numVert/threadsPerBlock.y + 1);
	
	fillOrder<<<size/numThreads + 1,numThreads>>>(d_order,size,numVert);
	cudaCheckErrors("filling arrays fail");
	
	copyingGraphs<<<size/numThreads + 1,numThreads>>>(d_weights_original, d_weights_copy, size);
	cudaCheckErrors("Copying arrays fail");
	
	/****************************************************************************************
	* Optimizing space
	*****************************************************************************************/
	cudaFree(d_weights_original);
	
	cudaMalloc(&d_result, size * sizeof(unsigned short)); //Resulting graph
	/****************************************************************************************
	* Sorting Section
	*****************************************************************************************/

	thrust::sort_by_key(thrust::device_ptr<int>(d_weights_copy) , thrust::device_ptr<int>(d_weights_copy + size), thrust::device_ptr<int> (d_order));
	cudaCheckErrors("Sort fail");
	
	/****************************************************************************************
	* End Sorting
	*****************************************************************************************/
	typeof(devFound) found;
	int totalCost;
	
	resetResult<<<numBlocks_d,numThreads>>>(d_result,numVert*numVert); //reset resulting graph
	
	resetArray<<<numBlocks,numThreads>>>(checkArray,numVert); //resetting the checking array to all 0s
	
	resetGlobalFound<<<1,1>>>(); //resseting the global found variable to 0
	cudaCheckErrors("Reset Found fail");
	
	initializeUnionMatrix<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert); //initializing union-find-matrix
	cudaCheckErrors("Union find initialization fail");

	int j; 
	int counter = 0;
	
	for(j = 0;(j < size) && (counter < numVert - 1);j++){ //if we got the min spaming tree
		getValue<<<1,1>>>(d_order, vertList,j,numVert);
		
		//checking if those vertices are not in any set
		checkSet<<<numBlocks,numThreads>>>(unionMatrix,checkArray,pitch,numVert,vertList);
		
	/***************************************************************************************
	* Inserting the node after it was checked that it didnt exist
	****************************************************************************************/
	
		arrayCheck<<<numBlocks,numThreads>>>( checkArray,numVert);
		
		cudaMemcpyFromSymbol(&found, devFound, sizeof(found), 0, cudaMemcpyDeviceToHost);
		
		if(found == 0){
				
			//insertResultingEdge<<<numBlocks,numThreads>>>(d_edges,d_resultEdges,counter,j);
			addToMinWeight<<<1,1>>>(d_weights_copy,j);
			
			counter++;

			//updating unionMatrix
			setValue<<<1,1>>>(unionMatrix,vertList, pitch);	
		
			//Or both inserted vertices's columns
			orCol<<<numBlocks,numThreads>>>(unionMatrix,pitch,numVert,vertList);
		
			//Freaki fast union find
			update<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert,vertList);
			
			//inserting edge into the resulting graph
			insertToResult<<<1,2>>>(d_weights_copy,d_result,vertList,numVert,j);
			
		}
		
		resetArray<<<numBlocks,numThreads>>>(checkArray,numVert); //resetting the checking array to all 0s
		resetGlobalFound<<<1,1>>>(); //resseting the global found variable to 0

	
	}
	

	cudaMemcpyFromSymbol(&totalCost, dev_totalCost, sizeof(totalCost), 0, cudaMemcpyDeviceToHost);
	cudaMemcpy(gpuResult,d_result, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	
	printf("\n\tMinimum cost = %d\n",totalCost);
	
	
	cudaFree(vertList); 	
	cudaFree(d_weights_copy);
	cudaFree(d_result);
	cudaFree(d_order);
	cudaFree(checkArray);
	cudaFree(unionMatrix);

}

void GPUKruskalPlugin::output(std::string file) {
        printf("\n");
        FILE* fileToWrite = fopen(file.c_str(),"w+");

        if(fileToWrite){
                printf("File % s created!\n",file.c_str());
        }else{
                fprintf(stderr,"Error creating %s file\n",file.c_str());
        }

        printf("Writing into file...\n");

        int i,j, counter = 0;
        unsigned short value;

        for(i = 0; i < numVert;i++){
                for(j = i + 1; j < numVert;j++){
                        value = *(gpuResult + i * numVert + j);
                        if(value != 0){
                                fprintf(fileToWrite,"edge (%d,%d) =%d\n",i,j,value);
                                counter++;
                        }

                }
        }



        if(fileToWrite){
                fprintf(fileToWrite,"Total amount of edges inserted: %d\n",counter);
                printf("File %s written successfully!\n",file.c_str());
                fclose(fileToWrite);
        }
}
PluginProxy<GPUKruskalPlugin> GPUKruskalPluginProxy = PluginProxy<GPUKruskalPlugin>("GPUKruskal", PluginManager::getInstance());

