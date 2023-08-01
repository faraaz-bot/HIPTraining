#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <hip/hip_runtime.h>

__global__ void movingFilter(const float* inData, float* outData, const int K)
{
        //Your Code goes here!
}

/* Function to load from filesystem into vector. */
void loadData(std::string file, std::vector<float> data)
{
    std::ifstream f;
    f.open(file);
    for( int i = 0; i <data.size(); i++)
    {
        f >> data[i];
    }
    f.close();
}

/* Function to write to filesystem from vector. */
void writeData(std::string file, std::vector<float> data)
{
    std::ofstream myFile;
    myFile.open(file);
    for(int i = 0; i < data.size(); i++)
    {
        myFile << data[i] << std::endl;
    }
    myFile.close();
}

int main()
{
    //Initialization
    
    //Allocation    
    
    //Kernel Parameters
    
    //Call your moving filter

    //Free

    return 0;
}
