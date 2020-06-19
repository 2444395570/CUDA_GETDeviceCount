#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

/*在CUDA程序中获取GPU设备属性*/

int main(void) {
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	//该函数返回支持CUDA的GPU设备的个数
	if (device_count ==0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", device_count);
	}


	//通用设备信息
	/*
	cudaDeviceProp结构体提供了可以用来识别设备以及确定使用的版本信息的属性。它提供的name属性，可以以字符串
	的形式返回设备的名称。还可以通过查询cudaDriverGetVersion和cudaRuntimeGetVersion属性获得设备使用的CUDA Driver
	和运行时引擎的版本。如果有多个设备，并希望使用其中的具有最多流处理器的那个，则可以通过multiProcessorCount
	属性来判断。该属性返回设备上的流多处理器个数。还可以通过使用clockRate属性获取GPU的时钟速率，以KHz返回时钟
	速率。
	*/
	int device;
	cudaDeviceProp device_Property;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&device_Property, device);
	printf("\nDevice %d:\"%s\"\n", device, device_Property.name);

	int driver_Version;
	int runtime_Version;
	cudaDriverGetVersion(&driver_Version);
	cudaRuntimeGetVersion(&runtime_Version);
	printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driver_Version / 1000, (driver_Version % 100) / 10, runtime_Version / 1000, (runtime_Version % 100) / 10);
	printf("Total amount of global memory:%.0f Mbytes (%1lu bytes)\n", (float)device_Property.totalGlobalMem / 1048576.0f, (unsigned long long)device_Property.totalGlobalMem);
	printf("(%2d) Multiprocessors", device_Property.multiProcessorCount);
	printf("GPU Max Clock rate:%.0f MHz (%0.2f GHz)\n", device_Property.clockRate * 1e-3f, device_Property.clockRate * 1e-6f);


	/*
	块和线程都可以时多维的，dim3类型。因此，最好知道每个维度中可以并行启动多少线程和块。对于每个多处理器的
	线程数量和每个块的线程数量也有限制。这个数字可以通过maxThreadsPerMultiProcessor和maxThreadsPerBlock找到。
	如果每个块中启动的线程数量超过每个块中可能的最大线程数量，则程序可能崩溃。
	可以通过maxThreadsDim来确定块中每个维度上的最大线程数量。同样，每个维度中每个网格的最大块可以通过
	maxGridSize来标识。它们都返回一个具有三个值的数组，分别显示x，y，z维度中的最大值。
	*/

	printf("Maximum number of threads per multiprocessor:%d\n", device_Property.maxThreadsPerMultiProcessor);
	printf("Maximum number of threads per block:%d\n", device_Property.maxThreadsPerBlock);
	printf("Max dimension size of a thread block (x,y,z):(%d,%d,%d)\n", device_Property.maxThreadsDim[0], 
		device_Property.maxThreadsDim[1],
		device_Property.maxThreadsDim[2]);
	printf("Max dimension size of a grid size (x,y,z):(%d,%d,%d)\n", device_Property.maxGridSize[0],
		device_Property.maxGridSize[1],
		device_Property.maxGridSize[2]);
}