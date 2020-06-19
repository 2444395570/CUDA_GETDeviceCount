#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

/*��CUDA�����л�ȡGPU�豸����*/

int main(void) {
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	//�ú�������֧��CUDA��GPU�豸�ĸ���
	if (device_count ==0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", device_count);
	}


	//ͨ���豸��Ϣ
	/*
	cudaDeviceProp�ṹ���ṩ�˿�������ʶ���豸�Լ�ȷ��ʹ�õİ汾��Ϣ�����ԡ����ṩ��name���ԣ��������ַ���
	����ʽ�����豸�����ơ�������ͨ����ѯcudaDriverGetVersion��cudaRuntimeGetVersion���Ի���豸ʹ�õ�CUDA Driver
	������ʱ����İ汾������ж���豸����ϣ��ʹ�����еľ�����������������Ǹ��������ͨ��multiProcessorCount
	�������жϡ������Է����豸�ϵ����ദ����������������ͨ��ʹ��clockRate���Ի�ȡGPU��ʱ�����ʣ���KHz����ʱ��
	���ʡ�
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
	����̶߳�����ʱ��ά�ģ�dim3���͡���ˣ����֪��ÿ��ά���п��Բ������������̺߳Ϳ顣����ÿ���ദ������
	�߳�������ÿ������߳�����Ҳ�����ơ�������ֿ���ͨ��maxThreadsPerMultiProcessor��maxThreadsPerBlock�ҵ���
	���ÿ�������������߳���������ÿ�����п��ܵ�����߳��������������ܱ�����
	����ͨ��maxThreadsDim��ȷ������ÿ��ά���ϵ�����߳�������ͬ����ÿ��ά����ÿ��������������ͨ��
	maxGridSize����ʶ�����Ƕ�����һ����������ֵ�����飬�ֱ���ʾx��y��zά���е����ֵ��
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