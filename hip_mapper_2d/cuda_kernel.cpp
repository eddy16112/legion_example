#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "hip/hip_runtime.h"
#include "legion.h"

enum FieldIDs {
  FID_X = 1,
  FID_Y,
};

using namespace Legion;

extern "C" {
hipError_t hipMemcpy_H(void* dst, const void* src, size_t size, hipMemcpyKind kind);
};

__global__
void init_field_task_kernel(int* ptr)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[tid] = 7;
}

void init_field_task_gpu(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);

  const int point = task->index_point.point_data[0];
  printf("GPU initializing field %d %d for block %d...\n", FID_X, FID_Y, point);

  const FieldAccessor<READ_WRITE,int,2,coord_t,
          Realm::AffineAccessor<int,2,coord_t> > acc_x(regions[0], FID_X);
  const FieldAccessor<READ_WRITE,int,2,coord_t,
          Realm::AffineAccessor<int,2,coord_t> > acc_y(regions[0], FID_Y);
  
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect));
  int *ptr_x = acc_x.ptr(rect.lo);
  int *ptr_y = acc_y.ptr(rect.lo);

  //init_field_task_kernel<<<1, 32, 0>>>(ptr, rect.volume());
  const unsigned blocks = 1;
  const unsigned threadsPerBlock = 32;
  
  printf("GPU mem %p %p, size %ld\n", ptr_x, ptr_y, rect.volume());
#if 1
  int *cpu_buff = (int*)malloc(sizeof(int) * rect.volume());
  hipError_t err = hipMemcpy(cpu_buff, ptr_x, sizeof(int)*rect.volume(), hipMemcpyDeviceToHost);
  assert(err == hipSuccess);
  for(int i = 0; i < rect.volume(); i++) {
    cpu_buff[i] += 1;
  }
  err = hipMemcpy(ptr_x, cpu_buff, sizeof(int)*rect.volume(), hipMemcpyHostToDevice);
  assert(err == hipSuccess);
  
  err = hipMemcpy(cpu_buff, ptr_y, sizeof(int)*rect.volume(), hipMemcpyDeviceToHost);
  assert(err == hipSuccess);
  for(int i = 0; i < rect.volume(); i++) {
    cpu_buff[i] += 2;
  }
  err = hipMemcpy(ptr_y, cpu_buff, sizeof(double)*rect.volume(), hipMemcpyHostToDevice);
  free(cpu_buff);
#else
  
  hipMemsetD32(ptr_x, 2, rect.volume());
  hipMemsetD32(ptr_y, 4, rect.volume());
#endif  

  //hipFunction_t my_kernel = NULL;
  hipLaunchKernelGGL((init_field_task_kernel), dim3(blocks), dim3(threadsPerBlock), 0, 0, ptr_x);
  //hipModuleLaunchKernel(init_field_task_kernel, 1, 0, 0, 32, 0, 0, 64, 0, ptr_x, NULL)
  //init_field_task_kernel<<<1, 32, 0>>>(ptr_x);
  
  //hipDeviceSynchronize();
  
  printf("done with GPU task\n");
}
