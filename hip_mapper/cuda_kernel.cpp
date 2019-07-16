#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

#include "hip/hip_runtime.h"

enum FieldIDs {
  FID_X = 1,
  FID_Y,
};

using namespace Legion;

__global__
void init_field_task_kernel(double *ptr, size_t size)
{
//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
//  ptr[tid] = 0.1;
}

__host__
void init_field_task_gpu(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);

  const int point = task->index_point.point_data[0];
  printf("GPU initializing field %d %d for block %d...\n", FID_X, FID_Y, point);

  const FieldAccessor<READ_WRITE,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_x(regions[0], FID_X);
  const FieldAccessor<READ_WRITE,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_y(regions[0], FID_Y);
  
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect));
  double *ptr_x = acc_x.ptr(rect.lo);
  double *ptr_y = acc_y.ptr(rect.lo);

  //init_field_task_kernel<<<1, 32, 0>>>(ptr, rect.volume());
  const unsigned blocks = 1;
  const unsigned threadsPerBlock = 32;
  
  printf("GPU mem %p %p, size %ld\n", ptr_x, ptr_y, rect.volume());
  double *cpu_buff = (double*)malloc(sizeof(double) * rect.volume());
  hipError_t err = hipMemcpy(cpu_buff, ptr_x, sizeof(double)*rect.volume(), hipMemcpyDeviceToHost);
  assert(err == hipSuccess);
  for(int i = 0; i < rect.volume(); i++) {
    cpu_buff[i] += 0.1;
  }
  err = hipMemcpy(ptr_x, cpu_buff, sizeof(double)*rect.volume(), hipMemcpyHostToDevice);
  assert(err == hipSuccess);
  
  for(int i = 0; i < rect.volume(); i++) {
    cpu_buff[i] += 0.1;
  }
  err = hipMemcpy(ptr_y, cpu_buff, sizeof(double)*rect.volume(), hipMemcpyHostToDevice);
  free(cpu_buff);

  //hipLaunchKernelGGL(init_field_task_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, ptr_x, rect.volume());
  
  printf("done with GPU task\n");
}
