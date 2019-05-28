#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

using namespace Legion;

__global__
void init_field_task_kernel(double *ptr, size_t size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[tid] = 0.1;
}

__host__
void init_field_task_gpu(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<READ_WRITE,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  double *ptr = acc.ptr(rect.lo);

  init_field_task_kernel<<<1, 32, 0>>>(ptr, rect.volume());
  
  printf("done with GPU task\n");
}