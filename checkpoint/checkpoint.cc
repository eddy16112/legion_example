/* Copyright 2019 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include "legion.h"
#ifdef USE_HDF
#include <hdf5.h>
#endif
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  CHECK_TASK_ID,
  EMPTY_TASK_ID,
};

enum FieldIDs {
  FID_X,
};

bool generate_disk_file(const char *file_name, int num_elements)
{
  // create the file if needed
  int fd = open(file_name, O_CREAT | O_WRONLY, 0666);
  if(fd < 0) {
    perror("open");
    return false;
  }

  // make it large enough to hold 'num_elements' doubles
  int res = ftruncate(fd, num_elements * sizeof(double));
  if(res < 0) {
    perror("ftruncate");
    close(fd);
    return false;
  }

  // now close the file - the Legion runtime will reopen it on the attach
  close(fd);
  return true;
}

#ifdef USE_HDF
bool generate_hdf_file(const char *file_name, const char *dataset_name, int num_elements)
{
  hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) {
    printf("H5Fcreate failed: %lld\n", (long long)file_id);
    return false;
  }

  hsize_t dims[1];
  dims[0] = num_elements;
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
  if(dataspace_id < 0) {
    printf("H5Screate_simple failed: %lld\n", (long long)dataspace_id);
    H5Fclose(file_id);
    return false;
  }

  hid_t dataset = H5Dcreate2(file_id, dataset_name,
			     H5T_IEEE_F64LE, dataspace_id,
			     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset < 0) {
    printf("H5Dcreate2 failed: %lld\n", (long long)dataset);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return false;
  }

  // close things up - attach will reopen later
  H5Dclose(dataset);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
  return true;
}
#endif

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 64; 
  char disk_file_name[256];
  strcpy(disk_file_name, "checkpoint.dat");
#ifdef USE_HDF
  char hdf5_file_name[256];
  char hdf5_dataset_name[256];
  hdf5_file_name[0] = 0;
  strcpy(hdf5_dataset_name, "FID_X");
#endif
  // Check for any command line arguments
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-f"))
	      strcpy(disk_file_name, command_args.argv[++i]);
#ifdef USE_HDF
      if (!strcmp(command_args.argv[i],"-h"))
	      strcpy(hdf5_file_name, command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-d"))
	      strcpy(hdf5_dataset_name, command_args.argv[++i]);
#endif
    }
  }

  printf("Running for %d elements...\n", num_elements);

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
  }

  // ****************************************** checkpoint **********************
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);

  TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument(NULL, 0));
  init_launcher.add_region_requirement(
        RegionRequirement(input_lr, WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.add_field(0/*idx*/, FID_X);
  runtime->execute_task(ctx, init_launcher);
  
  runtime->issue_execution_fence(ctx);
  
  PhysicalRegion cp_pr;
  LogicalRegion cp_lr = runtime->create_logical_region(ctx, is, input_fs);
#ifdef USE_HDF
  if(*hdf5_file_name) {
    // create the HDF5 file first - attach wants it to already exist
    bool ok = generate_hdf_file(hdf5_file_name, hdf5_dataset_name, num_elements);
    assert(ok);
    std::map<FieldID,const char*> field_map;
    field_map[FID_X] = hdf5_dataset_name;
    printf("Checkpointing data to HDF5 file '%s' (dataset='%s')\n", hdf5_file_name, hdf5_dataset_name);
    cp_pr = runtime->attach_hdf5(ctx, hdf5_file_name, input_lr, input_lr, field_map, LEGION_FILE_READ_WRITE);
  } else
#endif
  {
    // create the disk file first - attach wants it to already exist
    bool ok = generate_disk_file(disk_file_name, num_elements);
    assert(ok);
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_X);
    printf("Checkpointing data to disk file '%s'\n", disk_file_name);
    cp_pr = runtime->attach_file(ctx, disk_file_name, input_lr, input_lr, field_vec, LEGION_FILE_READ_WRITE);
  }
  
  cp_pr.wait_until_valid();
#if 0
  CopyLauncher copy_launcher;
  copy_launcher.add_copy_requirements(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr),
      RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
  copy_launcher.add_src_field(0, FID_X);
  copy_launcher.add_dst_field(0, FID_X);
  runtime->issue_copy_operation(ctx, copy_launcher);
#endif

#ifdef USE_HDF
  if(*hdf5_file_name) {
    runtime->detach_hdf5(ctx, cp_pr);
  } else
#endif
  {
    runtime->detach_file(ctx, cp_pr);
  }
  
  runtime->issue_execution_fence(ctx);
  
  // ************************************ restart ****************
  PhysicalRegion restart_pr;
  LogicalRegion restart_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalRegion input_lr2 = runtime->create_logical_region(ctx, is, input_fs);
#ifdef USE_HDF
  if(*hdf5_file_name) {
    std::map<FieldID,const char*> field_map;
    field_map[FID_X] = hdf5_dataset_name;
    printf("Recoverring data to HDF5 file '%s' (dataset='%s')\n", hdf5_file_name, hdf5_dataset_name);
    restart_pr = runtime->attach_hdf5(ctx, hdf5_file_name, restart_lr, restart_lr, field_map, LEGION_FILE_READ_WRITE);
  } else
#endif
  {
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_X);
    printf("Recoverring data to disk file '%s'\n", disk_file_name);
    restart_pr = runtime->attach_file(ctx, disk_file_name, restart_lr, restart_lr, field_vec, LEGION_FILE_READ_WRITE);
  }
  
  cp_pr.wait_until_valid();
#if 0
  CopyLauncher copy_launcher2;
  copy_launcher2.add_copy_requirements(
      RegionRequirement(restart_lr, READ_ONLY, EXCLUSIVE, restart_lr),
      RegionRequirement(input_lr2, WRITE_DISCARD, EXCLUSIVE, input_lr2));
  copy_launcher2.add_src_field(0, FID_X);
  copy_launcher2.add_dst_field(0, FID_X);
  runtime->issue_copy_operation(ctx, copy_launcher2);
#endif

#ifdef USE_HDF
  if(*hdf5_file_name) {
    runtime->detach_hdf5(ctx, restart_pr);
  } else
#endif
  {
    runtime->detach_file(ctx, restart_pr);
  }
  
  runtime->issue_execution_fence(ctx);
  
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(NULL, 0));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr2, READ_ONLY, EXCLUSIVE, input_lr2));
  check_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, cp_lr);
  runtime->destroy_logical_region(ctx, input_lr2);
  runtime->destroy_logical_region(ctx, restart_lr);
  runtime->destroy_field_space(ctx, input_fs);
}

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = 0.29;
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double received = acc_x[*pir];
    assert(received == 0.29);
  }
  printf("Success\n");
}

void empty_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
}


int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "init");
  }
  
  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }
  
  {
    TaskVariantRegistrar registrar(EMPTY_TASK_ID, "empty");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<empty_task>(registrar, "empty");
  }

  return Runtime::start(argc, argv);
}


