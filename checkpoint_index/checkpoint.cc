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
  CHECKPOINT_TASK_ID,
  RESTART_TASK_ID,
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
  int num_subregions = 2;
  char file_name[256];
  strcpy(file_name, "checkpoint.dat");
#ifdef USE_HDF
  char hdf5_dataset_name[256];
  strcpy(hdf5_dataset_name, "FID_X");
  int use_hdf5 = 0;
#endif
  // Check for any command line arguments
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-f"))
	      strcpy(file_name, command_args.argv[++i]);
#ifdef USE_HDF
      if (!strcmp(command_args.argv[i],"-h"))
	      use_hdf5 = atoi(command_args.argv[++i]);
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
  
  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  LogicalRegion input_lr2 = runtime->create_logical_region(ctx, is, input_fs);
  LogicalPartition input_lp2 = runtime->get_logical_partition(ctx, input_lr2, ip);

  // **************** init task *************************
  ArgumentMap arg_map;
  IndexLauncher init_launcher(INIT_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);  
  init_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, init_launcher);
  runtime->issue_execution_fence(ctx);
  
  // ************************************ checkpoint ****************
  IndexLauncher checkpoint_launcher(CHECKPOINT_TASK_ID, color_is, 
                              TaskArgument(&use_hdf5, sizeof(int)), arg_map);  
  checkpoint_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          READ_ONLY, EXCLUSIVE, input_lr));
  checkpoint_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, checkpoint_launcher);
  runtime->issue_execution_fence(ctx);
  
  // ************************************ restart ****************
  IndexLauncher restart_launcher(RESTART_TASK_ID, color_is, 
                              TaskArgument(&use_hdf5, sizeof(int)), arg_map);  
  restart_launcher.add_region_requirement(
        RegionRequirement(input_lp2, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr2));
  restart_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, restart_launcher);
  runtime->issue_execution_fence(ctx);
  
  
  // *************************** check result ********************  
#if 1
  IndexLauncher check_launcher(CHECK_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);
  check_launcher.add_region_requirement(
        RegionRequirement(input_lp2, 0/*projection ID*/, 
                          READ_ONLY, EXCLUSIVE, input_lr2));
  check_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, check_launcher);
#else
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(NULL, 0));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr2, READ_ONLY, EXCLUSIVE, input_lr2));
  check_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, check_launcher);
#endif  
  runtime->issue_execution_fence(ctx);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, input_lr2);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, is);
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
  printf("Initializing field %d for block %d, pid %d\n", fid, point, getpid());

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    acc[*pir] = 0.29 + point;
  }
  
  // ****************************************** checkpoint **********************
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  const int point = task->index_point.point_data[0];
  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double received = acc_x[*pir];
    assert(received == (0.29 + point));
  }
  printf("Success\n");
}

void checkpoint_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  std::string fname("checkpoint.dat");
  const int point = task->index_point.point_data[0];
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  char hdf5_dataset_name[256];
  strcpy(hdf5_dataset_name, "FID_X");
  
  int use_hdf5 = *((const int*)task->args);
  
  PhysicalRegion cp_pr;
  Rect<1> rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  LogicalRegion input_lr = regions[0].get_logical_region();
  LogicalRegion cp_lr = runtime->create_logical_region(ctx, input_lr.get_index_space(), input_lr.get_field_space());
#ifdef USE_HDF  
  if(use_hdf5 == 1) {
    // create the HDF5 file first - attach wants it to already exist
    bool ok = generate_hdf_file(file_name, hdf5_dataset_name, rect.volume());
    assert(ok);
    AttachLauncher hdf5_attach_launcher(EXTERNAL_HDF5_FILE, cp_lr, cp_lr);
    std::map<FieldID,const char*> field_map;
    field_map[FID_X] = hdf5_dataset_name;
    printf("Checkpointing data to HDF5 file '%s' (dataset='%s')\n", file_name, hdf5_dataset_name);
    hdf5_attach_launcher.attach_hdf5(file_name, field_map, LEGION_FILE_READ_WRITE);
    cp_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);
   // cp_pr.wait_until_valid();
  } else 
#endif  
  {
    // create the disk file first - attach wants it to already exist
    bool ok = generate_disk_file(file_name, rect.volume());
    assert(ok);
    AttachLauncher file_attach_launcher(EXTERNAL_POSIX_FILE, cp_lr, cp_lr);
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_X);
    printf("Checkpointing data to disk file '%s'\n", file_name);
    file_attach_launcher.attach_file(file_name, field_vec, LEGION_FILE_READ_WRITE);
    cp_pr = runtime->attach_external_resource(ctx, file_attach_launcher);
  }
  
  CopyLauncher copy_launcher1;
  copy_launcher1.add_copy_requirements(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr),
      RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
  copy_launcher1.add_src_field(0, FID_X);
  copy_launcher1.add_dst_field(0, FID_X);
  runtime->issue_copy_operation(ctx, copy_launcher1);
  
  
  Future fu = runtime->detach_external_resource(ctx, cp_pr, true);
  fu.wait();
  runtime->destroy_logical_region(ctx, cp_lr);
}

void restart_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  std::string fname("checkpoint.dat");
  const int point = task->index_point.point_data[0];
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  char hdf5_dataset_name[256];
  strcpy(hdf5_dataset_name, "FID_X");
  
  int use_hdf5 = *((const int*)task->args);
  
  PhysicalRegion restart_pr;
  LogicalRegion input_lr2 = regions[0].get_logical_region();
  LogicalRegion restart_lr = runtime->create_logical_region(ctx, input_lr2.get_index_space(), input_lr2.get_field_space());
#ifdef USE_HDF
  if(use_hdf5 == 1) {
    AttachLauncher hdf5_attach_launcher(EXTERNAL_HDF5_FILE, restart_lr, restart_lr);
    std::map<FieldID,const char*> field_map;
    field_map[FID_X] = hdf5_dataset_name;
    printf("Recoverring data to HDF5 file '%s' (dataset='%s')\n", file_name, hdf5_dataset_name);
    hdf5_attach_launcher.attach_hdf5(file_name, field_map, LEGION_FILE_READ_WRITE);
    restart_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);
  } else
#endif
  {
    AttachLauncher file_attach_launcher(EXTERNAL_POSIX_FILE, restart_lr, restart_lr);
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_X);
    printf("Recoverring data to disk file '%s'\n", file_name);
    file_attach_launcher.attach_file(file_name, field_vec, LEGION_FILE_READ_WRITE);
    restart_pr = runtime->attach_external_resource(ctx, file_attach_launcher);
  }
  
  CopyLauncher copy_launcher2;
  copy_launcher2.add_copy_requirements(
      RegionRequirement(restart_lr, READ_ONLY, EXCLUSIVE, restart_lr),
      RegionRequirement(input_lr2, WRITE_DISCARD, EXCLUSIVE, input_lr2));
  copy_launcher2.add_src_field(0, FID_X);
  copy_launcher2.add_dst_field(0, FID_X);
  runtime->issue_copy_operation(ctx, copy_launcher2);

  Future fu = runtime->detach_external_resource(ctx, restart_pr, true);
  fu.wait();
  runtime->destroy_logical_region(ctx, restart_lr);
}


int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "init");
  }
  
  {
    TaskVariantRegistrar registrar(CHECKPOINT_TASK_ID, "checkpoint");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<checkpoint_task>(registrar, "checkpoint");
  }
  
  {
    TaskVariantRegistrar registrar(RESTART_TASK_ID, "restart");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<restart_task>(registrar, "restart");
  }
  
  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}


