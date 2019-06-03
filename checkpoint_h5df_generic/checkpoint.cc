#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#ifdef USE_HDF
#include <hdf5.h>
#endif

#include "checkpoint_task.h"

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  RESTART_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y, 
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 64; 
  int num_subregions = 4;
  int num_files = 2;
  char file_name[256];
  strcpy(file_name, "checkpoint.dat");
  
  // Check for any command line arguments
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-m"))
        num_files = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-f"))
	      strcpy(file_name, command_args.argv[++i]);
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
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  
  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  
  Rect<1> file_color_bounds(0,num_files-1);
  IndexSpace file_is = runtime->create_index_space(ctx, file_color_bounds);
  IndexPartition file_ip = runtime->create_equal_partition(ctx, is, file_is);
  
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  LogicalRegion input_lr2 = runtime->create_logical_region(ctx, is, input_fs);
  LogicalPartition input_lp2 = runtime->get_logical_partition(ctx, input_lr2, ip);
  
  LogicalPartition file_checkpoint_lp = runtime->get_logical_partition(ctx, input_lr, file_ip);
  LogicalPartition file_recover_lp2 = runtime->get_logical_partition(ctx, input_lr2, file_ip);

  // **************** init task *************************
  ArgumentMap arg_map;
  {
  IndexLauncher init_launcher(INIT_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);  
  init_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, init_launcher);
  }
  {
  IndexLauncher init_launcher(INIT_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);  
  init_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, init_launcher);
  }
  runtime->issue_execution_fence(ctx);
  
  // ************************************ checkpoint ****************
  struct task_args_s task_arg;
  strcpy(task_arg.file_name, file_name);
  std::map<FieldID, std::string> field_string_map;
  field_string_map[FID_X] = "FID_X";
  field_string_map[FID_Y] = "FID_Y";
  
  Realm::Serialization::DynamicBufferSerializer dbs(0);
  dbs << field_string_map;
  task_arg.field_map_size = dbs.bytes_used();
  memcpy(task_arg.field_map_serial, dbs.detach_buffer(), task_arg.field_map_size);
  
  
  CheckpointIndexLauncher checkpoint_launcher(file_is, 
                              TaskArgument(&task_arg, sizeof(task_arg)), arg_map);  
  checkpoint_launcher.add_region_requirement(
        RegionRequirement(file_checkpoint_lp, 0/*projection ID*/, 
                          READ_ONLY, EXCLUSIVE, input_lr));
  checkpoint_launcher.region_requirements[0].add_field(FID_X);
  checkpoint_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, checkpoint_launcher);
  runtime->issue_execution_fence(ctx);
  
  // ************************************ restart ****************
  IndexLauncher restart_launcher(RESTART_TASK_ID, file_is, 
                              TaskArgument(&task_arg, sizeof(task_arg)), arg_map);  
  restart_launcher.add_region_requirement(
        RegionRequirement(file_recover_lp2, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr2));
  restart_launcher.region_requirements[0].add_field(FID_X);
  restart_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, restart_launcher);
  runtime->issue_execution_fence(ctx);
  
  
  // *************************** check result ********************  
  {
  IndexLauncher check_launcher(CHECK_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);
  check_launcher.add_region_requirement(
        RegionRequirement(input_lp2, 0/*projection ID*/, 
                          READ_ONLY, EXCLUSIVE, input_lr2));
  check_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, check_launcher);
  }
  {
    IndexLauncher check_launcher(CHECK_TASK_ID, color_is, 
                                TaskArgument(NULL, 0), arg_map);
    check_launcher.add_region_requirement(
          RegionRequirement(input_lp2, 0/*projection ID*/, 
                            READ_ONLY, EXCLUSIVE, input_lr2));
    check_launcher.region_requirements[0].add_field(FID_Y);
    runtime->execute_index_space(ctx, check_launcher);
  }
  runtime->issue_execution_fence(ctx);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, input_lr2);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, file_is);
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
  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double received = acc_x[*pir];
    assert(received == (0.29 + point));
  }
  printf("Success\n");
}

void restart_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  const int point = task->index_point.point_data[0];
  
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  std::map<FieldID, std::string> field_string_map;
  Realm::Serialization::FixedBufferDeserializer fdb(task_arg.field_map_serial, task_arg.field_map_size);
  bool ok  = fdb >> field_string_map;
  if(!ok) {
    printf("task args deserializer error\n");
  }
  
  std::string fname(task_arg.file_name);
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  
  
  PhysicalRegion restart_pr;
  LogicalRegion input_lr2 = regions[0].get_logical_region();
  LogicalRegion restart_lr = runtime->create_logical_region(ctx, input_lr2.get_index_space(), input_lr2.get_field_space());

  AttachLauncher hdf5_attach_launcher(EXTERNAL_HDF5_FILE, restart_lr, restart_lr);
  std::map<FieldID,const char*> field_map;
  for (std::map<FieldID, std::string>::iterator it = field_string_map.begin() ; it != field_string_map.end(); ++it) {
    field_map.insert(std::make_pair(it->first, (it->second).c_str()));
  }
  printf("Recoverring data to HDF5 file '%s' (dataset='%ld')\n", file_name, field_map.size());
  hdf5_attach_launcher.attach_hdf5(file_name, field_map, LEGION_FILE_READ_WRITE);
  restart_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);

  std::set<FieldID> field_set = task->regions[0].privilege_fields; 
  CopyLauncher copy_launcher2;
  copy_launcher2.add_copy_requirements(
      RegionRequirement(restart_lr, READ_ONLY, EXCLUSIVE, restart_lr),
      RegionRequirement(input_lr2, WRITE_DISCARD, EXCLUSIVE, input_lr2));
  for(std::set<FieldID>::iterator it = field_set.begin(); it != field_set.end(); ++it) {
    copy_launcher2.add_src_field(0, *it);
    copy_launcher2.add_dst_field(0, *it);
  }
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
  
  CheckpointIndexLauncher::register_task();
  
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


