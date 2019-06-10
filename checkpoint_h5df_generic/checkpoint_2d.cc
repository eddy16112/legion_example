#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <mpi.h>

#include "checkpoint_task.h"

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y, 
  FID_Z, 
  FID_W,
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
      if (!strcmp(command_args.argv[i],"-n"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-s"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-m"))
        num_files = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-f"))
	      strcpy(file_name, command_args.argv[++i]);
    }
  }

  printf("Running for %d elements...\n", num_elements);

  Point<2> lo(0, 0);
  Point<2> hi(num_elements-1, num_subregions-1);
  const Rect<2> elem_rect(lo, hi);
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
  //IndexPartition file_ip = runtime->create_equal_partition(ctx, is, file_is);
  IndexPartition file_ip = runtime->create_pending_partition(ctx, is, file_is);
  int idx = 0; 
  for (int point = 0; point < num_files; point++) {
    std::vector<IndexSpace> subspaces;
    for (int i = 0; i < num_subregions/num_files; i++) {
      subspaces.push_back(runtime->get_index_subspace(ctx, ip, idx));
      idx ++;
    }
    runtime->create_index_space_union(ctx, file_ip, point, subspaces);
  }
  
  
  LogicalRegion input_lr_1 = runtime->create_logical_region(ctx, is, input_fs);
  LogicalPartition input_lp_1 = runtime->get_logical_partition(ctx, input_lr_1, ip);
  LogicalRegion output_lr_1 = runtime->create_logical_region(ctx, is, input_fs);
  LogicalPartition output_lp_1 = runtime->get_logical_partition(ctx, output_lr_1, ip);
  
  LogicalPartition file_checkpoint_lp_input_1 = runtime->get_logical_partition(ctx, input_lr_1, file_ip);
  LogicalPartition file_recover_lp_output_1 = runtime->get_logical_partition(ctx, output_lr_1, file_ip);
  
  // **************** create hdf5 files *****************
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0) {   
    std::map<FieldID, std::string> field_string_map_1;
    field_string_map_1[FID_X] = "FID_X";
    field_string_map_1[FID_Y] = "FID_Y";
  
    HDF5File checkpoint_file(file_name, num_files);
    checkpoint_file.add_logical_region(input_lr_1, file_checkpoint_lp_input_1, "input_lr_1", field_string_map_1);
    for (int i = 0; i < num_files; i++) {
      checkpoint_file.generate_hdf5_file(i);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // **************** init task *************************
  FutureMap fumap;
  ArgumentMap arg_map;
  {
  IndexLauncher init_launcher(INIT_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);  
  init_launcher.add_region_requirement(
        RegionRequirement(input_lp_1, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr_1));
  init_launcher.region_requirements[0].add_field(FID_X);
  fumap = runtime->execute_index_space(ctx, init_launcher);
  fumap.wait_all_results();
  }
  {
  IndexLauncher init_launcher(INIT_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);  
  init_launcher.add_region_requirement(
        RegionRequirement(input_lp_1, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, input_lr_1));
  init_launcher.region_requirements[0].add_field(FID_Y);
  fumap = runtime->execute_index_space(ctx, init_launcher);
  fumap.wait_all_results();
  }
  
  runtime->issue_execution_fence(ctx);

#if 0  
  TaskLauncher output_launcher(CHECK_TASK_ID, 
                              TaskArgument(NULL, 0));
  output_launcher.add_region_requirement(
        RegionRequirement(input_lr_1, 
                          READ_ONLY, EXCLUSIVE, input_lr_1));
  output_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, output_launcher);
#endif 
  // ************************************ checkpoint ****************
  std::map<FieldID, std::string> field_string_map;
  field_string_map[FID_X] = "FID_X";
  field_string_map[FID_Y] = "FID_Y";

  
  //CheckpointIndexLauncher checkpoint_launcher(file_is, TaskArgument(&task_arg, sizeof(task_arg)), arg_map);
  CheckpointIndexLauncher checkpoint_launcher(file_is, file_name, field_string_map);    
  checkpoint_launcher.add_region_requirement(
        RegionRequirement(file_checkpoint_lp_input_1, 0/*projection ID*/, 
                          READ_ONLY, EXCLUSIVE, input_lr_1));
  checkpoint_launcher.region_requirements[0].add_field(FID_X);
  checkpoint_launcher.region_requirements[0].add_field(FID_Y);
  
  fumap = runtime->execute_index_space(ctx, checkpoint_launcher);
  fumap.wait_all_results();

  runtime->issue_execution_fence(ctx);
  
  // ************************************ restart ****************
 // RecoverIndexLauncher restart_launcher(file_is, TaskArgument(&task_arg, sizeof(task_arg)), arg_map); 
  RecoverIndexLauncher recover_launcher(file_is, file_name, field_string_map);  
  recover_launcher.add_region_requirement(
        RegionRequirement(file_recover_lp_output_1, 0/*projection ID*/, 
                          WRITE_DISCARD, EXCLUSIVE, output_lr_1));
  recover_launcher.region_requirements[0].add_field(FID_X);
  recover_launcher.region_requirements[0].add_field(FID_Y);

  fumap = runtime->execute_index_space(ctx, recover_launcher);
  fumap.wait_all_results();
  runtime->issue_execution_fence(ctx);
  
  
  // *************************** check result ********************  
  {
    IndexLauncher check_launcher(CHECK_TASK_ID, color_is, 
                                TaskArgument(NULL, 0), arg_map);
    check_launcher.add_region_requirement(
          RegionRequirement(output_lp_1, 0/*projection ID*/, 
                            READ_ONLY, EXCLUSIVE, output_lr_1));
    check_launcher.region_requirements[0].add_field(FID_X);
    fumap = runtime->execute_index_space(ctx, check_launcher);
    fumap.wait_all_results();
  }
  {
    IndexLauncher check_launcher(CHECK_TASK_ID, color_is, 
                                TaskArgument(NULL, 0), arg_map);
    check_launcher.add_region_requirement(
          RegionRequirement(output_lp_1, 0/*projection ID*/, 
                            READ_ONLY, EXCLUSIVE, output_lr_1));
    check_launcher.region_requirements[0].add_field(FID_Y);
    fumap = runtime->execute_index_space(ctx, check_launcher);
    fumap.wait_all_results();
  }
  runtime->issue_execution_fence(ctx);

  runtime->destroy_logical_region(ctx, input_lr_1);
  runtime->destroy_logical_region(ctx, output_lr_1);
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
  printf("Initializing field %d for block %d %d, pid %d\n", fid, point, getpid());

  const FieldAccessor<WRITE_DISCARD,double,2> acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++) {
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
  const FieldAccessor<READ_ONLY,double,2> acc_x(regions[0], fid);

  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++)
  {
    double received = acc_x[*pir];
    assert(received == (0.29 + point));
  //  printf("%f ", received);
  }
  printf("Success\n");
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
  
  RecoverIndexLauncher::register_task();
  
  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}


