#include <hdf5.h>
#include "checkpoint_task.h"

bool generate_hdf_file(const char *file_name, std::map<FieldID, std::string> &field_string_map, int num_elements)
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
  for (std::map<FieldID, std::string>::iterator it = field_string_map.begin() ; it != field_string_map.end(); ++it) {
    const char* dataset_name = (it->second).c_str();
    hid_t dataset = H5Dcreate2(file_id, dataset_name,
  			     H5T_IEEE_F64LE, dataspace_id,
  			     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(dataset < 0) {
      printf("H5Dcreate2 failed: %lld\n", (long long)dataset);
      H5Sclose(dataspace_id);
      H5Fclose(file_id);
      return false;
    }
    H5Dclose(dataset);
  }

  // close things up - attach will reopen later
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
  return true;
}

CheckpointIndexLauncher::CheckpointIndexLauncher(IndexSpace launch_space, TaskArgument global_arg, ArgumentMap map)
  : IndexLauncher(CheckpointIndexLauncher::TASK_ID, launch_space, global_arg, map)
{
  
}

/*static*/ 
const char * const CheckpointIndexLauncher::TASK_NAME = "checkpoint";

void CheckpointIndexLauncher::cpu_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
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
  
  
  PhysicalRegion cp_pr;
  Rect<1> rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  LogicalRegion input_lr = regions[0].get_logical_region();
  LogicalRegion cp_lr = runtime->create_logical_region(ctx, input_lr.get_index_space(), input_lr.get_field_space());
  
  // create the HDF5 file first - attach wants it to already exist
  ok = generate_hdf_file(file_name, field_string_map, rect.volume());
  assert(ok);
  AttachLauncher hdf5_attach_launcher(EXTERNAL_HDF5_FILE, cp_lr, cp_lr);
  std::map<FieldID,const char*> field_map;
  for (std::map<FieldID, std::string>::iterator it = field_string_map.begin() ; it != field_string_map.end(); ++it) {
    field_map.insert(std::make_pair(it->first, (it->second).c_str()));
  }
  printf("Checkpointing data to HDF5 file '%s' (dataset='%ld')\n", file_name, field_map.size());
  hdf5_attach_launcher.attach_hdf5(file_name, field_map, LEGION_FILE_READ_WRITE);
  cp_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);
 // cp_pr.wait_until_valid();

  std::set<FieldID> field_set = task->regions[0].privilege_fields;  
  CopyLauncher copy_launcher1;
  copy_launcher1.add_copy_requirements(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr),
      RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
  for(std::set<FieldID>::iterator it = field_set.begin(); it != field_set.end(); ++it) {
    copy_launcher1.add_src_field(0, *it);
    copy_launcher1.add_dst_field(0, *it);
  }
  runtime->issue_copy_operation(ctx, copy_launcher1);
  
  
  Future fu = runtime->detach_external_resource(ctx, cp_pr, true);
  fu.wait();
  runtime->destroy_logical_region(ctx, cp_lr);
}

/*static*/
void CheckpointIndexLauncher::register_task(void)
{
  TaskVariantRegistrar registrar(CheckpointIndexLauncher::TASK_ID, CheckpointIndexLauncher::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<cpu_impl>(registrar, CheckpointIndexLauncher::TASK_NAME);
}