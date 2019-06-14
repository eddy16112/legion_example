#include <fstream>
#include <hdf5.h>
#include "checkpoint_task.h"

bool is_file_exist (char* file_name) {
    std::ifstream f(file_name);
    return f.good();
}

bool generate_hdf_file(const char *file_name, bool new_file, std::map<FieldID, std::string> &field_string_map, int num_elements)
{
  hid_t file_id;
  if (new_file) {
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
  } else {
    file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
  }
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
  H5Fflush(file_id, H5F_SCOPE_LOCAL);
  H5Fclose(file_id);
  return true;
}

CheckpointIndexLauncher::CheckpointIndexLauncher(IndexSpace launchspace, TaskArgument task_arg, ArgumentMap map)
  : IndexLauncher()
{
  task_id = CheckpointIndexLauncher::TASK_ID;
  launch_space = launchspace;
  global_arg = task_arg;
}

CheckpointIndexLauncher::CheckpointIndexLauncher(IndexSpace launchspace, const char* file_name, std::vector<std::map<FieldID, std::string>> &field_string_map_vector, bool attach_file)
  : IndexLauncher()
{
  strcpy(task_argument.file_name, file_name);
  
  Realm::Serialization::DynamicBufferSerializer dbs(0);
  dbs << field_string_map_vector;
  task_argument.field_map_size = dbs.bytes_used();
  if (task_argument.field_map_size > SERIALIZATION_BUFFER_SIZE) {
    assert(0);
  }
  memcpy(task_argument.field_map_serial, dbs.detach_buffer(), task_argument.field_map_size);
  task_argument.attach_file_flag = attach_file;
  
  task_id = CheckpointIndexLauncher::TASK_ID;
  launch_space = launchspace;
  global_arg = TaskArgument(&task_argument, sizeof(task_argument));
//  printf("filename %s, task_arg size %ld\n", task_argument.file_name, global_arg.get_size()); 
}

CheckpointIndexLauncher::CheckpointIndexLauncher(IndexSpace launchspace, std::string file_name, std::vector<std::map<FieldID, std::string>> &field_string_map_vector, bool attach_file)
  : CheckpointIndexLauncher(launchspace, file_name.c_str(), field_string_map_vector, attach_file)
{ 
}

/*static*/ 
const char * const CheckpointIndexLauncher::TASK_NAME = "checkpoint";

void CheckpointIndexLauncher::cpu_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  
  if (task_arg.attach_file_flag == true) {
    CheckpointIndexLauncher::attach_impl(task, regions, ctx, runtime);
  } else {
    CheckpointIndexLauncher::no_attach_impl(task, regions, ctx, runtime);
  }
}

void CheckpointIndexLauncher::attach_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  
  const int point = task->index_point.point_data[0];
  
  std::vector<std::map<FieldID, std::string>> field_string_map_vector;
  Realm::Serialization::FixedBufferDeserializer fdb(task_arg.field_map_serial, task_arg.field_map_size);
  bool ok  = fdb >> field_string_map_vector;
  if(!ok) {
    printf("task args deserializer error\n");
  }
  
  std::string fname(task_arg.file_name);
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  
  
  for (unsigned int rid = 0; rid < regions.size(); rid++) {
    PhysicalRegion cp_pr;
    LogicalRegion input_lr = regions[rid].get_logical_region();
    LogicalRegion cp_lr = runtime->create_logical_region(ctx, input_lr.get_index_space(), input_lr.get_field_space());
  
    AttachLauncher hdf5_attach_launcher(EXTERNAL_HDF5_FILE, cp_lr, cp_lr);
    std::map<FieldID,const char*> field_map;
    std::set<FieldID> field_set = task->regions[rid].privilege_fields;  
    std::map<FieldID, std::string>::iterator map_it;
    for (std::set<FieldID>::iterator it = field_set.begin() ; it != field_set.end(); ++it) {
      map_it = field_string_map_vector[rid].find(*it);
      if (map_it != field_string_map_vector[rid].end()) {
        field_map.insert(std::make_pair(*it, (map_it->second).c_str()));
      } else {
        assert(0);
      }
    }
    printf("Checkpointing data to HDF5 file attach '%s' region %d, (datasets='%ld'), vector size %ld\n", file_name, rid, field_map.size(), field_string_map_vector.size());
    hdf5_attach_launcher.attach_hdf5(file_name, field_map, LEGION_FILE_READ_WRITE);
    cp_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);
   // cp_pr.wait_until_valid();

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
}

void CheckpointIndexLauncher::no_attach_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  
  const int point = task->index_point.point_data[0];
  
  std::vector<std::map<FieldID, std::string>> field_string_map_vector;
  Realm::Serialization::FixedBufferDeserializer fdb(task_arg.field_map_serial, task_arg.field_map_size);
  bool ok  = fdb >> field_string_map_vector;
  if(!ok) {
    printf("task args deserializer error\n");
  }
  
  std::string fname(task_arg.file_name);
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  
  hid_t file_id;
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT); 
  if(file_id < 0) {
    printf("H5Fopen failed: %lld\n", (long long)file_id);
    assert(0);
  }
  
  for (unsigned int rid = 0; rid < regions.size(); rid++) {
    LogicalRegion input_lr = regions[rid].get_logical_region();

    std::set<FieldID> field_set = task->regions[rid].privilege_fields;  
    std::map<FieldID, std::string>::iterator map_it;
    for (std::set<FieldID>::iterator it = field_set.begin() ; it != field_set.end(); ++it) {
      map_it = field_string_map_vector[rid].find(*it);
      if (map_it != field_string_map_vector[rid].end()) {
        const FieldAccessor<READ_ONLY,double,1,coord_t, Realm::AffineAccessor<double,1,coord_t> > acc_fid(regions[rid], *it);
        Rect<1> rect = runtime->get_index_space_domain(ctx, task->regions[rid].region.get_index_space());
        const double *dset_data = acc_fid.ptr(rect.lo);
        hid_t dataset_id = H5Dopen2(file_id, (map_it->second).c_str(), H5P_DEFAULT);
        if(dataset_id < 0) {
          printf("H5Dopen2 failed: %lld\n", (long long)dataset_id);
          H5Fclose(file_id);
          assert(0);
        }
        H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
        H5Dclose(dataset_id);
      } else {
        assert(0);
      }
    }
    printf("Checkpointing data to HDF5 file no attach '%s' region %d, (datasets='%ld'), vector size %ld\n", file_name, rid, field_set.size(), field_string_map_vector.size());
  }
  
  H5Fflush(file_id, H5F_SCOPE_LOCAL);
  H5Fclose(file_id);
}

/*static*/
void CheckpointIndexLauncher::register_task(void)
{
  TaskVariantRegistrar registrar(CheckpointIndexLauncher::TASK_ID, CheckpointIndexLauncher::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<cpu_impl>(registrar, CheckpointIndexLauncher::TASK_NAME);
}

RecoverIndexLauncher::RecoverIndexLauncher(IndexSpace launchspace, TaskArgument task_arg, ArgumentMap map)
  : IndexLauncher()
{
  task_id = RecoverIndexLauncher::TASK_ID;
  launch_space = launchspace;
  global_arg = task_arg;
}

RecoverIndexLauncher::RecoverIndexLauncher(IndexSpace launchspace, const char* file_name, std::vector<std::map<FieldID, std::string>> &field_string_map_vector, bool attach_file)
  : IndexLauncher()
{
  strcpy(task_argument.file_name, file_name);
  
  Realm::Serialization::DynamicBufferSerializer dbs(0);
  dbs << field_string_map_vector;
  task_argument.field_map_size = dbs.bytes_used();
  if (task_argument.field_map_size > SERIALIZATION_BUFFER_SIZE) {
    assert(0);
  }
  memcpy(task_argument.field_map_serial, dbs.detach_buffer(), task_argument.field_map_size);
  task_argument.attach_file_flag = attach_file;
  
  task_id = RecoverIndexLauncher::TASK_ID;
  launch_space = launchspace;
  global_arg = TaskArgument(&task_argument, sizeof(task_argument));
//  printf("filename %s, task_arg size %ld\n", task_argument.file_name, global_arg.get_size()); 
}

RecoverIndexLauncher::RecoverIndexLauncher(IndexSpace launchspace, std::string file_name, std::vector<std::map<FieldID, std::string>> &field_string_map_vector, bool attach_file)
  : RecoverIndexLauncher(launchspace, file_name.c_str(), field_string_map_vector, attach_file)
{ 
}

/*static*/ 
const char * const RecoverIndexLauncher::TASK_NAME = "recover";

void RecoverIndexLauncher::cpu_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  
  if (task_arg.attach_file_flag == true) {
    RecoverIndexLauncher::attach_impl(task, regions, ctx, runtime);
  } else {
    RecoverIndexLauncher::no_attach_impl(task, regions, ctx, runtime);
  }
}

void RecoverIndexLauncher::attach_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  const int point = task->index_point.point_data[0];
  
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  std::vector<std::map<FieldID, std::string>> field_string_map_vector;
  Realm::Serialization::FixedBufferDeserializer fdb(task_arg.field_map_serial, task_arg.field_map_size);
  bool ok  = fdb >> field_string_map_vector;
  if(!ok) {
    printf("task args deserializer error\n");
  }
  
  std::string fname(task_arg.file_name);
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  
  for (unsigned int rid = 0; rid < regions.size(); rid++) {
    PhysicalRegion restart_pr;
    LogicalRegion input_lr2 = regions[rid].get_logical_region();
    LogicalRegion restart_lr = runtime->create_logical_region(ctx, input_lr2.get_index_space(), input_lr2.get_field_space());

    AttachLauncher hdf5_attach_launcher(EXTERNAL_HDF5_FILE, restart_lr, restart_lr);
    std::map<FieldID,const char*> field_map;
    std::set<FieldID> field_set = task->regions[rid].privilege_fields;  
    std::map<FieldID, std::string>::iterator map_it;
    for (std::set<FieldID>::iterator it = field_set.begin() ; it != field_set.end(); ++it) {
      map_it = field_string_map_vector[rid].find(*it);
      if (map_it != field_string_map_vector[rid].end()) {
        field_map.insert(std::make_pair(*it, (map_it->second).c_str()));
      } else {
        assert(0);
      }
    }
    printf("Recoverring data to HDF5 file attach '%s' region %d, (datasets='%ld'), vector size %ld\n", file_name, rid, field_map.size(), field_string_map_vector.size());
    hdf5_attach_launcher.attach_hdf5(file_name, field_map, LEGION_FILE_READ_WRITE);
    restart_pr = runtime->attach_external_resource(ctx, hdf5_attach_launcher);

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
}

void RecoverIndexLauncher::no_attach_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  const int point = task->index_point.point_data[0];
  
  struct task_args_s task_arg = *(struct task_args_s *) task->args;
  std::vector<std::map<FieldID, std::string>> field_string_map_vector;
  Realm::Serialization::FixedBufferDeserializer fdb(task_arg.field_map_serial, task_arg.field_map_size);
  bool ok  = fdb >> field_string_map_vector;
  if(!ok) {
    printf("task args deserializer error\n");
  }
  
  std::string fname(task_arg.file_name);
  fname = fname + std::to_string(point);
  char *file_name = const_cast<char*>(fname.c_str());
  
  hid_t file_id;
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT); 
  if(file_id < 0) {
    printf("H5Fopen failed: %lld\n", (long long)file_id);
    assert(0);
  }
  
  for (unsigned int rid = 0; rid < regions.size(); rid++) {
    LogicalRegion input_lr2 = regions[rid].get_logical_region();

    std::set<FieldID> field_set = task->regions[rid].privilege_fields;  
    std::map<FieldID, std::string>::iterator map_it;
    for (std::set<FieldID>::iterator it = field_set.begin() ; it != field_set.end(); ++it) {
      map_it = field_string_map_vector[rid].find(*it);
      if (map_it != field_string_map_vector[rid].end()) {
        const FieldAccessor<WRITE_DISCARD,double,1,coord_t, Realm::AffineAccessor<double,1,coord_t> > acc_fid(regions[rid], *it);
        Rect<1> rect = runtime->get_index_space_domain(ctx, task->regions[rid].region.get_index_space());
        double *dset_data = acc_fid.ptr(rect.lo);
        hid_t dataset_id = H5Dopen2(file_id, (map_it->second).c_str(), H5P_DEFAULT);
        if(dataset_id < 0) {
          printf("H5Dopen2 failed: %lld\n", (long long)dataset_id);
          H5Fclose(file_id);
          assert(0);
        }
        H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
        H5Dclose(dataset_id);
      } else {
        assert(0);
      }
    }
    printf("Recoverring data to HDF5 file no attach '%s' region %d, (datasets='%ld'), vector size %ld\n", file_name, rid, field_set.size(), field_string_map_vector.size());
  }
}

/*static*/
void RecoverIndexLauncher::register_task(void)
{
  TaskVariantRegistrar registrar(RecoverIndexLauncher::TASK_ID, RecoverIndexLauncher::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<cpu_impl>(registrar, RecoverIndexLauncher::TASK_NAME);
}

HDF5LogicalRegion::HDF5LogicalRegion(LogicalRegion lr, LogicalPartition lp, std::string lr_name, std::map<FieldID, std::string> &field_string_map)
  :logical_region(lr), logical_partition(lp), logical_region_name(lr_name), field_string_map(field_string_map)
{
  Runtime *runtime = Runtime::get_runtime();
  Context ctx = Runtime::get_context();
  if (lr.get_dim() == 1) {
    Domain domain = runtime->get_index_space_domain(ctx, lr.get_index_space());
    dim_size[0] = domain.get_volume();
    printf("ID logical region size %ld\n", dim_size[0]);
  } else {
    Domain domain = runtime->get_index_space_domain(ctx, lr.get_index_space());
    dim_size[0] = domain.get_volume();
    printf("2D ID logical region size %ld\n", dim_size[0]);
  }
}

HDF5File::HDF5File(const char* file_name, int num_files)
  :HDF5File(std::string(file_name), num_files)
{
}

HDF5File::HDF5File(std::string file_name, int num_files)
  :file_name(file_name), num_files(num_files)
{
  logical_region_vector.clear();  
}

void HDF5File::add_logical_region(LogicalRegion lr, LogicalPartition lp, std::string lr_name, std::map<FieldID, std::string> field_string_map)
{
  HDF5LogicalRegion h5_lr(lr, lp, lr_name, field_string_map);
  logical_region_vector.push_back(h5_lr);
}

bool HDF5File::generate_hdf5_file(int file_idx)
{
  hid_t file_id;

  std::string fname = file_name + std::to_string(file_idx);
  file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
  if(file_id < 0) {
    printf("H5Fcreate failed: %lld\n", (long long)file_id);
    return false;
  }
  
  Runtime *runtime = Runtime::get_runtime();
  Context ctx = Runtime::get_context();

  for (std::vector<HDF5LogicalRegion>::iterator lr_it = logical_region_vector.begin(); lr_it != logical_region_vector.end(); ++lr_it) {
    hid_t dataspace_id = -1;
    if ((*lr_it).logical_region.get_index_space().get_dim() == 1) {
      LogicalRegion sub_lr = runtime->get_logical_subregion_by_color(ctx, (*lr_it).logical_partition, file_idx);
      Domain domain = runtime->get_index_space_domain(ctx, sub_lr.get_index_space());
      hsize_t dims[1];
      dims[0] = domain.get_volume();
      dataspace_id = H5Screate_simple(1, dims, NULL);
    } else {
      LogicalRegion sub_lr = runtime->get_logical_subregion_by_color(ctx, (*lr_it).logical_partition, file_idx);
      Domain domain = runtime->get_index_space_domain(ctx, sub_lr.get_index_space());
      hsize_t dims[1];
      dims[0] = domain.get_volume();
      dataspace_id = H5Screate_simple(1, dims, NULL);
    }
    if(dataspace_id < 0) {
      printf("H5Screate_simple failed: %lld\n", (long long)dataspace_id);
      H5Fclose(file_id);
      return false;
    }
#if 0
    hid_t group_id = H5Gcreate2(file_id, (*lr_it).logical_region_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
      printf("H5Gcreate2 failed: %lld\n", (long long)group_id);
      H5Sclose(dataspace_id);
      H5Fclose(file_id);
      return false;
    }
#endif
    for (std::map<FieldID, std::string>::iterator it = (*lr_it).field_string_map.begin() ; it != (*lr_it).field_string_map.end(); ++it) {
      const char* dataset_name = (it->second).c_str();
      hid_t dataset = H5Dcreate2(file_id, dataset_name,
    			     H5T_IEEE_F64LE, dataspace_id,
    			     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if(dataset < 0) {
        printf("H5Dcreate2 failed: %lld\n", (long long)dataset);
    //    H5Gclose(group_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
      }
      H5Dclose(dataset);
    }
 //   H5Gclose(group_id);
    H5Sclose(dataspace_id);
  }
  H5Fflush(file_id, H5F_SCOPE_LOCAL);
  H5Fclose(file_id);
  return true;
}