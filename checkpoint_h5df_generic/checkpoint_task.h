#ifndef __CHECKPOINT_TASK_H__
#define __CHECKPOINT_TASK_H__

#include "legion.h"

using namespace Legion;

struct task_args_s{
  size_t field_map_size; 
  char field_map_serial[4096];
  char file_name[32];
};

class CheckpointIndexLauncher : public IndexLauncher
{
public:
  CheckpointIndexLauncher(IndexSpace launchspace, TaskArgument task_arg, ArgumentMap map);
  CheckpointIndexLauncher(IndexSpace launchspace, const char* file_name, std::map<FieldID, std::string> &field_string_map);
  CheckpointIndexLauncher(IndexSpace launchspace, std::string file_name, std::map<FieldID, std::string> &field_string_map);

public:
  struct task_args_s task_argument;
  
public:
  static void cpu_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void register_task(void);
  
public:
  static const char* const TASK_NAME;
  static const int TASK_ID = 198;
};

class RecoverIndexLauncher : public IndexLauncher
{
public:
  RecoverIndexLauncher(IndexSpace launchspace, TaskArgument task_arg, ArgumentMap map);
  RecoverIndexLauncher(IndexSpace launchspace, const char* file_name, std::map<FieldID, std::string> &field_string_map);
  RecoverIndexLauncher(IndexSpace launchspace, std::string file_name, std::map<FieldID, std::string> &field_string_map);

public:
  struct task_args_s task_argument;
  
public:
  static void cpu_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void register_task(void);
  
public:
  static const char* const TASK_NAME;
  static const int TASK_ID = 199;
};

class HDF5LogicalRegion
{
public:
  HDF5LogicalRegion(LogicalRegion lr, std::string lr_name, std::map<FieldID, std::string> &field_string_map);
public:
  LogicalRegion logical_region;
  std::string logical_region_name;
  std::map<FieldID, std::string> field_string_map;
  size_t dim_size[3];
};

class HDF5File
{
public:
  HDF5File(const char* file_name, int num_files);
  HDF5File(std::string file_name, int num_files);
  void add_logical_region(LogicalRegion lr, std::string lr_name, std::map<FieldID, std::string> field_string_map);
  bool generate_hdf5_file(int file_idx);
  
public:
  std::string file_name;
  int num_files;
  std::vector<HDF5LogicalRegion> logical_region_vector;
};

bool generate_hdf_file(const char *file_name, bool new_file, std::map<FieldID, std::string> &field_string_map, int num_elements);

#endif