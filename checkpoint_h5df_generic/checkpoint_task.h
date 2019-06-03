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
  CheckpointIndexLauncher(IndexSpace launch_space, TaskArgument global_arg, ArgumentMap map);

public:
  static void cpu_impl(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void register_task(void);
  
public:
  static const char* const TASK_NAME;
  static const int TASK_ID = 198;
};

#endif