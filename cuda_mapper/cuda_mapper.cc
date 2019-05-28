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
#include "legion.h"

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

//#define USE_DEFAULT_MAP_TASK
//#define USE_ZEROCOPY_MEM

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_CPU_ID,
  INIT_FIELD_TASK_GPU_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
};

void init_field_task_gpu(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);


class AdversarialMapper : public DefaultMapper {
public:
  AdversarialMapper(Machine machine, 
      Runtime *rt, Processor local);
public:
  void map_region(const MapperContext ctx,
                  LogicalRegion region, Memory target,
                  std::vector<PhysicalInstance> &instances);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                              MapTaskOutput& output);

protected:
  Memory system_mem;
  Memory zerocopy_mem;
  Memory gpu_mem;
  std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance> local_instances;
};

void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new AdversarialMapper(machine, rt, *it), *it);
  }
}

// Here is the constructor for our adversarial mapper.
// We'll use the constructor to illustrate how mappers can
// get access to information regarding the current machine.
AdversarialMapper::AdversarialMapper(Machine m, 
                                     Runtime *rt, Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p) // pass arguments through to TestMapper 
{

  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  
  system_mem = Memory::NO_MEMORY;
  zerocopy_mem = Memory::NO_MEMORY;
  gpu_mem = Memory::NO_MEMORY;

  // Print out how many processors there are and each
  // of their kinds.
  printf("There are %zd processors:\n", all_procs.size());
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    // For every processor there is an associated kind
    Processor::Kind kind = it->kind();
    switch (kind)
    {
      // Latency-optimized cores (LOCs) are CPUs
      case Processor::LOC_PROC:
        {
          printf("  Processor ID " IDFMT " is CPU\n", it->id); 
          break;
        }
      // Throughput-optimized cores (TOCs) are GPUs
      case Processor::TOC_PROC:
        {
          printf("  Processor ID " IDFMT " is GPU\n", it->id);
          break;
        }
      // Processor for doing I/O
      case Processor::IO_PROC:
        {
          printf("  Processor ID " IDFMT " is I/O Proc\n", it->id);
          break;
        }
      // Utility processors are helper processors for
      // running Legion runtime meta-level tasks and 
      // should not be used for running application tasks
      case Processor::UTIL_PROC:
        {
          printf("  Processor ID " IDFMT " is utility\n", it->id);
          break;
        }
      default:
        assert(false);
    }
  }
  // We can also get the list of all the memories available
  // on the target architecture and print out their info.
  std::set<Memory> all_mems;
  machine.get_all_memories(all_mems);
  printf("There are %zd memories:\n", all_mems.size());
  for (std::set<Memory>::const_iterator it = all_mems.begin();
        it != all_mems.end(); it++)
  {
    Memory::Kind kind = it->kind();
    size_t memory_size_in_kb = it->capacity() >> 10;
    switch (kind)
    {
      // RDMA addressable memory when running with GASNet
      case Memory::GLOBAL_MEM:
        {
          printf("  GASNet Global Memory ID " IDFMT " has %zd KB\n", 
                  it->id, memory_size_in_kb);
          break;
        }
      // DRAM on a single node
      case Memory::SYSTEM_MEM:
        {
          if (system_mem == Memory::NO_MEMORY) {
           system_mem = *it;
          }
          printf("  System Memory ID " IDFMT " has %zd KB\n",
                  it->id, memory_size_in_kb);
          break;
        }
      // Pinned memory on a single node
      case Memory::REGDMA_MEM:
        {
          printf("  Pinned Memory ID " IDFMT " has %zd KB\n",
                  it->id, memory_size_in_kb);
          break;
        }
      // Zero-copy memory betweeen CPU DRAM and
      // all GPUs on a single node
      case Memory::Z_COPY_MEM:
        {
          if (zerocopy_mem == Memory::NO_MEMORY) {
           zerocopy_mem = *it;
          }
          printf("  Zero-Copy Memory ID " IDFMT " has %zd KB\n",
                  it->id, memory_size_in_kb);
          break;
        }
      // GPU framebuffer memory for a single GPU
      case Memory::GPU_FB_MEM:
        {
          if (gpu_mem == Memory::NO_MEMORY) {
           gpu_mem = *it;
          }
          printf("  GPU Frame Buffer Memory ID " IDFMT " has %zd KB\n",
                  it->id, memory_size_in_kb);
          break;
        }
      default:
        break;
    }
  }
  
  assert(system_mem != Memory::NO_MEMORY);
  assert(zerocopy_mem != Memory::NO_MEMORY);
  assert(gpu_mem != Memory::NO_MEMORY);
}

void AdversarialMapper::map_task(const MapperContext         ctx,
                                 const Task&                 task,
                                 const MapTaskInput&         input,
                                       MapTaskOutput&        output)
{
#ifndef USE_DEFAULT_MAP_TASK
  if ((task.task_id != TOP_LEVEL_TASK_ID) && (task.task_id != CHECK_TASK_ID))
  {
 //   printf("mapper %p, system_mem " IDFMT " gpu mem " IDFMT "\n", this, system_mem.id, gpu_mem.id);
    Processor::Kind target_kind = task.target_proc.kind();
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
                      true/*needs tight bound*/, true/*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    output.task_priority = 0;
    output.postmap_task = false;
    default_policy_select_target_processors(ctx, task, output.target_procs);

    bool map_to_gpu = task.target_proc.kind() == Processor::TOC_PROC;

    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if ((task.regions[idx].privilege == NO_ACCESS) ||
          (task.regions[idx].privilege_fields.empty())) continue;

      Memory target_memory;
#ifdef USE_ZEROCOPY_MEM
      target_memory = zerocopy_mem
#else      
      if (!map_to_gpu) {
        target_memory = system_mem;
      } else {
        target_memory = gpu_mem;
      }
#endif
      
      printf("mapper %p, task %d, gpu %d, memid " IDFMT"\n", this, task.task_id, map_to_gpu, target_memory.id);
      map_region(ctx, task.regions[idx].region, target_memory, output.chosen_instances[idx]);
    }
    runtime->acquire_instances(ctx, output.chosen_instances);
  } else {
    DefaultMapper::map_task(ctx, task, input, output);
  }
#else
  DefaultMapper::map_task(ctx, task, input, output);
#endif
  
}

void AdversarialMapper::map_region(const MapperContext ctx,
                                   LogicalRegion region, Memory target,
                                   std::vector<PhysicalInstance> &instances)
{
  const std::pair<LogicalRegion,Memory> key(region, target);
  std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance>::const_iterator
    finder = local_instances.find(key);
  if (finder != local_instances.end()) {
    instances.push_back(finder->second);
    return;
  }
  // First time through, then we make an instance
  std::vector<LogicalRegion> regions(1, region);  
  LayoutConstraintSet layout_constraints;
  // No specialization
  layout_constraints.add_constraint(SpecializedConstraint());
  // SOA-Fortran dimension ordering
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_X;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_Z;
  dimension_ordering[3] = DIM_F;
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, 
                                                       false/*contiguous*/));
  // Constrained for the target memory kind
  layout_constraints.add_constraint(MemoryConstraint(target.kind()));
  // Have all the field for the instance available
  std::vector<FieldID> all_fields;
  runtime->get_field_space_fields(ctx, region.get_field_space(), all_fields);
  layout_constraints.add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                                    false/*inorder*/));

  PhysicalInstance result; bool created;
  if (!runtime->find_or_create_physical_instance(ctx, target, layout_constraints,
        regions, result, created, true/*acquire*/, GC_NEVER_PRIORITY)) {
    printf("ERROR: Mapper failed to allocate instance\n");
    assert(false);
  }
  instances.push_back(result);
  // Save the result for future use
  local_instances[key] = result;
}

/*
 * Everything below here is the standard daxpy example
 * except for the registration of the callback function
 * for creating custom mappers which is explicitly commented
 * and the call to select_tunable_value to determine the number
 * of sub-regions.
 */
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 64; 
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }

  printf("Running daxpy for %d elements...\n", num_elements);

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
  }

  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);

  {
    TaskLauncher init_launcher(INIT_FIELD_TASK_CPU_ID, TaskArgument(NULL, 0));
    init_launcher.add_region_requirement(
          RegionRequirement(input_lr, WRITE_DISCARD, EXCLUSIVE, input_lr));
    init_launcher.add_field(0/*idx*/, FID_X);
    runtime->execute_task(ctx, init_launcher);
  }

  {
    TaskLauncher init_launcher(INIT_FIELD_TASK_GPU_ID, TaskArgument(NULL, 0));
    init_launcher.add_region_requirement(
          RegionRequirement(input_lr, READ_WRITE, EXCLUSIVE, input_lr));
    init_launcher.add_field(0/*idx*/, FID_X);
    runtime->execute_task(ctx, init_launcher);
  }
  
  {
    TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(NULL, 0));
    check_launcher.add_region_requirement(
        RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
    check_launcher.region_requirements[0].add_field(FID_X);
    runtime->execute_task(ctx, check_launcher);
  }

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_field_space(ctx, input_fs);
}

void init_field_task_cpu(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("CPU initializing field %d for block %d...\n", fid, point);

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
  int ct = 0;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double received = acc_x[*pir];
    if (ct < 32) {
      assert(received == 0.1);
    } else {
      assert(received == 0.29);
    }
    ct ++;
  }
  printf("Success\n");
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
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_CPU_ID, "init_field_cpu");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task_cpu>(registrar, "init_field_cpu");
  }
  
  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_GPU_ID, "init_field_gpu");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task_gpu>(registrar, "init_field_gpu");
  }
  
  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  // Here is where we register the callback function for 
  // creating custom mappers.
  Runtime::add_registration_callback(mapper_registration);

  return Runtime::start(argc, argv);
}

