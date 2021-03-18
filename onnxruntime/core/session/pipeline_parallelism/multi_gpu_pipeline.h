// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_cxx_api.h>
#include "task_thread_pool.h"
#include "concurrent_queue.h"
#include <vector>

namespace onnxruntime {
namespace experimental {
using ReqId = int64_t;

// Used to hold OrtValue* for convenient memory deallocation.
// Unlike Ort::Value, this gives the flexibility to not own the ptr which is useful
// to own input OrtValue*'s.
class OrtValueHandle {
 public:
  OrtValueHandle() = default;
  OrtValueHandle(OrtValue* ort_val0, bool owns0 = true) : ort_val(ort_val0), owns(owns0) {}
  ~OrtValueHandle() {
    if (owns && ort_val) {
      Ort::OrtRelease(ort_val);
      ort_val = nullptr;
    }
  }
  operator OrtValue*() { return ort_val; }
  operator const OrtValue*() const { return ort_val; }
  OrtValueHandle(const OrtValueHandle&) = delete;
  OrtValueHandle& operator=(const OrtValueHandle&) = delete;
  OrtValueHandle(OrtValueHandle&& v) noexcept : ort_val{v.ort_val}, owns{v.owns} { v.ort_val = nullptr; }
  void operator=(OrtValueHandle&& v) noexcept {
    Ort::OrtRelease(ort_val);
    ort_val = v.ort_val;
    owns = v.owns;
    v.ort_val = nullptr;
  }

  ::Ort::TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const {
    ::OrtTensorTypeAndShapeInfo* output;
    ::Ort::ThrowOnError(::Ort::GetApi().GetTensorTypeAndShape(ort_val, &output));
    return ::Ort::TensorTypeAndShapeInfo{output};
  }

  template <typename T>
  const T* GetTensorData() const {
    T* out;
    Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(ort_val, (void**)&out));
    return out;
  }

  OrtValue* release() {
    OrtValue* p = ort_val;
    ort_val = nullptr;
    return p;
  }

 private:
  OrtValue* ort_val{};
  bool owns{};
};

// intentionally left it as a POD for convenience
struct ModelEnsembleConfig {
  struct ModelConfig {
    std::string model_name;
    std::string model_file_path;
    std::vector<std::string> input_names;                                       // same order as model
    std::vector<std::string> output_names;                                      // same order as model
    std::unordered_map<std::string, std::string> inter_stage_output_input_map;  // maps output of this step to input of the next step in the pipeline
    // past_input_names and present_output_names should have 1-1 correspondence
    std::vector<std::string> past_input_names;      // names of inputs whose values come from the previous output
    std::vector<std::string> present_output_names;  // names of outputs that feed the next inputs
    int device_id;
    int batch_dim_index_in_state;
    int batch_dim_index_in_input;
    int seq_len_dim_index_in_state;
    std::string input_to_use_for_seq_len;
    int seq_len_dim_index_in_input;
    int batch_dim_in_inter_stage_output;
    int seq_len_dim_in_inter_stage_output;
  };

  int max_seq_len;
  int eos_token;
  int num_stages;  // = model_config_vec.size()
  std::string input_ids_name;
  std::string position_ids_name;
  std::string logits_name;
  std::vector<ModelConfig> model_config_vec;
  std::unordered_map<std::string, int> model_idx_map;  // maps model name to index in models vector
};

// Represents one ORT req in the batch.
// intentionally left it as a POD for convenience
struct OrtReq {
  std::vector<std::string> input_names;
  std::vector<OrtValue*> input_values;
};

// Represents one ORT resp on the batch.
// intentionally left it as a POD for convenience
struct OrtResp {
  std::vector<std::string> output_names;
  std::vector<OrtValue*> output_values;              // can be null if output_mem_info is non-null
  std::vector<const OrtMemoryInfo*> output_meminfo;  // specify location of outputs or null for preallocated
};

// Represents what flows between pipeline stages
// intentionally left it as a POD for convenience
struct Token {
  void Clear() {
    ort_value_names.clear();
    ort_values.clear();
    error_msg.clear();
    req_id = -1;
    step_id = -2;
  }

  void Init(ReqId req_id0, int step_id0, const std::vector<std::string>& ort_value_names0,
            std::vector<OrtValueHandle>&& ort_values0) {
    req_id = req_id0;
    step_id = step_id0;
    ort_value_names = ort_value_names0;
    ort_values = std::move(ort_values0);
  }

  ReqId req_id;
  int step_id;
  std::vector<std::string> ort_value_names;
  std::vector<OrtValueHandle> ort_values;
  std::string error_msg;
};

class PipelineSession;
struct RequestExecutionFrame {
  RequestExecutionFrame(PipelineSession& psess,
                        int req_idx0,
                        ReqId req_id0,
                        int batch_size0,
                        const int orig_input_seq_len0,
                        int stage_id0,
                        OrtResp& ort_resp0);

  struct RunState {
    // needs to be stored per model since it's associated with a session
    std::unique_ptr<Ort::IoBinding> io_binding;
    // needs to be stored per model since it's associated with a session
    // storing it here as opposed to ModelEnsembleConfig since it may not be thread-safe when multiple requests
    // are getting executed in parallel
    std::unique_ptr<Ort::Allocator> cuda_allocator;
    std::unordered_map<std::string, OrtValueHandle> output_val_map;  // (present_00..) output generated after running a stage
    // pre-allocated on cuda; order should be same as ModelConfig::present_output_names/past_input_names
    std::vector<Ort::MemoryAllocation> present_past_prealloc_buffer_1_vec;
    std::vector<Ort::MemoryAllocation> present_past_prealloc_buffer_2_vec;
    // pre-allocated on cuda
    std::unordered_map<std::string, Ort::MemoryAllocation> inter_stage_output_prealloc_buffer_map;
  };

  const int req_index;  // index of the request in the batch
  const ReqId req_id;   // request to be executed
  const int batch_size;
  const int orig_input_seq_len;
  int stage_id;  // stage to be executed; stage_id can be used as an index into model_run_state_vec
  std::vector<RunState> model_run_state_vec;

  struct NextStepInput {
    std::vector<int64_t> data;  // TODO don't hardcode the type
    std::vector<int64_t> shape;
  };
  // "input_ids" gets changed in the next iteration (step>0); this is used to hold the buffer
  // in scope
  std::unordered_map<std::string, NextStepInput> next_step_input_buffer_map;

  OrtResp& ort_resp;
  Token token;
};

class PipelineSession {
 public:
  PipelineSession(const std::string& ensemble_config_file_path, const OrtEnv& env);
  OrtStatus* Run(const std::vector<OrtReq>& req_vec, std::vector<OrtResp>& resp_vec, int num_steps);

 private:
  using ResponseQueue = ConcurrentQueue<Token*>;
  OrtStatus* CopyFinalOutput(Token& token, OrtResp& ort_resp);
  OrtStatus* HandleAndReturnFailure(const char* error_msg);
  void SetupAndScheduleAllRequestsToStage0(const std::vector<OrtReq>& req_list,
                                           std::vector<OrtResp>& resp_list,
                                           std::unordered_map<ReqId, RequestExecutionFrame>& req_frame_map,
                                           ResponseQueue& resp_queue);
  struct PerModelSessionInfo;
  void ThreadWorkerFn(const ModelEnsembleConfig::ModelConfig& mcfg,
                      const Ort::MemoryInfo& cuda_mem_info,
                      ResponseQueue& resp_queue,
                      Token& token,
                      Ort::Session& session,
                      RequestExecutionFrame& exec_frame);
  OrtStatus* ProcessResponses(int num_reqs,
                              int num_steps,
                              std::unordered_map<ReqId, RequestExecutionFrame>& req_frame_map,
                              ResponseQueue& resp_queue,
                              std::vector<OrtResp>& resp_list);
  void ParseEnsembleFile(const std::string& ensemble_config_file_path, ModelEnsembleConfig& ens);

  PipelineSession(const ModelEnsembleConfig& ens, const OrtEnv& env);
  void Init(const OrtEnv& env, ModelEnsembleConfig& ens0);
  void Validate(const ModelEnsembleConfig& ens);

  struct PerModelSessionInfo {
    Ort::Session session;
    // needs to be stored per model since it's associated with device id
    Ort::MemoryInfo cuda_mem_info;
  };

  struct PipelineStageRunner {
    PipelineStageRunner(int device_id0, int tp_size0 = 1)
        : device_id(device_id0),
          tp_size(tp_size0),
          tp(tp_size0) {
    }

    void ScheduleTask(std::function<void()>&& task) {
      tp.RunTask(std::move(task));
    }

    void DrainAllInflightRequests() {
      tp.WaitWorkComplete();
    }

    int device_id;
    int tp_size;
    TaskThreadPool tp;  // chose this threadpool for simplicity; we can always use a different one if the situation demands
  };

  ModelEnsembleConfig model_ensemble_cfg;
  std::vector<PerModelSessionInfo> per_model_session_info_vec;        // indices correspond to model_ensemble_cfg.model_config_vec
  std::vector<std::unique_ptr<PipelineStageRunner>> pipeline_stages;  // indices correspond to model_ensemble_cfg.model_config_vec
  friend struct RequestExecutionFrame;
};
}  // namespace experimental
}  // namespace onnxruntime
