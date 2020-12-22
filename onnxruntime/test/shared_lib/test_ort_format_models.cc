// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// if we can't load an ORT format model we can't really test anything
#if defined(ENABLE_ORT_FORMAT_LOAD)

#include "core/common/make_unique.h"
#include "core/graph/constants.h"

//#include "core/framework/data_types.h"
//#include "core/framework/tensorprotoutils.h"
//#include "core/graph/onnx_protobuf.h"
//#include "core/session/inference_session.h"
//#include "core/session/onnxruntime_session_options_config_keys.h"
//#include "core/graph/model.h"
//#include "test/test_environment.h"
//#include "test_utils.h"
//#include "test/util/include/asserts.h"
//#include "test/util/include/inference_session_wrapper.h"
//#include "core/flatbuffers/schema/ort.fbs.h"
//#include "flatbuffers/idl.h"
//#include "flatbuffers/util.h"

#include "core/session/onnxruntime_cxx_api.h"

#include "test_allocator.h"
#include "utils.h"

#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

template <typename OutT>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<Input>& inputs, const char* output_name,
                const std::vector<int64_t>& dims_y, const std::vector<OutT>& values_y,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor<float>(allocator->Info(allocator), const_cast<float*>(inputs[i].values.data()),
                                        inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                       &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                     &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);
    output_tensor = &ort_outputs[0];
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  OutT* f = output_tensor->GetTensorMutableData<OutT>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

template <typename OutT>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::vector<Input>& inputs, const char* output_name,
                          const std::vector<int64_t>& expected_dims_y, const std::vector<OutT>& expected_values_y,
                          OrtCustomOpDomain* custom_op_domain) {
  Ort::SessionOptions session_options;

  session_options.Add(custom_op_domain);

  //if (custom_op_library_filename) {
  //  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(session_options,
  //                                                           custom_op_library_filename, library_handle));
  //}

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  auto default_allocator = onnxruntime::make_unique<MockedOrtAllocator>();

  //without preallocated output tensor
  RunSession<OutT>(default_allocator.get(),
                   session,
                   inputs,
                   output_name,
                   expected_dims_y,
                   expected_values_y,
                   nullptr);
  //with preallocated output tensor
  Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(),
                                                       expected_dims_y.data(), expected_dims_y.size());

  //test it twice
  for (int i = 0; i != 2; ++i)
    RunSession<OutT>(default_allocator.get(),
                     session,
                     inputs,
                     output_name,
                     expected_dims_y,
                     expected_values_y,
                     &value_y);
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(OrtFormatCustomOpTests, ConvertOnnxModelToOrt) {
  const std::basic_string<ORTCHAR_T> onnx_file = ORT_TSTR("testdata/foo_1.onnx");
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/foo_1.onnx.test_output.ort");

  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider};
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  // convert to ort by loading the onnx model
  {
    Ort::SessionOptions so;
    so.Add(custom_op_domain);
    so.SetLogId("CustomOp");
    so.SetOptimizedModelFilePath(ort_file.c_str());

    Ort::Session session(*ort_env, onnx_file.c_str(), so);
  }

  // now load the ORT format model and execute it
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  TestInference<float>(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain);
}
#endif  // if !defined(ORT_MINIMAL_BUILD)

TEST(OrtFormatCustomOpTests, LoadOrtModelInMinimalBuild) {
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/foo_1.onnx.ort");

  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider};
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  // now load the ORT format model and execute it
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  TestInference<float>(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain);
}
#endif  // #if defined(ENABLE_ORT_FORMAT_LOAD)
