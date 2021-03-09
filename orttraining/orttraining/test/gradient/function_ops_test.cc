// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "orttraining/core/graph/training_op_defs.h"
#include "test/test_environment.h"

#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#include "test/framework/test_utils.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace test {

typedef std::vector<onnxruntime::NodeArg*> ArgMap;

static ONNX_NAMESPACE::TypeProto makeFloatTensorType(std::vector<int64_t> dims) {
  ONNX_NAMESPACE::TypeProto typeProto;
  typeProto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* shape = typeProto.mutable_tensor_type()->mutable_shape();
  for (auto dim : dims)
    shape->add_dim()->set_dim_value(dim);
  return typeProto;
}

static std::vector<OrtValue>
Run(onnxruntime::Model& model, NameMLValMap& feeds, std::vector<std::string> output_names) {
  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::string serialized_model;
  const bool serialization_status = model.ToProto().SerializeToString(&serialized_model);
  EXPECT_TRUE(serialization_status) << "Failed to serialize proto to string";
  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  EXPECT_TRUE(status.IsOK());
  status = session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, output_names, &fetches);
  EXPECT_TRUE(status.IsOK()) << "Session Run failed.";

  return fetches;
}

void AssertEqual(const Tensor& tensor1, const Tensor& tensor2) {
  auto size = tensor1.Shape().Size();
  auto* data1 = tensor1.template Data<float>();
  auto* data2 = tensor2.template Data<float>();

  float threshold = 0.001f;

  for (int i = 0; i < size; ++i) {
    std::cout << data2[i] << " at " << i << "\n";
    ASSERT_NEAR(data1[i], data2[i], threshold) << "as position i:" << i;
  }
}

struct FunctionTestCase {
  const char* opname;

  std::vector<NodeArg> input_args;
  std::vector<std::pair<std::string, OrtValue>> input_values;
  NameMLValMap input_value_map;

  std::vector<std::string> output_names;
  std::vector<NodeArg> output_args;

  NodeAttributes attributes;
  std::unique_ptr<IExecutionProvider> provider;

  FunctionTestCase(const char* _opname) : opname(_opname), provider(new CPUExecutionProvider(CPUExecutionProviderInfo())) {}

  void AddInput(std::string input_name, std::vector<int64_t> shape, std::vector<float> data) {
    auto arg_type = makeFloatTensorType(shape);
    input_args.emplace_back(input_name, &arg_type);

    OrtValue ort_value;
    CreateMLValue<float>(provider->GetAllocator(0, OrtMemTypeDefault), shape, data, &ort_value);
    input_values.push_back(std::make_pair(input_name, ort_value));
    input_value_map.insert(std::make_pair(input_name, ort_value));
  }

  void AddOutput(std::string output_name) {
    output_names.emplace_back(output_name);
    output_args.emplace_back(output_name, nullptr);
  }

  void AddAttribute(const char* attr_name, int64_t attr_val) {
    ONNX_NAMESPACE::AttributeProto axis_attr;
    axis_attr.set_name(attr_name);
    axis_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    axis_attr.set_i(attr_val);
    attributes[attr_name] = axis_attr;
  }

  onnxruntime::Node& AddCallNodeTo(onnxruntime::Graph& graph) {
    std::vector<NodeArg*> input_arg_ptrs;

    for (auto& arg : input_args)
      input_arg_ptrs.push_back(&arg);

    std::vector<NodeArg*> output_arg_ptrs;
    for (auto& arg : output_args)
      output_arg_ptrs.push_back(&arg);

    return graph.AddNode("fncallnode", opname, "function call node", input_arg_ptrs, output_arg_ptrs, &attributes, onnxruntime::kMSDomain);
  }

  void RunTest() {
    onnxruntime::training::RegisterTrainingOpSchemas();
    onnxruntime::Model model("test", false, DefaultLoggingManager().DefaultLogger());
    onnxruntime::Graph& graph = model.MainGraph();

    auto& call_node = AddCallNodeTo(graph);
    ASSERT_TRUE(graph.Resolve().IsOK());

    std::cout << graph << std::endl;
    auto results1 = Run(model, input_value_map, output_names);

    // Now, inline function body.
    graph.InlineFunction(call_node);
    ASSERT_TRUE(graph.Resolve().IsOK());

    std::cout << graph << std::endl;
    auto results2 = Run(model, input_value_map, output_names);

    ASSERT_EQ(results1.size(), results2.size());
    for (int i = 0; i < results1.size(); i++) {
      auto& value1 = results1[i].Get<Tensor>();
      auto& value2 = results2[i].Get<Tensor>();
      AssertEqual(value1, value2);
    }
  }
};

static void InitSoftmaxGradTestCase(FunctionTestCase& testCase, std::vector<int64_t> shape) {
  int64_t size = 1;
  for (auto dim : shape)
    size *= dim;

  std::vector<float> value(size);
  for (int64_t i = 0; i < size; i++)
    value[i] = float(i);

  testCase.AddInput("dY", shape, value);
  testCase.AddInput("Y", shape, value);
  testCase.AddOutput("dX");
}

TEST(SoftmaxGradExpansionTest, DefaultAxis) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, NegativeAxis) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.AddAttribute("axis", -1);
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, PositiveAxis) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.AddAttribute("axis", 1);
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, 3D) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2, 2});
  testCase.RunTest();
}

}  // namespace test
}  // namespace onnxruntime