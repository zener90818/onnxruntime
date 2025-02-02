// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_def_builder.h"

#include "gtest/gtest.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace test {

TEST(KernelDefTest, HashIgnoresTypeConstraintTypeOrdering) {
  auto build_kernel_def = [](std::vector<MLDataType> type_constraint_types) {
    return KernelDefBuilder{}
        .SetName("MyOp")
        .SetDomain("MyDomain")
        .Provider("MyProvider")
        .TypeConstraint("T", type_constraint_types)
        .Build();
  };

  const auto a = build_kernel_def(BuildKernelDefConstraints<int, float>());
  const auto b = build_kernel_def(BuildKernelDefConstraints<float, int>());

  ASSERT_EQ(a->GetHash(), b->GetHash());
}

}  // namespace test
}  // namespace onnxruntime
