// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <condition_variable>

namespace onnxruntime {
namespace contrib {

class OrtEventPool final {
 public:
  // TOken IDs are added to events to keep trace of the different
  // reasons for passing control between ORT and Python.  They should
  // probably just be a debugging aid, with shared knowledge of the
  // model being used to keep the pieces in sync.  Currently the
  // proof-of-concept uses the values to dispatch to different code
  // paths.
  //
  // These numbers must match ortmodule.py.  In addition,
  // ORTModule.max_id must be <= the difference between successive
  // values here.  For instance, if we assign IDs [0,100) to custom
  // autograd functions then we need a range [100,200) for the tokens
  // returned between TOKEN_HOLE_FORWARD and TOKEN_YIELD_END_FORWARD.
  static constexpr int TOKEN_HOLE_FORWARD = 100;
  static constexpr int TOKEN_YIELD_END_FORWARD = 200;
  static constexpr int TOKEN_HOLE_BACKWARD = 300;
  static constexpr int TOKEN_END_BACKWARD = 400;
  
  static OrtEventPool& GetInstance() {
    static OrtEventPool instance_;
    return instance_;
  }
  void SignalEvent(int64_t id, int64_t token = 0);
  bool QueryEvent(int64_t id) const;
  int64_t ResetAndWaitEvent(int64_t id);
  int64_t WaitEvent(int64_t id) const;
  void ResetAllEvents();

  static size_t GetPoolSize() {
    return MaxNumItems;
  }

 private:
  OrtEventPool() = default;
  ~OrtEventPool() = default;
  OrtEventPool(const OrtEventPool&) = delete;
  OrtEventPool& operator=(const OrtEventPool&) = delete;

  void CheckRange(const int64_t event_id) const;

  struct Item {
    std::atomic<bool> signaled;
    int64_t token;
    mutable std::mutex mutex;
    mutable std::condition_variable cv;

    Item() {
      signaled.store(false);
    }
  };

  enum {
    MaxNumItems = 4096
  };

  Item pool_[MaxNumItems];
};

}  // namespace contrib
}  // namespace onnxruntime
