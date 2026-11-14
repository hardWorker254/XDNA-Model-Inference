/*===- xrt_shim.cpp -------------------------------------------*- C++ -*-===//
 *
 * NpuEmbeddings -- the XRT shim: Rust's calls, turned into XRT's calls.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * WHAT THIS FILE IS, AND WHAT IT IS NOT
 * -------------------------------------
 * It is a TRANSPORT. Every function here turns one call into one XRT call
 * and reports whether that call worked. There is no policy in this file: no
 * buffer is sized here, no width is rounded here, no decision is made here.
 *
 * The reason is that a shim which decides anything is a second implementation
 * of the runtime's rules, and the two would drift. The rules -- the widths a
 * design accepts, the packing of an operand, the order of the buffers -- live
 * in Rust, where they can be read, tested and changed without a C++ compiler.
 *
 * WHY IT IS C++ AND NOT C
 * -----------------------
 * XRT's API is C++ and there is no C API for it. The ABI exposed here is C,
 * so the Rust side needs no C++ headers, no C++ standard library and no
 * knowledge of XRT's types.
 *
 * THE SHIM'S SHAPE IS THE C++ RUNTIME'S
 * -------------------------------------
 * The C++ runtime did exactly this -- `xrt::bo`, `xrt::hw_context`,
 * `xrt::kernel`, `xrt::run` -- so the semantics of these calls are known:
 * the same buffer flags, the same group ids, the same argument order. A port
 * whose transport differs from the original's is a port with a second thing
 * to be wrong about.
 *===----------------------------------------------------------------------===*/

#include "xrt_shim.h"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "xrt/experimental/xrt_ext.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace {

// The error text of the last failed call ON THIS THREAD.
//
// A thread-local, because the runtime dispatches one queue per thread and a
// shared error string would let one thread's failure be reported against
// another's call.
thread_local std::string g_error;

void set_error(const std::string &what) { g_error = what; }

// EVERY ENTRY POINT GOES THROUGH HERE.
//
// An exception that escapes into Rust is a process that dies in a place with
// no message, and a C ABI has no way to carry an exception. So every call
// catches, records the text and reports a failure -- and the Rust side turns
// that into a refusal with the driver's own words.
template <typename F> int guard(F &&body) {
  try {
    body();
    return 0;
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  } catch (...) {
    set_error("unknown error");
    return -1;
  }
}

xrt::bo *as_bo(void *p) { return static_cast<xrt::bo *>(p); }
xrt::kernel *as_kernel(void *p) { return static_cast<xrt::kernel *>(p); }
xrt::device *as_device(void *p) { return static_cast<xrt::device *>(p); }
xrt::hw_context *as_context(void *p) { return static_cast<xrt::hw_context *>(p); }

} // namespace

extern "C" {

const char *shim_last_error(void) { return g_error.c_str(); }

int shim_device_open(uint32_t index, void **out_device) {
  return guard([&]() {
    auto *dev = new xrt::device(index);
    *out_device = dev;
  });
}

void shim_device_close(void *device) { delete as_device(device); }

int shim_context_open(void *device, const uint8_t *xclbin, size_t bytes,
                      void **out_context) {
  return guard([&]() {
    // FROM MEMORY, not from a path. A design set that lives inside a
    // container has no file to point at, and the C++ runtime solved the same
    // problem the same way (`xrt::xclbin` takes a vector of chars).
    std::vector<char> blob(reinterpret_cast<const char *>(xclbin),
                           reinterpret_cast<const char *>(xclbin) + bytes);
    xrt::xclbin xcl(blob);
    auto uuid = as_device(device)->register_xclbin(xcl);
    auto *ctx = new xrt::hw_context(*as_device(device), uuid);
    *out_context = ctx;
  });
}

void shim_context_close(void *context) { delete as_context(context); }

int shim_kernel_open(void *context, const char *name, void **out_kernel) {
  return guard([&]() {
    auto *k = new xrt::kernel(*as_context(context), std::string(name));
    *out_kernel = k;
  });
}

void shim_kernel_close(void *kernel) { delete as_kernel(kernel); }

int shim_kernel_group_id(void *kernel, int argno) {
  int group = -1;
  const int rc = guard([&]() { group = as_kernel(kernel)->group_id(argno); });
  return rc == 0 ? group : -1;
}

int shim_bo_alloc(void *device, uint64_t bytes, int flags, int group,
                  void **out_bo) {
  return guard([&]() {
    auto *b = new xrt::bo(
        *as_device(device), static_cast<size_t>(bytes),
        (flags & SHIM_BO_CACHEABLE) ? XCL_BO_FLAGS_CACHEABLE
                                    : XRT_BO_FLAGS_HOST_ONLY, group);
    *out_bo = b;
  });
}

int shim_bo_write(void *bo, const void *data, uint64_t bytes, uint64_t offset) {
  return guard([&]() {
    void *host = as_bo(bo)->map<void *>();
    std::memcpy(static_cast<char *>(host) + offset, data,
                static_cast<size_t>(bytes));
  });
}

int shim_bo_read(void *bo, void *data, uint64_t bytes, uint64_t offset) {
  return guard([&]() {
    const void *host = as_bo(bo)->map<const void *>();
    std::memcpy(data, static_cast<const char *>(host) + offset,
                static_cast<size_t>(bytes));
  });
}

int shim_bo_sync(void *bo, int to_device, uint64_t bytes, uint64_t offset) {
  return guard([&]() {
    as_bo(bo)->sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE
                              : XCL_BO_SYNC_BO_FROM_DEVICE,
                    static_cast<size_t>(bytes),
                    static_cast<size_t>(offset));
  });
}

uint64_t shim_bo_size(void *bo) {
  uint64_t size = 0;
  guard([&]() { size = as_bo(bo)->size(); });
  return size;
}

void shim_bo_free(void *bo) { delete as_bo(bo); }

int shim_run_start(void *kernel, void *instr_bo, uint32_t words, void **args,
                   int n_args, void **out_run) {
  return guard([&]() {
    auto *k = as_kernel(kernel);
    auto *instr = as_bo(instr_bo);
    // THE KERNEL'S ARGUMENT LIST IS ITS SIGNATURE, AND THE SIGNATURE IS THE
    // DESIGN'S.
    //
    // The leading integer is the design's own first argument -- the AIE
    // kernel takes a scalar before the instruction stream -- and it is NOT an
    // iteration count: `xrt::autostart` is the type that would say that, and
    // a plain integer is an argument like any other. The C++ runtime and the
    // Python design harness both pass 3 here, and this is the same call.
    xrt::run *r = nullptr;
    switch (n_args) {
    case 2:
      r = new xrt::run((*k)(3, *instr, words, *as_bo(args[0]),
                            *as_bo(args[1])));
      break;
    case 3:
      r = new xrt::run((*k)(3, *instr, words, *as_bo(args[0]),
                            *as_bo(args[1]), *as_bo(args[2])));
      break;
    case 4:
      r = new xrt::run((*k)(3, *instr, words, *as_bo(args[0]),
                            *as_bo(args[1]), *as_bo(args[2]),
                            *as_bo(args[3])));
      break;
    default:
      set_error("the kernel takes two, three or four buffers, not " +
                std::to_string(n_args));
      return;
    }
    *out_run = r;
  });
}

int shim_run_wait(void *run) {
  int state = -1;
  const int rc = guard([&]() {
    state = static_cast<int>(static_cast<xrt::run *>(run)->wait());
  });
  return rc == 0 ? state : -1;
}

void shim_run_free(void *run) { delete static_cast<xrt::run *>(run); }

} // extern "C"
