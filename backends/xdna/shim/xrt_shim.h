/*===- xrt_shim.h ---------------------------------------------*- C -*-===//
 *
 * NpuEmbeddings -- the C ABI of the XRT shim.
 *
 * SPDX-License-Identifier: Apache-2.0
 *===----------------------------------------------------------------------===*/
#ifndef NPUE_XRT_SHIM_H
#define NPUE_XRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* THE BO FLAGS, SPELLED HERE AND NOT IN XRT's HEADERS.
 *
 * The Rust side must not include XRT's headers -- it must not even know that
 * XRT exists -- so the two flags the runtime uses are named here and mapped
 * to XRT's constants inside the shim. A flag whose value is XRT's number is
 * a flag that changes meaning when XRT's header changes.
 */
#define SHIM_BO_CACHEABLE 1 /* the instruction stream: device-side, cached */
#define SHIM_BO_HOST_ONLY 2 /* the data: host-visible with a device address */

/* THE ERROR TEXT OF THE LAST FAILED CALL ON THIS THREAD.
 *
 * A shim call that fails says what failed. A shim call that fails and says
 * nothing leaves the caller with an errno and a guess, and the guess is
 * usually "the board is busy" when the truth is "the xclbin has no kernel
 * with that name".
 */
const char *shim_last_error(void);

/* THE DEVICE. */
int shim_device_open(uint32_t index, void **out_device);
void shim_device_close(void *device);

/* THE HARDWARE CONTEXT AND THE KERNEL IN IT.
 *
 * The xclbin is handed over as BYTES, not as a path: a design set that lives
 * inside a container has no file to point at, and the C++ runtime solved the
 * same problem by handing the blob to XRT (`xrt::xclbin` takes a vector).
 */
int shim_context_open(void *device, const uint8_t *xclbin, size_t bytes,
                      void **out_context);
void shim_context_close(void *context);

int shim_kernel_open(void *context, const char *name, void **out_kernel);
void shim_kernel_close(void *kernel);

/* The memory-bank group id of one kernel argument.
 *
 * The data buffers are not allocated "for the kernel": they are allocated for
 * a GROUP, and the group decides which memory bank they land in. The
 * instruction stream is group 1 and the data arguments are 3, 4, 5 -- the
 * numbers the design's own kernel signature uses.
 */
int shim_kernel_group_id(void *kernel, int argno);

/* THE BUFFERS. */
int shim_bo_alloc(void *device, uint64_t bytes, int flags, int group,
                  void **out_bo);
int shim_bo_write(void *bo, const void *data, uint64_t bytes, uint64_t offset);
int shim_bo_read(void *bo, void *data, uint64_t bytes, uint64_t offset);
int shim_bo_sync(void *bo, int to_device, uint64_t bytes, uint64_t offset);
uint64_t shim_bo_size(void *bo);
void shim_bo_free(void *bo);

/* THE RUN.
 *
 * `words` is the instruction stream's length in 32-bit words, not in bytes:
 * XRT counts instructions, and a byte count is a number the caller has to
 * remember to divide.
 */
int shim_run_start(void *kernel, void *instr_bo, uint32_t words, void **args,
                   int n_args, void **out_run);
/* 0 when the kernel completed; the ERT state otherwise. */
int shim_run_wait(void *run);
void shim_run_free(void *run);

#ifdef __cplusplus
}
#endif

#endif /* NPUE_XRT_SHIM_H */
