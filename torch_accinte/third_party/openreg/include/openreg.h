#pragma once

#include <cstddef>

#ifdef _WIN32
#define ACCINTE_EXPORT __declspec(dllexport)
#else
#define ACCINTE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum orError_t {
  orSuccess = 0,
  orErrorUnknown = 1,
  orErrorNotReady = 2
} orError_t;

typedef enum orMemcpyKind {
  orMemcpyHostToHost = 0,
  orMemcpyHostToDevice = 1,
  orMemcpyDeviceToHost = 2,
  orMemcpyDeviceToDevice = 3
} orMemcpyKind;

typedef enum orMemoryType {
  orMemoryTypeUnmanaged = 0,
  orMemoryTypeHost = 1,
  orMemoryTypeDevice = 2
} orMemoryType;

struct orPointerAttributes {
  orMemoryType type = orMemoryType::orMemoryTypeUnmanaged;
  int device;
  void* pointer;
};

typedef enum orEventFlags {
  orEventDisableTiming = 0x0,
  orEventEnableTiming = 0x1,
} orEventFlags;

struct orStream;
struct orEvent;
typedef struct orStream* orStream_t;
typedef struct orEvent* orEvent_t;

// Memory
ACCINTE_EXPORT orError_t orMalloc(void** devPtr, size_t size);
ACCINTE_EXPORT orError_t orFree(void* devPtr);
ACCINTE_EXPORT orError_t orMallocHost(void** hostPtr, size_t size);
ACCINTE_EXPORT orError_t orFreeHost(void* hostPtr);
ACCINTE_EXPORT orError_t
orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind);
ACCINTE_EXPORT orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream);
ACCINTE_EXPORT orError_t
orPointerGetAttributes(orPointerAttributes* attributes, const void* ptr);
ACCINTE_EXPORT orError_t orMemoryUnprotect(void* devPtr);
ACCINTE_EXPORT orError_t orMemoryProtect(void* devPtr);

// Device
ACCINTE_EXPORT orError_t orGetDeviceCount(int* count);
ACCINTE_EXPORT orError_t orSetDevice(int device);
ACCINTE_EXPORT orError_t orGetDevice(int* device);
ACCINTE_EXPORT orError_t
orDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
ACCINTE_EXPORT orError_t orDeviceSynchronize(void);

// Stream
ACCINTE_EXPORT orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    unsigned int flags,
    int priority);
ACCINTE_EXPORT orError_t orStreamCreate(orStream_t* stream);
ACCINTE_EXPORT orError_t orStreamGetPriority(orStream_t stream, int* priority);
ACCINTE_EXPORT orError_t orStreamDestroy(orStream_t stream);
ACCINTE_EXPORT orError_t orStreamQuery(orStream_t stream);
ACCINTE_EXPORT orError_t orStreamSynchronize(orStream_t stream);
ACCINTE_EXPORT orError_t
orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int flags);

// Event
ACCINTE_EXPORT orError_t
orEventCreateWithFlags(orEvent_t* event, unsigned int flags);
ACCINTE_EXPORT orError_t orEventCreate(orEvent_t* event);
ACCINTE_EXPORT orError_t orEventDestroy(orEvent_t event);
ACCINTE_EXPORT orError_t orEventRecord(orEvent_t event, orStream_t stream);
ACCINTE_EXPORT orError_t orEventSynchronize(orEvent_t event);
ACCINTE_EXPORT orError_t orEventQuery(orEvent_t event);
ACCINTE_EXPORT orError_t
orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

#define ACCINTE_H
#include "accinte.inl"

#endif
