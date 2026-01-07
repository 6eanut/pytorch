#include <c10/util/Exception.h>
#include <include/openreg.h>

#include "AccInteException.h"
#include "AccInteFunctions.h"

namespace c10::accinte {

// orGetDevice and orSetDevice implementation is in
// acc_inte/third_party/accinte/csrc/device.cpp
orError_t SetDevice(DeviceIndex device) {
  int cur_device = -1;
  ACCINTE_CHECK(orGetDevice(&cur_device));
  if (device == cur_device) {
    return orSuccess;
  }
  return orSetDevice(device);
}

ACCINTE_EXPORT void set_device(DeviceIndex device) {
  check_device_index(device);
  ACCINTE_CHECK(SetDevice(device));
}

orError_t GetDeviceCount(int* dev_count) {
  return orGetDeviceCount(dev_count);
}

orError_t GetDeviceCount(int* dev_count) {
  return orGetDeviceCount(dev_count);
}

int device_count_impl() {
  int count = 0;
  GetDeviceCount(&count);
  return count;
}

ACCINTE_EXPORT DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      int result = device_count_impl();
      ACCINTE_CHECK(orGetDeviceCount(&result));
      TORCH_CHECK(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many devices, DeviceIndex overflowed");
      return result;
    } catch (const Error& ex) {
      TORCH_WARN("Device initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

orError_t GetDevice(DeviceIndex* device) {
  int tmp_device = -1;
  auto err = orGetDevice(&tmp_device);
  *device = static_cast<DeviceIndex>(tmp_device);
  return err;
}

ACCINTE_EXPORT DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  ACCINTE_CHECK(GetDevice(&cur_device));
  return cur_device;
}

ACCINTE_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device) {
  int current_device = -1;
  orGetDevice(&current_device);
  if (device != current_device) {
    orSetDevice(device);
  }
  return current_device;
}

ACCINTE_EXPORT DeviceIndex maybe_exchange_device(DeviceIndex to_device) {
  check_device_index(to_device);
  return ExchangeDevice(to_device);

} // namespace c10::accinte