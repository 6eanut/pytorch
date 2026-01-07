#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <include/Macros.h>

#include <limits>

namespace c10::accinte {

ACCINTE_EXPORT void set_device(DeviceIndex device);
ACCINTE_EXPORT DeviceIndex device_count() noexcept;
ACCINTE_EXPORT DeviceIndex current_device();
ACCINTE_EXPORT DeviceIndex maybe_exchange_device(DeviceIndex to_device);

ACCINTE_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device);

static inline void check_device_index(int64_t device) {
  TORCH_CHECK(
      device >= 0 && device < c10::accinte::device_count(),
      "The device index is out of range. It must be in [0, ",
      static_cast<int>(c10::accinte::device_count()),
      "), but got ",
      static_cast<int>(device),
      ".");
}

} // namespace c10::accinte