#pragma once

#include <include/openreg.h>

#include "AccInteException.h"
#include "AccInteFunctions.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

namespace c10::accinte {

// Derive compile-time priority count from shared accinte backend constant.
static constexpr int max_compile_time_stream_priorities = 2;

class AccInteStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit AccInteStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::PrivateUse1);
  }

  explicit AccInteStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const AccInteStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const AccInteStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  operator orStream_t() const {
    return stream();
  }

  operator Stream() const {
    return unwrap();
  }

  DeviceType device_type() const {
    return DeviceType::PrivateUse1;
  }

  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  Device device() const {
    return Device(DeviceType::PrivateUse1, device_index());
  }

  StreamId id() const {
    return stream_.id();
  }

  bool query() const {
    DeviceGuard guard{stream_.device()};

    if (orStreamQuery(stream()) == orSuccess) {
      return true;
    }

    return false;
  }

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    ACCINTE_CHECK(orStreamSynchronize(stream()));
  }

  int priority() const {
    DeviceGuard guard{stream_.device()};
    int priority = 0;
    ACCINTE_CHECK(orStreamGetPriority(stream(), &priority));
    return priority;
  }

  orStream_t stream() const;

  Stream unwrap() const {
    return stream_;
  }

  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  static AccInteStream unpack3(
      StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    return AccInteStream(Stream::unpack3(stream_id, device_index, device_type));
  }

 private:
  Stream stream_;
};

/*
 * Get a stream from the pool in a round-robin fashion.
 *
 * You can request a stream from the highest priority pool by setting
 * isHighPriority to true for a specific device.
 */
ACCINTE_EXPORT AccInteStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

/*
 * Get a stream from the pool in a round-robin fashion.
 *
 * You can request a stream by setting a priority value for a specific device.
 * The priority number lower, the priority higher.
 */
ACCINTE_EXPORT AccInteStream
getStreamFromPool(const int priority, DeviceIndex device = -1);

/*
 * Get a AccInteStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
ACCINTE_EXPORT AccInteStream
getStreamFromExternal(orStream_t ext_stream, DeviceIndex device_index);

/*
 * Get the default AccInte stream, for the passed AccInte device, or for the
 * current device if no device index is passed.
 */
ACCINTE_EXPORT AccInteStream
getDefaultAccInteStream(DeviceIndex device_index = -1);

/*
 * Get the current AccInte stream, for the passed AccInte device, or for the
 * current device if no device index is passed.
 */
ACCINTE_EXPORT AccInteStream
getCurrentAccInteStream(DeviceIndex device_index = -1);

/*
 * Set the current stream on the device of the passed in stream to be the passed
 * in stream.
 */
ACCINTE_EXPORT void setCurrentAccInteStream(AccInteStream stream);

ACCINTE_EXPORT std::ostream& operator<<(
    std::ostream& stream,
    const AccInteStream& s);

} // namespace c10::accinte

namespace std {
template <>
struct hash<c10::accinte::AccInteStream> {
  size_t operator()(c10::accinte::AccInteStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
