// Minimal stand-in for Vitis HLS <hls_stream.h>, used ONLY by Makefile.gcc
// (C simulation with plain g++). Vitis HLS uses its own header.
#ifndef HLS_STREAM_STUB_H
#define HLS_STREAM_STUB_H
#include <deque>
#include <stdexcept>
namespace hls {
template <typename T> class stream {
    std::deque<T> q;
  public:
    stream() {}
    explicit stream(const char *) {}
    void write(const T &v) { q.push_back(v); }
    T read() {
        if (q.empty()) throw std::runtime_error("hls::stream read from empty stream");
        T v = q.front();
        q.pop_front();
        return v;
    }
    bool empty() const { return q.empty(); }
    size_t size() const { return q.size(); }
};
}  // namespace hls
#endif
