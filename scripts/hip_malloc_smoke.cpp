#include <cstdio>
#include <hip/hip_runtime.h>

int main() {
  std::fprintf(stderr, "before hipGetDeviceCount\n");
  std::fflush(stderr);

  int count = 0;
  hipError_t error = hipGetDeviceCount(&count);
  std::fprintf(stderr, "after hipGetDeviceCount error=%d count=%d %s\n",
               static_cast<int>(error), count, hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess || count < 1) {
    return 1;
  }

  std::fprintf(stderr, "before hipSetDevice\n");
  std::fflush(stderr);
  error = hipSetDevice(0);
  std::fprintf(stderr, "after hipSetDevice error=%d %s\n",
               static_cast<int>(error), hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess) {
    return 2;
  }

  void* ptr = nullptr;
  std::fprintf(stderr, "before hipMalloc\n");
  std::fflush(stderr);
  error = hipMalloc(&ptr, 4);
  std::fprintf(stderr, "after hipMalloc error=%d ptr=%p %s\n",
               static_cast<int>(error), ptr, hipGetErrorString(error));
  std::fflush(stderr);
  if (ptr) {
    hipFree(ptr);
  }
  return error == hipSuccess ? 0 : 3;
}
