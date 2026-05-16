#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void add_one(float* value) {
  value[0] += 1.0f;
}

int main() {
  int count = 0;
  hipError_t error = hipGetDeviceCount(&count);
  std::fprintf(stderr, "hipGetDeviceCount error=%d count=%d %s\n",
               static_cast<int>(error), count, hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess || count < 1) {
    return 1;
  }

  error = hipSetDevice(0);
  std::fprintf(stderr, "hipSetDevice error=%d %s\n", static_cast<int>(error), hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess) {
    return 2;
  }

  float host = 41.0f;
  float* device = nullptr;
  error = hipMalloc(&device, sizeof(float));
  std::fprintf(stderr, "hipMalloc error=%d ptr=%p %s\n", static_cast<int>(error), device, hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess) {
    return 3;
  }

  error = hipMemcpy(device, &host, sizeof(float), hipMemcpyHostToDevice);
  std::fprintf(stderr, "hipMemcpy H2D error=%d %s\n", static_cast<int>(error), hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess) {
    return 4;
  }

  std::fprintf(stderr, "before kernel\n");
  std::fflush(stderr);
  hipLaunchKernelGGL(add_one, dim3(1), dim3(1), 0, 0, device);
  error = hipGetLastError();
  std::fprintf(stderr, "after launch error=%d %s\n", static_cast<int>(error), hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess) {
    return 5;
  }

  std::fprintf(stderr, "before sync\n");
  std::fflush(stderr);
  error = hipDeviceSynchronize();
  std::fprintf(stderr, "after sync error=%d %s\n", static_cast<int>(error), hipGetErrorString(error));
  std::fflush(stderr);
  if (error != hipSuccess) {
    return 6;
  }

  host = 0.0f;
  error = hipMemcpy(&host, device, sizeof(float), hipMemcpyDeviceToHost);
  std::fprintf(stderr, "hipMemcpy D2H error=%d value=%.1f %s\n", static_cast<int>(error), host, hipGetErrorString(error));
  std::fflush(stderr);

  hipFree(device);
  return host == 42.0f ? 0 : 7;
}
