// Author: Patrick Wieschollek <mail@patwie.com>
// apply PSF kernel to an image on the GPU
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <cuda_runtime.h>

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t error = (expression);                        \
  if (error != cudaSuccess) {                              \
    throw std::runtime_error(cudaGetErrorString(error));   \
  }                                                        \
}

void run_blur_image(
  int gpu_id,
  const float *img, unsigned int iH, unsigned int iW, unsigned int iC,
  const float *psf, unsigned int pH, unsigned int pW,
  float **output);


void blur(pybind11::array_t<float> image,
          pybind11::array_t<float> psf,
          int gpu_id) {
  auto img_hnd = image.request();
  if (img_hnd.ndim != 3) {
    std::stringstream strstr;
    strstr << "[blur] expect img.ndim == 3 but is " << img_hnd.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }
  const unsigned int iH = img_hnd.shape[1];
  const unsigned int iW = img_hnd.shape[2];
  const unsigned int iC = img_hnd.shape[0];

  if ((iC != 1) && (iC != 3)) {
    std::stringstream strstr;
    strstr << "[blur] expect img.shape[0] == 1,3 in CHW-format but is " << img_hnd.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  const auto psf_hnd = psf.request();
  if (psf_hnd.ndim != 2) {
    std::stringstream strstr;
    strstr << "[blur] expect psf.ndim == 2 but is " << psf_hnd.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  const unsigned int pH = psf_hnd.shape[0];
  const unsigned int pW = psf_hnd.shape[1];
  const int image_bytes = iC * iH * iW * sizeof(float);


  float **flat_output_d;
  run_blur_image(gpu_id,
                 reinterpret_cast<float*>(img_hnd.ptr), iH, iW, iC,
                 reinterpret_cast<float*>(psf_hnd.ptr), pH, pW,
                 flat_output_d);

  checkCUDA(cudaMemcpy(img_hnd.ptr, *flat_output_d, image_bytes, cudaMemcpyDeviceToHost));
  checkCUDA(cudaFree(*flat_output_d));
}

void blur_batch(std::vector<pybind11::array_t<float>> image,
                std::vector<pybind11::array_t<float>> psf,
                int gpu_id) {
  const int iB = image.size();
  const int pB = psf.size();

  // init_gpu

  if (iB != pB) {
    std::stringstream strstr;
    strstr << "number of psf kernels and images should match but are " << iB << " and " << pB << std::endl;
    throw std::runtime_error(strstr.str());
  }

  for (int i = 0; i < iB; ++i) {
    blur(image[i], psf[i], 0);
  }
}

PYBIND11_MODULE(blur_library, m) {
  m.doc() = "GPU accelerated blur method";
  m.def("blur", blur);
  m.def("blur_batch", blur_batch);
}
