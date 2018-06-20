// Author: Patrick Wieschollek <mail@patwie.com>
// apply PSF kernel to an image on the GPU
// TODO: create batch-version of run_blur_image

#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

#define checkCUDNN(expression)                               \
{                                                            \
  cudnnStatus_t status = (expression);                       \
  if (status != CUDNN_STATUS_SUCCESS) {                      \
    std::stringstream strstr;                                \
    strstr << "Error on line " << __LINE__ << ": "           \
    << cudnnGetErrorString(status) << std::endl;             \
    throw strstr.str();                                      \
  }                                                          \
}

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t error = (expression);                        \
  if (error != cudaSuccess) {                              \
    throw std::runtime_error(cudaGetErrorString(error));   \
  }                                                        \
}

void run_blur_image
(
  int gpu_id,
  const float *img_h, unsigned int iH, unsigned int iW, unsigned int iC,
  const float *psf_h, unsigned int pH, unsigned int pW,
  float **output_d) {

  cudaSetDevice(gpu_id);

  const int image_bytes = iC * iH * iW * sizeof(float);
  const int psf_bytes = pH * pW * sizeof(float);

  // copy to device memory
  float *img_d;
  checkCUDA(cudaMalloc(&img_d, image_bytes));
  checkCUDA(cudaMemcpy(img_d, img_h, image_bytes, cudaMemcpyHostToDevice));

  float *psf_d;
  checkCUDA(cudaMalloc(&psf_d, psf_bytes));
  checkCUDA(cudaMemcpy(psf_d, psf_h, psf_bytes, cudaMemcpyHostToDevice));

  checkCUDA(cudaMalloc(&*output_d, image_bytes));
  checkCUDA(cudaMemset(*output_d, 0, image_bytes));


  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
               input_descriptor,
               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, iH, iW));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(
               kernel_descriptor,
               CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, pH, pW));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(
               convolution_descriptor,
               (pH - 1) / 2, (pW - 1) / 2, 1, 1, 1, 1,
               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
               output_descriptor,
               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, iH, iW));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
               cudnn,
               input_descriptor,
               kernel_descriptor,
               convolution_descriptor,
               output_descriptor,
               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
               0, &convolution_algorithm));

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
               cudnn,
               input_descriptor,
               kernel_descriptor,
               convolution_descriptor,
               output_descriptor,
               convolution_algorithm,
               &workspace_bytes));

  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  const float alpha = 1.0f, beta = 0.0f;
  // no NVIDIA-cudnn depthwise primitive?
  for (int i = 0; i < iC; ++i) {
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       img_d + i * iH * iW,
                                       kernel_descriptor,
                                       psf_d,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       *output_d + i * iH * iW));
  }

  checkCUDA(cudaFree(psf_d));
  checkCUDA(cudaFree(img_d));
  checkCUDA(cudaFree(d_workspace));
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}


