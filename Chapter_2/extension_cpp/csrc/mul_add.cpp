#include <ATen/Operators.h>
#include <Python.h>
#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}
}

namespace extension_cpp {
at::Tensor mymul_cpu(const at::Tensor &a, const at::Tensor &b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float *a_ptr = a_contig.data_ptr<float>();
  const float *b_ptr = b_contig.data_ptr<float>();

  float *result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}

at::Tensor mymuladd_cpu(const at::Tensor &a, const at::Tensor &b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float *a_ptr = a_contig.data_ptr<float>();
  const float *b_ptr = b_contig.data_ptr<float>();

  float *result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

void myadd_out_cpu(const at::Tensor &a, const at::Tensor &b, at::Tensor &out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float *a_ptr = a_contig.data_ptr<float>();
  const float *b_ptr = b_contig.data_ptr<float>();

  float *result_ptr = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
}

TORCH_LIBRARY(extension_cpp, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("mymul_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out_cpu", &myadd_out_cpu);
}

} // namespace extension_cpp