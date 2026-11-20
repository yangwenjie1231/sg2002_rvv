#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include "rvv.hpp"

namespace py = pybind11;

// ---------- 通用错误提示宏 ----------
#define ERR_SHAPE(msg) \
    throw std::invalid_argument(std::string("Shape/Type error: ") + (msg))

#define ERR_TYPE(expected, actual) \
    throw std::invalid_argument( \
        "dtype mismatch: expected " + std::string(expected) \
        + ", got " + std::string(actual))

// ---------- float32 工具 ----------
std::vector<float> np_to_vec_f32(py::array_t<float> a) {
    py::buffer_info info = a.request();
    if (info.ndim != 1)
        ERR_SHAPE("Only 1-D array accepted");
    // PEP 3118 format code for IEEE-754 binary32 ("float32" is not a valid
    // struct format and never matches NumPy ndarray buffers).
    if (info.format != "f")
        ERR_TYPE("float32", info.format);
    if (info.strides[0] != static_cast<py::ssize_t>(sizeof(float))) {
        throw std::invalid_argument(
            "Shape/Type error: input must be a C-contiguous float32 array");
    }
    float* ptr = static_cast<float*>(info.ptr);
    return std::vector<float>(ptr, ptr + info.size);
}

py::array_t<float> vec_to_np_f32(const std::vector<float>& v) {
    py::array_t<float> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// ---------- int8 工具 ----------
std::vector<int8_t> np_to_vec_i8(py::array_t<int8_t> a) {
    py::buffer_info info = a.request();
    if (info.ndim != 1)
        ERR_SHAPE("Only 1-D array accepted");
    if (info.format != "b")   // PEP 3118 signed char / numpy int8
        ERR_TYPE("int8", info.format);
    if (info.strides[0] != 1) {
        throw std::invalid_argument(
            "Shape/Type error: input must be a C-contiguous int8 array");
    }
    int8_t* ptr = static_cast<int8_t*>(info.ptr);
    return std::vector<int8_t>(ptr, ptr + info.size);
}

py::array_t<int8_t> vec_to_np_i8(const std::vector<int8_t>& v) {
    py::array_t<int8_t> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}


//--------------------------------------
// 向量运算封装
//--------------------------------------
//--------------------------------------
// float32 向量运算（带详细错误提示）
//--------------------------------------
py::array_t<float> py_add(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec_f32(a);
    auto vb = np_to_vec_f32(b);
    if (va.size() != vb.size()) {
        throw std::invalid_argument(
            "[add] shape mismatch: a.size=" + std::to_string(va.size()) +
            " vs b.size=" + std::to_string(vb.size()));
    }
    std::vector<float> vc(va.size());
    rvv::core::add(va.data(), vb.data(), vc.data(), va.size());
    return vec_to_np_f32(vc);
}

py::array_t<float> py_sub(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec_f32(a);
    auto vb = np_to_vec_f32(b);
    if (va.size() != vb.size()) {
        throw std::invalid_argument(
            "[sub] shape mismatch: a.size=" + std::to_string(va.size()) +
            " vs b.size=" + std::to_string(vb.size()));
    }
    std::vector<float> vc(va.size());
    rvv::core::sub(va.data(), vb.data(), vc.data(), va.size());
    return vec_to_np_f32(vc);
}

py::array_t<float> py_scale(py::array_t<float> a, float k) {
    auto va = np_to_vec_f32(a);
    std::vector<float> vb(va.size());
    rvv::core::scale(va.data(), k, vb.data(), va.size());
    return vec_to_np_f32(vb);
}

float py_dot(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec_f32(a);
    auto vb = np_to_vec_f32(b);
    if (va.size() != vb.size()) {
        throw std::invalid_argument(
            "[dot] shape mismatch: a.size=" + std::to_string(va.size()) +
            " vs b.size=" + std::to_string(vb.size()));
    }
    return rvv::core::dot(va.data(), vb.data(), va.size());
}

float py_norm_l2(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    return rvv::core::norm_l2(va.data(), va.size());
}

py::array_t<float> py_normalize(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    std::vector<float> vb(va.size());
    rvv::core::normalize(va.data(), vb.data(), va.size());
    return vec_to_np_f32(vb);
}

py::array_t<float> py_abs_f32(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    std::vector<float> vb(va.size());
    rvv::core::abs_f32(va.data(), vb.data(), va.size());
    return vec_to_np_f32(vb);
}

double py_sum_f32(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    return rvv::core::sum_f32(va.data(), va.size());
}

py::tuple py_mean_std_f32(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    const auto result = rvv::core::mean_std_f32(va.data(), va.size());
    return py::make_tuple(result.first, result.second);
}

py::array_t<float> py_mul_f32(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec_f32(a);
    auto vb = np_to_vec_f32(b);
    if (va.size() != vb.size()) {
        throw std::invalid_argument(
            "[mul_f32] shape mismatch: a.size=" + std::to_string(va.size()) +
            " vs b.size=" + std::to_string(vb.size()));
    }
    std::vector<float> vc(va.size());
    rvv::core::mul_f32(va.data(), vb.data(), vc.data(), va.size());
    return vec_to_np_f32(vc);
}

py::array_t<float> py_add_scalar_f32(py::array_t<float> a, float k) {
    auto va = np_to_vec_f32(a);
    std::vector<float> vb(va.size());
    rvv::core::add_scalar_f32(va.data(), k, vb.data(), va.size());
    return vec_to_np_f32(vb);
}

py::array_t<float> py_clip_f32(py::array_t<float> a, float lo, float hi) {
    if (std::isnan(lo) || std::isnan(hi) || lo > hi) {
        throw std::invalid_argument(
            "[clip_f32] require non-NaN bounds with lo <= hi");
    }
    auto va = np_to_vec_f32(a);
    std::vector<float> vb(va.size());
    rvv::core::clip_f32(va.data(), lo, hi, vb.data(), va.size());
    return vec_to_np_f32(vb);
}

float py_max_f32(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    return rvv::core::max_f32(va.data(), va.size());
}

float py_min_f32(py::array_t<float> a) {
    auto va = np_to_vec_f32(a);
    return rvv::core::min_f32(va.data(), va.size());
}

//--------------------------------------
// 矩阵运算封装（2-D array）
//--------------------------------------
using MatF = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::vector<float> mat_to_vec(const MatF& m) {
    py::buffer_info info = m.request();
    if (info.ndim != 2)
        throw std::runtime_error("Need 2-D array");
    return std::vector<float>(static_cast<float*>(info.ptr),
                              static_cast<float*>(info.ptr) + info.size);
}

MatF vec_to_mat(const std::vector<float>& v, std::size_t rows, std::size_t cols) {
    MatF out({rows, cols});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

MatF py_add2d(MatF A, MatF B) {
    auto vA = mat_to_vec(A);
    auto vB = mat_to_vec(B);
    if (vA.size() != vB.size())
        throw std::runtime_error("Shape mismatch");
    std::vector<float> vC(vA.size());
    py::buffer_info info = A.request();
    rvv::core::add2d(vA.data(), vB.data(), vC.data(),
                     info.shape[0], info.shape[1]);
    return vec_to_mat(vC, info.shape[0], info.shape[1]);
}

MatF py_scale2d(MatF A, float k) {
    auto vA = mat_to_vec(A);
    std::vector<float> vB(vA.size());
    py::buffer_info info = A.request();
    rvv::core::scale2d(vA.data(), k, vB.data(),
                       info.shape[0], info.shape[1]);
    return vec_to_mat(vB, info.shape[0], info.shape[1]);
}

MatF py_matmul(MatF A, MatF B) {
    py::buffer_info infoA = A.request();
    py::buffer_info infoB = B.request();
    if (infoA.ndim != 2 || infoB.ndim != 2)
        throw std::runtime_error("Need 2-D arrays");
    std::size_t rows = infoA.shape[0];
    std::size_t k    = infoA.shape[1];
    std::size_t cols = infoB.shape[1];
    if (infoB.shape[0] != k)
        throw std::runtime_error("Incompatible shapes for matmul");
    auto vA = mat_to_vec(A);
    auto vB = mat_to_vec(B);
    std::vector<float> vC(rows * cols);
    rvv::core::matmul(vA.data(), vB.data(), vC.data(), rows, k, cols);
    return vec_to_mat(vC, rows, cols);
}

MatF py_transpose(MatF A) {
    py::buffer_info info = A.request();
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];
    auto vA = mat_to_vec(A);
    std::vector<float> vB(rows * cols);
    rvv::core::transpose(vA.data(), vB.data(), rows, cols);
    return vec_to_mat(vB, cols, rows);
}

py::array_t<float> py_mv(MatF A, py::array_t<float> x) {
    py::buffer_info infoA = A.request();
    auto vx = np_to_vec_f32(x);
    if (infoA.ndim != 2 || vx.size() != infoA.shape[1])
        throw std::runtime_error("Shape mismatch");
    std::size_t rows = infoA.shape[0];
    std::size_t cols = infoA.shape[1];
    auto vA = mat_to_vec(A);
    std::vector<float> vy(rows);
    rvv::core::mv(vA.data(), vx.data(), vy.data(), rows, cols);
    return vec_to_np_f32(vy);
}

//--------------------------------------
// int8 向量运算（新增）
//--------------------------------------
py::array_t<int8_t> py_add_i8(py::array_t<int8_t> a, py::array_t<int8_t> b) {
    auto va = np_to_vec_i8(a);
    auto vb = np_to_vec_i8(b);
    if (va.size() != vb.size()) {
        throw std::invalid_argument(
            "[add_i8] shape mismatch: a.size=" + std::to_string(va.size()) +
            " vs b.size=" + std::to_string(vb.size()));
    }
    std::vector<int8_t> vc(va.size());
    rvv::core::add_i8(va.data(), vb.data(), vc.data(), va.size());
    return vec_to_np_i8(vc);
}

py::array_t<int8_t> py_scale_i8(py::array_t<int8_t> a, int8_t k) {
    auto va = np_to_vec_i8(a);
    std::vector<int8_t> vb(va.size());
    rvv::core::scale_i8(va.data(), k, vb.data(), va.size());
    return vec_to_np_i8(vb);
}

int32_t py_dot_i8(py::array_t<int8_t> a, py::array_t<int8_t> b) {
    auto va = np_to_vec_i8(a);
    auto vb = np_to_vec_i8(b);
    if (va.size() != vb.size()) {
        throw std::invalid_argument(
            "[dot_i8] shape mismatch: a.size=" + std::to_string(va.size()) +
            " vs b.size=" + std::to_string(vb.size()));
    }
    return rvv::core::dot_i8(va.data(), vb.data(), va.size());
}

//--------------------------------------
// int8 矩阵运算
//--------------------------------------
using MatI8 = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;

std::vector<int8_t> mat_to_vec_i8(const MatI8& m) {
    py::buffer_info info = m.request();
    if (info.ndim != 2)
        throw std::invalid_argument("[mat_i8] Need 2-D array");
    if (info.format != "b")
        ERR_TYPE("int8", info.format);
    return std::vector<int8_t>(static_cast<int8_t*>(info.ptr),
                               static_cast<int8_t*>(info.ptr) + info.size);
}

MatI8 vec_to_mat_i8(const std::vector<int8_t>& v, std::size_t rows, std::size_t cols) {
    MatI8 out({rows, cols});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

MatI8 py_add2d_i8(MatI8 A, MatI8 B) {
    auto vA = mat_to_vec_i8(A);
    auto vB = mat_to_vec_i8(B);
    if (vA.size() != vB.size()) {
        throw std::invalid_argument(
            "[add2d_i8] shape mismatch: A.total=" + std::to_string(vA.size()) +
            " vs B.total=" + std::to_string(vB.size()));
    }
    std::vector<int8_t> vC(vA.size());
    py::buffer_info info = A.request();
    rvv::core::add2d_i8(vA.data(), vB.data(), vC.data(),
                        info.shape[0], info.shape[1]);
    return vec_to_mat_i8(vC, info.shape[0], info.shape[1]);
}

MatI8 py_scale2d_i8(MatI8 A, int8_t k) {
    auto vA = mat_to_vec_i8(A);
    std::vector<int8_t> vB(vA.size());
    py::buffer_info info = A.request();
    rvv::core::scale2d_i8(vA.data(), k, vB.data(),
                          info.shape[0], info.shape[1]);
    return vec_to_mat_i8(vB, info.shape[0], info.shape[1]);
}


//--------------------------------------
// Python 模块定义
//--------------------------------------
PYBIND11_MODULE(rvv, m) {
    m.doc() = "SG2002 RVV 0.7.1 加速库，兼容 NumPy（支持 float32 / int8）";

    // ---------- float32 ----------
    m.def("add",      &py_add,      "向量加法");
    m.def("sub",      &py_sub,      "向量减法");
    m.def("scale",    &py_scale,    "标量乘法");
    m.def("dot",      &py_dot,      "点积");
    m.def("norm_l2",  &py_norm_l2,  "L2 范数");
    m.def("normalize",&py_normalize,"向量归一化");

    // ---------- float32 通用扩展 ----------
    m.def("abs_f32", &py_abs_f32, py::arg("a"),
          "逐元素绝对值");
    m.def("sum_f32", &py_sum_f32, py::arg("a"),
          "float32 求和，double 累加；空向量返回 0.0");
    m.def("mean_std_f32", &py_mean_std_f32, py::arg("a"),
          "返回 (mean, population_std)；空向量返回 (nan, nan)");
    m.def("mul_f32", &py_mul_f32, py::arg("a"), py::arg("b"),
          "逐元素乘法，两个输入必须等长");
    m.def("add_scalar_f32", &py_add_scalar_f32,
          py::arg("a"), py::arg("k"), "逐元素加标量");
    m.def("clip_f32", &py_clip_f32, py::arg("a"), py::arg("lo"),
          py::arg("hi"), "逐元素截断到 [lo, hi]，保留 NaN；要求 lo <= hi");
    m.def("max_f32", &py_max_f32, py::arg("a"),
          "最大值；空数组或含 NaN 时返回 NaN");
    m.def("min_f32", &py_min_f32, py::arg("a"),
          "最小值；空数组或含 NaN 时返回 NaN");

    m.def("add2d",    &py_add2d,    "矩阵加法");
    m.def("scale2d",  &py_scale2d,  "矩阵标量乘法");
    m.def("matmul",   &py_matmul,   "矩阵乘法");
    m.def("transpose",&py_transpose,"矩阵转置");
    m.def("mv",       &py_mv,       "矩阵 × 向量");

    // ---------- int8 ----------
    m.def("add_i8",      &py_add_i8,      "int8 向量加法");
    m.def("scale_i8",    &py_scale_i8,    "int8 标量乘法");
    m.def("dot_i8",      &py_dot_i8,      "int8 点积");
    m.def("add2d_i8",    &py_add2d_i8,    "int8 矩阵加法");
    m.def("scale2d_i8",  &py_scale2d_i8,  "int8 矩阵标量乘法");
}
