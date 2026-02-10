#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <sstream>
#include <string>

#include "rvv.hpp"

namespace py = pybind11;

namespace {

// We keep NumPy support when available (py::array_t), but we also accept generic
// buffer-protocol objects (including numpy.ndarray) so the wheel doesn't have
// to depend on the numpy Python package.
struct F32Buffer {
    const float* data = nullptr;
    py::ssize_t ndim = 0;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    size_t size = 0; // number of float elements
    py::object owner;
};

static F32Buffer as_f32_buffer(const py::handle& obj, const char* name, int expected_ndim = -1) {
    // Try NumPy-style conversion first (works when NumPy is installed).
    try {
        using ArrF32C = py::array_t<float, py::array::c_style | py::array::forcecast>;
        ArrF32C arr = py::cast<ArrF32C>(obj);
        if (expected_ndim >= 0 && arr.ndim() != expected_ndim) {
            std::ostringstream oss;
            oss << name << " must have ndim=" << expected_ndim << ", got ndim=" << arr.ndim();
            throw py::value_error(oss.str());
        }
        py::buffer_info info = arr.request();

        F32Buffer out;
        out.owner = arr;
        out.ndim = info.ndim;
        out.shape.assign(info.shape.begin(), info.shape.end());
        out.strides.assign(info.strides.begin(), info.strides.end());
        out.size = (size_t)info.size;
        out.data = static_cast<const float*>(info.ptr);
        return out;
    } catch (const py::cast_error&) {
        // Fall through to buffer protocol.
    }

    // Generic buffer protocol path.
    // Requirements:
    // - format must be 32-bit float ('f')
    // - itemsize == 4
    // - C-contiguous
    py::buffer buf = py::reinterpret_borrow<py::buffer>(obj);
    py::buffer_info info = buf.request();

    if (expected_ndim >= 0 && info.ndim != expected_ndim) {
        std::ostringstream oss;
        oss << name << " must have ndim=" << expected_ndim << ", got ndim=" << info.ndim;
        throw py::value_error(oss.str());
    }

    if (info.itemsize != (py::ssize_t)sizeof(float)) {
        std::ostringstream oss;
        oss << name << " must be float32-compatible buffer (itemsize=4), got itemsize=" << info.itemsize;
        throw py::type_error(oss.str());
    }

    // PEP 3118 format: 'f' means C float.
    if (info.format != "f") {
        std::ostringstream oss;
        oss << name << " must be float32-compatible buffer (format='f'), got format='" << info.format << "'";
        throw py::type_error(oss.str());
    }

    // Check C-contiguous.
    py::ssize_t expected_stride = info.itemsize;
    for (py::ssize_t i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != expected_stride) {
            throw py::value_error(std::string(name) + " must be C-contiguous float32 buffer");
        }
        expected_stride *= info.shape[i];
        if (i == 0) break;
    }

    F32Buffer out;
    out.owner = buf;
    out.ndim = info.ndim;
    out.shape.assign(info.shape.begin(), info.shape.end());
    out.strides.assign(info.strides.begin(), info.strides.end());
    out.size = (size_t)info.size;
    out.data = static_cast<const float*>(info.ptr);
    return out;
}

static py::array_t<float> alloc_out_1d(py::ssize_t n) { return py::array_t<float>(n); }

} // namespace

PYBIND11_MODULE(rvv, m) {
    m.doc() = "RVV accelerated vector/matrix ops (NumPy ndarray-compatible, float32).";

    m.def("add", [](py::handle a_obj, py::handle b_obj) {
        auto a = as_f32_buffer(a_obj, "a", 1);
        auto b = as_f32_buffer(b_obj, "b", 1);
        if (a.shape[0] != b.shape[0]) throw py::value_error("add: a and b must have the same length");
        auto out = alloc_out_1d(a.shape[0]);
        rvv::add_f32(a.data, b.data, out.mutable_data(), (size_t)a.shape[0]);
        return out;
    });

    m.def("sub", [](py::handle a_obj, py::handle b_obj) {
        auto a = as_f32_buffer(a_obj, "a", 1);
        auto b = as_f32_buffer(b_obj, "b", 1);
        if (a.shape[0] != b.shape[0]) throw py::value_error("sub: a and b must have the same length");
        auto out = alloc_out_1d(a.shape[0]);
        rvv::sub_f32(a.data, b.data, out.mutable_data(), (size_t)a.shape[0]);
        return out;
    });

    m.def("scale", [](py::handle a_obj, float k) {
        auto a = as_f32_buffer(a_obj, "a", 1);
        auto out = alloc_out_1d(a.shape[0]);
        rvv::scale_f32(a.data, k, out.mutable_data(), (size_t)a.shape[0]);
        return out;
    });

    m.def("dot", [](py::handle a_obj, py::handle b_obj) {
        auto a = as_f32_buffer(a_obj, "a", 1);
        auto b = as_f32_buffer(b_obj, "b", 1);
        if (a.shape[0] != b.shape[0]) throw py::value_error("dot: a and b must have the same length");
        return rvv::dot_f32(a.data, b.data, (size_t)a.shape[0]);
    });

    m.def("norm_l2", [](py::handle a_obj) {
        auto a = as_f32_buffer(a_obj, "a", 1);
        return rvv::norm_l2_f32(a.data, (size_t)a.shape[0]);
    });

    m.def("matmul", [](py::handle A_obj, py::handle B_obj) {
        auto A = as_f32_buffer(A_obj, "A", 2);
        auto B = as_f32_buffer(B_obj, "B", 2);
        if (A.shape[1] != B.shape[0]) {
            std::ostringstream oss;
            oss << "matmul: shape mismatch: A is (" << A.shape[0] << "," << A.shape[1] << ") but B is (" << B.shape[0] << "," << B.shape[1]
                << ")";
            throw py::value_error(oss.str());
        }
        py::array_t<float> C({A.shape[0], B.shape[1]});
        rvv::matmul_f32(A.data, B.data, C.mutable_data(), (size_t)A.shape[0], (size_t)A.shape[1], (size_t)B.shape[1]);
        return C;
    });
}
