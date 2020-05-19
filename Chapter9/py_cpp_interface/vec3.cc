#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
namespace py = pybind11;

template<typename T>
class vec3 {
public:
    vec3(T x, T y, T z): _x(x), _y(y), _z(z){};

    vec3<T> operator+(const vec3<T>& o) const {
        return vec3<T>(_x + o._x, _y + o._y, _z + o._z);
    }

    vec3<T> operator-(const vec3<T>& o) const {
        return vec3<T>(_x - o._x, _y - o._y, _z - o._z);
    }

    vec3<T> operator*(T s) const {
        return vec3<T>(_x * s, _y * s,  _z * s);
    }

    friend vec3<T> operator*(T s, const vec3<T>& v) {
        return v*s;
    }

    std::string toString() const {
        return "vec3: [" + std::to_string(_x) + ","
                         + std::to_string(_y) + ","
                         + std::to_string(_z) + "]";
    }

    std::tuple<T, T, T> toTuple() const {
        return std::make_tuple(_x, _y, _z);
    }
private:
    T _x, _y, _z;
};

PYBIND11_MODULE(vec3, m) {
    m.doc() = "Vector class examples";
    py::class_<vec3<float>>(m, "Vec3")
        .def(py::init<float, float, float>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def("to_tuple", &vec3<float>::toTuple)
        .def("__repr__", &vec3<float>::toString);
}
