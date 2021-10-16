#include <cmath>

static float lower_bound(float center_f32, double eps_f64) {
    double center_f64 = center_f32;
    float ret_f32 = std::nextafterf(center_f64 - eps_f64, -1);
    // find a lower bound by over estimating
    while (static_cast<double>(center_f32 - ret_f32) <= eps_f64 &&
           ret_f32 >= 0) {
        ret_f32 = std::nextafterf(ret_f32, -1);
    }
    if (ret_f32 <= 0) {
        ret_f32 = 0;
    }
    while (static_cast<double>(center_f32 - ret_f32) > eps_f64 ||
           center_f64 - static_cast<double>(ret_f32) > eps_f64) {
        ret_f32 = std::nextafterf(ret_f32, 1);
    }
    return ret_f32;
}

static float upper_bound(float center_f32, double eps_f64) {
    double center_f64 = center_f32;
    float ret_f32 = std::nextafterf(center_f64 + eps_f64, 2);
    while (static_cast<double>(ret_f32 - center_f32) <= eps_f64 &&
           ret_f32 <= 1) {
        ret_f32 = std::nextafterf(ret_f32, 2);
    }
    if (ret_f32 >= 1) {
        ret_f32 = 1;
    }
    while (static_cast<double>(ret_f32 - center_f32) > eps_f64 ||
           static_cast<double>(ret_f32) - center_f64 > eps_f64) {
        ret_f32 = std::nextafterf(ret_f32, 0);
    }
    return ret_f32;
}
