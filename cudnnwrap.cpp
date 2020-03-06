#include <cudnn.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdlib>

#include <atomic>
#include <fstream>
#include <string>
#include <unordered_map>

#define SSPRINTF(fmt...)                 \
    ({                                   \
        char msg[256];                   \
        snprintf(msg, sizeof(msg), fmt); \
        msg;                             \
    })

[[noreturn]] void report_error(const char* msg) {
    fprintf(stderr, "fatal error from cudnn wrapper: %s\n", msg);
    abort();
}

namespace {
class CudnnLibHandle {
    void* m_handle = nullptr;

    static std::string locate_cudnn_lib() {
        std::ifstream fin{"/proc/self/maps"};
        for (std::string line; std::getline(fin, line);) {
            if (line.find("cudnn") != std::string::npos) {
                return line.substr(line.rfind(" ") + 1);
            }
        }
        report_error("can not find cudnn");
    }

    CudnnLibHandle() {
        auto cudnn = locate_cudnn_lib();
        printf("found cudnn: %s\n", cudnn.c_str());
        void* handle = dlopen(cudnn.c_str(), RTLD_LAZY);
        if (!handle) {
            report_error(SSPRINTF("can not dlopen cudnn: %s", dlerror()));
        }
        m_handle = handle;
    }

public:
    ~CudnnLibHandle() {
        if (m_handle) {
            dlclose(m_handle);
        }
    }

    static CudnnLibHandle& inst() {
        static CudnnLibHandle obj;
        return obj;
    }

    void* resolve(const char* name) {
        if (m_handle) {
            auto ret = dlsym(m_handle, name);
            if (!ret) {
                report_error(SSPRINTF("cannot resolve symbol %s", name));
            }
            return ret;
        }
        report_error("attempt to resolve symbol from invalid handle");
    }
};

std::atomic_uint_fast32_t g_call_cnt;
std::atomic<cudnnConvolutionFwdAlgo_t> g_fwd_algo{
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT};

cudnnConvolutionFwdAlgo_t get_algo(cudnnConvolutionFwdAlgo_t setting) {
    auto g_setting = g_fwd_algo.load();
    if (g_setting != CUDNN_SEQDATA_DIM_COUNT) {
        return g_setting;
    }
    return setting;
}

#define GET_IMPL(func) \
    reinterpret_cast<decltype(&func)>(CudnnLibHandle::inst().resolve(#func))

bool is_algo_replaced() {
    return g_fwd_algo.load() != CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
}

}  // anonymous namespace

extern "C" {
void set_algo(const char* name) {
    static std::unordered_map<std::string, cudnnConvolutionFwdAlgo_t> algos{
#define ON(n) {#n, CUDNN_CONVOLUTION_FWD_ALGO_##n}
            ON(IMPLICIT_GEMM),
            ON(IMPLICIT_PRECOMP_GEMM),
            ON(GEMM),
            ON(DIRECT),
            ON(FFT),
            ON(FFT_TILING),
            ON(WINOGRAD),
            ON(WINOGRAD_NONFUSED),
            {"", CUDNN_CONVOLUTION_FWD_ALGO_COUNT},
#undef ON
    };
    g_fwd_algo = algos.at(name);
}

cudnnStatus_t cudnnConvolutionForward(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t xDesc, const void* x,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionFwdAlgo_t algo, void* workSpace,
        size_t workSpaceSizeInBytes, const void* beta,
        const cudnnTensorDescriptor_t yDesc, void* y) {
    static auto impl = GET_IMPL(cudnnConvolutionForward);
    g_call_cnt.fetch_add(1, std::memory_order_relaxed);
    algo = get_algo(algo);
    return impl(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
                workSpaceSizeInBytes, beta, yDesc, y);
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
        size_t* sizeInBytes) {
    static auto impl = GET_IMPL(cudnnGetConvolutionForwardWorkspaceSize);
    if (!is_algo_replaced()) {
        return impl(handle, xDesc, wDesc, convDesc, yDesc,
                    static_cast<cudnnConvolutionFwdAlgo_t>(algo), sizeInBytes);
    }

    // use max size to be safe
    size_t max_size = 0;
    for (int i = 0; i < CUDNN_CONVOLUTION_FWD_ALGO_COUNT; ++i) {
        size_t size = 0;
        auto status = impl(handle, xDesc, wDesc, convDesc, yDesc,
                           static_cast<cudnnConvolutionFwdAlgo_t>(i), &size);
        if (status != CUDNN_STATUS_SUCCESS) {
            return status;
        }
        if (size > max_size) {
            max_size = size;
        }
    }
    *sizeInBytes = max_size;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
        cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
        int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults) {
    static auto impl = GET_IMPL(cudnnGetConvolutionForwardAlgorithm_v7);
    auto ret = impl(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount,
                    returnedAlgoCount, perfResults);
    auto dst_algo = g_fwd_algo.load();

    if (dst_algo != CUDNN_CONVOLUTION_FWD_ALGO_COUNT &&
        ret == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < *returnedAlgoCount; ++i) {
            if (perfResults[i].algo == dst_algo) {
                perfResults[0] = perfResults[i];
                *returnedAlgoCount = 1;
                return ret;
            }
        }
        report_error("target algo not in perf results");
    }
    return ret;
}

int reset_call_cnt() {
    return g_call_cnt.exchange(0, std::memory_order_relaxed);
}
}  // extern "C"
