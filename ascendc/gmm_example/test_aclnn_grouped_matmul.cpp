#include <iostream>
#include <vector>
#include <memory>
#include <random>

#include "aclnnop/aclnn_grouped_matmul.h"
#include "acl/acl.h"
#include "acl/acl_tensor.h"

#define CHECK_RET(express, ret, message) \
    do { \
        if (!(express)) { \
            std::cerr << "[ERROR] " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return ret; \
        } \
    } while (0)

#define INFO_LOG(message) \
    do { \
        std::cout << "[INFO] " << message << std::endl; \
    } while (0)

// 计算形状乘积
template <typename T>
T GetShapeSize(const std::vector<T> &shape) {
    T size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// 将device结果拷回host并打印
template <typename T>
void PrintOutResult(const std::string &name, const T *data, const std::vector<int64_t> &shape) {
    INFO_LOG(name << " shape: [");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i != shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    size_t size = GetShapeSize(shape);
    INFO_LOG(name << " data: [");
    for (size_t i = 0; i < std::min(size, static_cast<size_t>(10)); ++i) {
        std::cout << data[i];
        if (i != std::min(size, static_cast<size_t>(10)) - 1) {
            std::cout << ", ";
        }
    }
    if (size > 10) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

// 初始化ACL runtime
int Init(uint32_t deviceId, aclrtStream *stream) {
    // 初始化ACL
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclInit failed");
    INFO_LOG("aclInit success");

    // 设置设备
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtSetDevice failed");
    INFO_LOG("aclrtSetDevice success");

    // 创建流
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtCreateStream failed");
    INFO_LOG("aclrtCreateStream success");

    return ACL_ERROR_NONE;
}

// 清理ACL runtime
int Finalize(uint32_t deviceId, aclrtStream stream) {
    aclError ret = ACL_ERROR_NONE;

    // 销毁流
    if (stream != nullptr) {
        ret = aclrtDestroyStream(stream);
        CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtDestroyStream failed");
        INFO_LOG("aclrtDestroyStream success");
    }

    // 释放设备
    ret = aclrtResetDevice(deviceId);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtResetDevice failed");
    INFO_LOG("aclrtResetDevice success");

    // 终结ACL
    ret = aclFinalize();
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclFinalize failed");
    INFO_LOG("aclFinalize success");

    return ACL_ERROR_NONE;
}

// 创建ACL张量
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
                   aclTensor **tensor, void **deviceAddr) {
    // 计算数据大小
    size_t dataSize = GetShapeSize(shape) * sizeof(T);

    // 分配device内存
    aclError ret = aclrtMalloc(deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtMalloc failed");

    // 将数据从host拷贝到device
    ret = aclrtMemcpy(*deviceAddr, dataSize, hostData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtMemcpy failed");

    // 计算strides（假设是连续张量）
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // 创建ACL张量
    aclTensorFormat format = ACL_FORMAT_ND;
    aclDataType dtype = (sizeof(T) == 2) ? ACL_FLOAT16 : ACL_FLOAT;
    ret = aclCreateTensor(tensor, shape.size(), shape.data(), strides.data(), dtype, format);
    CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclCreateTensor failed");

    return ACL_ERROR_NONE;
}

// 销毁ACL张量
template <typename T>
int DestroyAclTensor(aclTensor *tensor, void *deviceAddr) {
    aclError ret = ACL_ERROR_NONE;

    // 销毁张量
    if (tensor != nullptr) {
        ret = aclDestroyTensor(tensor);
        CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclDestroyTensor failed");
    }

    // 释放device内存
    if (deviceAddr != nullptr) {
        ret = aclrtFree(deviceAddr);
        CHECK_RET(ret == ACL_ERROR_NONE, ret, "aclrtFree failed");
    }

    return ACL_ERROR_NONE;
}

int main() {
    // 设置设备ID
    uint32_t deviceId = 0;
    aclrtStream stream = nullptr;

    // 初始化ACL runtime
    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "Init failed");

    // 定义分组矩阵乘法的参数
    // 假设我们有2个分组
    int groupNum = 2;
    // 每个分组的M值
    std::vector<int64_t> mPerGroup = {16, 16}; // 总M=32
    // 每个分组的N值
    std::vector<int64_t> nPerGroup = {32, 32}; // 总N=64
    // 每个分组的K值
    std::vector<int64_t> kPerGroup = {64, 64}; // 总K=128

    // 计算总形状
    int64_t totalM = 0, totalN = 0, totalK = 0;
    for (int i = 0; i < groupNum; ++i) {
        totalM += mPerGroup[i];
        totalN += nPerGroup[i];
        totalK += kPerGroup[i];
    }

    // 定义输入输出形状
    // 输入x的形状为[totalM, totalK]
    std::vector<int64_t> xShape = {totalM, totalK}; // [32, 128]
    // 权重weight的形状为[totalK, totalN]（不转置的情况下）
    std::vector<int64_t> weightShape = {totalK, totalN}; // [128, 64]
    // 偏置bias的形状为[totalN]
    std::vector<int64_t> biasShape = {totalN}; // [64]
    // 输出y的形状为[totalM, totalN]
    std::vector<int64_t> yShape = {totalM, totalN}; // [32, 64]

    // 生成随机输入数据（使用float16类型）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 生成输入x数据
    size_t xSize = GetShapeSize(xShape);
    std::vector<uint16_t> xHostData(xSize);
    for (size_t i = 0; i < xSize; ++i) {
        xHostData[i] = *reinterpret_cast<uint16_t*>(&dist(gen));
    }

    // 生成权重weight数据
    size_t weightSize = GetShapeSize(weightShape);
    std::vector<uint16_t> weightHostData(weightSize);
    for (size_t i = 0; i < weightSize; ++i) {
        weightHostData[i] = *reinterpret_cast<uint16_t*>(&dist(gen));
    }

    // 生成偏置bias数据
    size_t biasSize = GetShapeSize(biasShape);
    std::vector<uint16_t> biasHostData(biasSize);
    for (size_t i = 0; i < biasSize; ++i) {
        biasHostData[i] = *reinterpret_cast<uint16_t*>(&dist(gen));
    }

    // 创建ACL张量
    aclTensor *xTensor = nullptr;
    void *xDeviceAddr = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xTensor, &xDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "CreateAclTensor for x failed");

    aclTensor *weightTensor = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightHostData, weightShape, &weightTensor, &weightDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "CreateAclTensor for weight failed");

    aclTensor *biasTensor = nullptr;
    void *biasDeviceAddr = nullptr;
    ret = CreateAclTensor(biasHostData, biasShape, &biasTensor, &biasDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "CreateAclTensor for bias failed");

    aclTensor *yTensor = nullptr;
    void *yDeviceAddr = nullptr;
    ret = CreateAclTensor(std::vector<uint16_t>(GetShapeSize(yShape)), yShape, &yTensor, &yDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "CreateAclTensor for y failed");

    // 设置分组信息
    std::vector<int64_t> splitItem;
    for (int i = 0; i < groupNum; ++i) {
        splitItem.push_back(mPerGroup[i]);
        splitItem.push_back(kPerGroup[i]);
        splitItem.push_back(nPerGroup[i]);
    }

    // 获取workspace大小
    size_t workspaceSize = 0;
    ret = aclnnGroupedMatmulGetWorkspaceSize(xTensor, weightTensor, biasTensor, yTensor,
                                            splitItem.data(), splitItem.size(), 0, 0, &workspaceSize, nullptr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclnnGroupedMatmulGetWorkspaceSize failed");
    INFO_LOG("Workspace size: " << workspaceSize << " bytes");

    // 分配workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclrtMalloc workspace failed");
    }

    // 创建执行器
    aclnnExecutable *executor = nullptr;
    ret = aclnnCreateExecutor(&executor);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclnnCreateExecutor failed");

    // 执行GMM算子
    ret = aclnnGroupedMatmul(workspaceAddr, workspaceSize, executor, stream, xTensor, weightTensor,
                           biasTensor, yTensor, splitItem.data(), splitItem.size(), 0, 0);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclnnGroupedMatmul failed");

    // 同步流
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclrtSynchronizeStream failed");

    // 拷贝输出结果到host
    std::vector<uint16_t> yHostData(GetShapeSize(yShape));
    ret = aclrtMemcpy(yHostData.data(), GetShapeSize(yShape) * sizeof(uint16_t),
                     yDeviceAddr, GetShapeSize(yShape) * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclrtMemcpy for y failed");

    // 打印结果
    PrintOutResult("Input x", reinterpret_cast<float*>(xHostData.data()), xShape);
    PrintOutResult("Output y", reinterpret_cast<float*>(yHostData.data()), yShape);

    // 清理资源
    ret = DestroyAclTensor<uint16_t>(xTensor, xDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "DestroyAclTensor for x failed");

    ret = DestroyAclTensor<uint16_t>(weightTensor, weightDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "DestroyAclTensor for weight failed");

    ret = DestroyAclTensor<uint16_t>(biasTensor, biasDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "DestroyAclTensor for bias failed");

    ret = DestroyAclTensor<uint16_t>(yTensor, yDeviceAddr);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "DestroyAclTensor for y failed");

    if (workspaceAddr != nullptr) {
        ret = aclrtFree(workspaceAddr);
        CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclrtFree workspace failed");
    }

    if (executor != nullptr) {
        ret = aclnnDestroyExecutor(executor);
        CHECK_RET(ret == ACL_ERROR_NONE, 1, "aclnnDestroyExecutor failed");
    }

    // 清理ACL runtime
    ret = Finalize(deviceId, stream);
    CHECK_RET(ret == ACL_ERROR_NONE, 1, "Finalize failed");

    INFO_LOG("Grouped Matmul example completed successfully");
    return 0;
}