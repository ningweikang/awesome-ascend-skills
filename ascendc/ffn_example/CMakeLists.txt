cmake_minimum_required(VERSION 3.14)
project(ffn_example)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找ACL库
find_library(ACL_LIB acl HINTS $ENV{ASCEND_HOME}/acllib/lib64)
find_library(ACLRT_LIB aclrt HINTS $ENV{ASCEND_HOME}/acllib/lib64)
find_library(ACLNNOPS_LIB aclnnops HINTS $ENV{ASCEND_HOME}/acllib/lib64)

# 包含头文件目录
include_directories($ENV{ASCEND_HOME}/acllib/include)
include_directories($ENV{ASCEND_HOME}/acllib/include/aclnnop)

# 创建可执行文件
add_executable(test_aclnn_ffn test_aclnn_ffn.cpp)

# 链接库
target_link_libraries(test_aclnn_ffn ${ACL_LIB} ${ACLRT_LIB} ${ACLNNOPS_LIB})
