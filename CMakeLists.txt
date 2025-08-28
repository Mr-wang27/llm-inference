cmake_minimum_required(VERSION 3.2 FATAL_ERROR) # 第三个参数表示如果版本参数过低，则终止
project(oneLLM LANGUAGES CXX CUDA)  # 项目名称oneLLM 使用到的语言CXX，CUDA

find_package(CUDAToolkit REQUIRED) # 查找依赖包，用于查找本地或系统的库/工具包
# REQUIRTED表示必须找到，否则会报错退出(强制依赖)
# CMake 会自动设置 CUDA_TOOLKIT_ROOT_DIR 变量，例如：/usr/local/cuda
# 

# 设置变量：
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS "CUDA_PATH = ${CUDA_PATH}")      # CUDA_PATH = /usr/local/cuda

# 找到CUDA的Cmake_module_apth
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# 设置编译器选项
set(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS}") # 将宏的值赋值给自己
set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS}") # 没意义，可以删除

# -Xcompiler -Wall表示：把-Xcompailer后面的-Wall传递给主机(CPU)编译器。 -Wall是GCC/Clang的警告信息开关
# 这是CUDA C++编译时的常规做法
set(CMAKE_CUDA_FLAGS    "${CMAKE_CUDA_FLAGS}    -Xcompiler -Wall")# 添加了两个参数：

# 设置CUDA支持的显卡架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                       ")
#                       -rdc=true") # not sure the effect of this option, retain it temply

# 这一行的作用与前面set(CMAKE_CUDA_FLAGS )的作用是相同的。会自动生成上面手写的内容
# 都是设置CUDA编译器的标志和设置Cmake中的CUDA_ARCHITECTURES 参数
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
message("-- Assign GPU architecture (sm=70 75 80 86)")




# 设置debug模式下的编译器选项(flags),以便在调试时使用合适的编译选项
# 设置C编译器的Debug标志，其中-Wall表示:启用所有警告信息
# -O0表示：禁用优化，确保在调试时代码行为更易预测。因此编译器不会优化掉变量等内容
set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -Wall -O0")
# 设置CUDA编译器的Debug标志
# -O0表示禁用优化，确保编译器不会对CUDA代码进行优化
# -G:启用调式信息的生成，这样可以进行源代码级别的调试，例如使用cuda-gdb
# -Xcompiler -Wall :将-Wall传递给C++编译器，用以启用警告信息
set(CMAKE_CUDA_FLAGS_DEBUG  "${CMAKE_CUDA_FLAGS_DEBUG}  -O0 -G -Xcompiler -Wall")
# 这也是常规的CUDA C++编译写法
message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})



# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # 确保cmake强制要求编译器支持指定的C++标准，如果不支持，cmake就会报错并停止配置


# 根据所选择的C++版本来配置CUDA编译器选项：也是常见基本固定的写法
if(CMAKE_CXX_STANDARD STREQUAL "11")
    # 将--expt-extended-lambda标志添加到CUDA编译器选项中，启用扩展的lambda表达式支持
    # 时CUDA编译器可以支持更多的C++ 11 lambda的特性。例如通过捕获默认参数、可变参数等增强lambda表达式能力
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    # 将--expt-relaxed-constexpr标志添加到CUDA编译器选项中
    # --expt-relaxed-constexpr启用对C++11 constexpr的放宽支持
    # 使得CUDA编译器能够根号地处理constexpr函数和常量表达式
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    # 将--std=c++11标志添加到CUDA编译器选项中,告诉CUDA编译器使用C++11标准进行编译
    # 确保CUDA内核代码中的C++11特性能够得到正确的编译和支持
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()



# 这也是常规写法
# 设置构建项目的Release配置中的优化级别
# -O3时最高级别的优化，编译器会启用所有优化选项：包括循环展开、内联、向量化等
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# 为CUDA编译器的Release构建配置添加-O3优化，并传递给主机编译器。
# -Xcompiler： 用于将后续的编译器选项传递给主机编译器(即非CUDA编译器)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")




# CMAKE_BINARY_DIR时build目录
# 这里的目录是哪个目录不是很清楚？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
# 这也是常规写法
# 指定构建过程中生成的不同类型的输出文件的保存目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # 动态库文件等(linux下，动态库也有可执行权限)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # 静态库文件等
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # 可执行文件




# 设置头文件路径
set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CUDA_PATH}/include
)
message(STATUS "COMMON_HEADER_DIRS = ${COMMON_HEADER_DIRS}")
# COMMON_HEADER_DIRS = /home/llm_inference;/usr/local/cuda/include


# 设置需要使用到的库文件路径
set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
)


# 设置inlude的查找路径
include_directories(
    ${COMMON_HEADER_DIRS}
)

# 生成可执行程序之前，需要指定要连接的库的路径(静态库可以不指定，但是动态库一定要指定)
# 设置连接的库文件路径
link_directories(
    ${COMMON_LIB_DIRS}
)



# 定义三个选项开关，可以在CMake配置时手动开启或关闭
# cmake -DPERF=ON -DPRINT_DATA=ON -DSAVE_DATA=OFF ..
option (PERF
    "measure model inference performance"
    OFF # 默认值为关闭
)
option(PRINT_DATA
    "print kernel output to debug"
    OFF
)
option(SAVE_DATA
    "save kernel output to debug"
    OFF
)
# 这三个选项开关配合这三个条件编译
#cmake .. -DPRINT_DATA=ON && make
#cmake .. -DPRINT_DATA=ON -DSAVE_DATA=ON && make
#cmake .. -DPERF=ON && make
#cmake .. && make
if (PERF)
    add_compile_options(-DPERF)
endif()
if(PRINT_DATA)
    add_compile_options(-DPRINT_DATA)
endif()
if(SAVE_DATA)
    add_compile_options(-DSAVE_DATA)
endif()

# 添加条件编译选项：add_compile_options
# 根据开关的选项是否打开，添加对应的宏定义到编译选项中
# 添加效果：
# -DPERF：等价于在代码中写：#define PERF
# -DPRINT_DATA: 等价于： #define PRINT_DATA
# -DSAVE_DATA:  等价于： #define SAVE_DATA
# 而这些宏在代码中用来控制条件编译
# #ifdef PERF
#   //性能统计相关代码
# #endif
# 如果关系PERF,即没有在Cmake编译参数中设置PERF开关为ON，将不会添加编译参数-DPREF
# 也就不会在代码中定义宏PERF，那么性能统计相关的代码也就不会被编译
# 反之同理





# 递归搜索项目src目录下的源文件
file(GLOB_RECURSE LLM_CXX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LLM_CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)




# 动态库： linux:     libxxxxxxxxx.so
#        windows:    libxxxxxxxxx.dll

# 静态库： linux:     libxxxxxxxxx.a
#        windows:    libxxxxxxxxx.lib
# **制作静态库：add_library(库名称 STATIC 源文件1 [源文件2] ...)**
# **制作动态库：add_library(库名称 SHARED 源文件1 [源文件2] ...)**
# **制作目标文件：add_library(库名称 OBJECT 源文件1 [源文件2] ...)**
# 制作目标文件表示只会生成.o（linux）或.obj(windows)文件
# 目标文件可以被其他库或者可执行文件连接和复用
# 多用于模块化编译、多目标文件复用、分离构建
# 将前面搜索到的源文件全部编译成目标文件，用于后面的链接
add_library(llmengine OBJECT
            ${LLM_CXX_SOURCES}
            ${LLM_CUDA_SOURCES}
)
# add_library(llmengine STATIC
#             ${LLM_CXX_SOURCES}
#             ${LLM_CUDA_SOURCES}
# )


# 设置所有静态库 (.a) 的输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# 如果你有动态库 (.so)，也设置一下
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# 设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 你也可以针对不同的构建类型 (Debug, Release 等) 进行设置，例如：
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/Debug")
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/Release")
# ... 类似地设置 CMAKE_LIBRARY_OUTPUT_DIRECTORY_... 和 CMAKE_RUNTIME_OUTPUT_DIRECTORY_...


# 构建CMakeLists.txt的子目录
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)


# 生成可执行文件
add_executable(main user_entry.cpp)

# 链接动态库，链接CUDA的runtime
# 前面link_directories指定了链接库的地址
# llmengine是所有文件构建的目标文件，直接进行连接即可
# target_link_libraries(main PUBLIC -lcublas -lcudart -lcudadevrt llmengine Llama llamaweights cublasWrapper)
target_link_libraries(main PUBLIC -lcublas -lcudart -lcudadevrt llmengine)
# 将 build/lib 目录添加到 'main' 目标的链接器搜索路径中
# target_link_directories(main PRIVATE "${CMAKE_BINARY_DIR}/lib")
# target_link_libraries(main PUBLIC -lcublas -lcudart -lcudadevrt Llama llamaweights layerweights weightutils LlamaCtxdecoder Llamaselfdecoder Llamaffn LlamaselfAttn LlamaCtxAttn cublasWrapper)






# 现在还没有user_entry.cpp文件，所以还不能直接使用这个cmakelists.txt文件进行构建








# 在cmake进行编译的时候，添加参数 -DCMAKE_BUILD_TYPE=debug/release 自定编译模式