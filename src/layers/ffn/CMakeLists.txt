# 定义一个静态库
add_library(Llamaffn STATIC ffn.cpp)
set_property(TARGET Llamaffn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llamaffn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llamaffn PUBLIC
                             -lcudart
                             -lcudadevrt
#                             cublasWrapper
                             act
                             linear)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(Llamaffn PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()

