cmake_minimum_required(VERSION 3.8)

# 编译成静态库
add_library(LlamaCtxdecoder STATIC context_decoder.cpp)
set_property(TARGET LlamaCtxdecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaCtxdecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaCtxdecoder PUBLIC
                                            LlamaCtxAttn,
                                            Llamaffn,
                                            # layerweights,
                                            llamaweights,
                                            build_casual_mask,
                                            cal_paddingoffset,
                                            fused_addresidual_norm,
                                            add_residual,
                                            rmsnorm
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(LlamaselfAttn PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()





add_library(Llamaselfdecoder STATIC self_decoder.cpp)
set_property(TARGET Llamaselfdecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llamaselfdecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llamaselfdecoder PUBLIC
                                            LlamaselfAttn,
                                            Llamaffn,
                                            # layerweights,
                                            llamaweights,
                                            fused_addresidual_norm,
                                            add_residual,
                                            rmsnorm
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(LlamaselfAttn PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()