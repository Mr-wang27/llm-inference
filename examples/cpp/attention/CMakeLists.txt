cmake_minimum_required(VERSION 3.8)

add_executable(CtxAttnExample context_attn_example.cpp)
set_property(TARGET CtxAttnExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CtxAttnExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CtxAttnExample PUBLIC
                             -lcudart
                             -lcudadevrt
                             LlamaCtxAttn)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(CtxAttnExample PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()


                    
add_executable(selfAttnExample self_attn_example.cpp)
set_property(TARGET selfAttnExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET selfAttnExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(selfAttnExample PUBLIC
                            -lcudart
                            -lcudadevrt                             
                            LlamaselfAttn
			                linear)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(selfAttnExample PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()