# 在这个目录下需要CMakeLists.txt进行管理
# 要将每一个测试文件编译成可执行文件进行测试
# 因为这里是单元测试，每个文件需要单独运行
# 并不需要编译到整体的LLMInferen项目中



# cudart:表示cuda runtime库(libcudart.so动态库)，是CUDA程序必须的运行时支持库
# cudadevrt:表示CUDA device runtime库(libcudadevrt.a静态库)，如果用到了device-side的runtime支持(如device global vars)就需要这个库
# embeddingFunctor 是在src/kernel目录下构建的OBJECT library或者静态库.a
# -l表示手动链接库
add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    embedding PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    embeddingFunctor
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(embedding PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()


add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    paddingoffset PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    cal_paddingoffset
)


add_executable(rms_norm
    test_rmsnorm.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    rms_norm PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    rmsnorm
)

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    causalmask PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    build_casual_mask
)



add_executable(testlinear
    test_linear.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    testlinear PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    linear
)


add_executable(biasRope
    test_bias_and_RoPE.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    biasRope PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope
)


add_executable(test_concat_kv
    test_concat_kv.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_concat_kv PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    concat_kv
)



add_executable(test_repeat_kv
    test_repeat_kv.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_repeat_kv PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    repeat_kv
)



add_executable(test_mask_softmax
    test_mask_softmax.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_mask_softmax PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    mask_softmax
)




add_executable(test_fused_trans_remv_pad
    test_fused_trans_remv_pad.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_fused_trans_remv_pad PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    fused_transpose_and_remv_pad
)




add_executable(test_fused_addresidual_norm
    test_fused_addresidual_norm.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_fused_addresidual_norm PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    fused_addresidual_norm
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_fused_addresidual_norm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>
    )
endif()



add_executable(test_act
    test_act.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_act PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    act
)                                                           # 是否使用debug模式进行编译，需要在cmake生成makefile的时候指定camke的生成模式，生成的是debug，然后再进行make的时候，才回使用debug模式进行编译
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(test_act PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()




add_executable(test_fused_decoder_attention
    test_fused_decoder_attention.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_fused_decoder_attention PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    fused_decoder_self_attention
)                                                           # 是否使用debug模式进行编译，需要在cmake生成makefile的时候指定camke的生成模式，生成的是debug，然后再进行make的时候，才回使用debug模式进行编译
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(test_act PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()





add_executable(test_topk
    test_topk.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_topk PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    topk
)                                                           # 是否使用debug模式进行编译，需要在cmake生成makefile的时候指定camke的生成模式，生成的是debug，然后再进行make的时候，才回使用debug模式进行编译
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(test_topk PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()



add_executable(test_sampling
    test_sampling.cu
)
target_link_libraries(  # 添加需要动态连接的库，PUBLIC表示没有相互依赖关系
    test_sampling PUBLIC    # 表示要链接到embedding上
    -lcudart
    -lcudadevrt
    sampling
)                                                           # 是否使用debug模式进行编译，需要在cmake生成makefile的时候指定camke的生成模式，生成的是debug，然后再进行make的时候，才回使用debug模式进行编译
if(CMAKE_BUILD_TYPE STREQUAL "Debug")                       # 如果是debug模型进行make
    target_compile_options(test_topk PRIVATE                 # 这里就是使用debug进行编译的附加参数
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>                   # -g表示再主机端的debug模式进行编译
        $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>                  # -G表示再设备端的debug模型编译 
    )                                                       # -O0表示禁止编译器进行代码优化
endif()