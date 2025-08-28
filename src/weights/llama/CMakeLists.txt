add_library(layerweights STATIC layer_weights.cc)
target_link_libraries(layerweights PUBLIC -lcudart weightutils)# weightutils是在utils目录下编译的静态库


add_library(llamaweights STATIC llama_weights.cc)
target_link_libraries(llamaweights PUBLIC -lcudart layerweights)# weightutils是在utils目录下编译的静态库
