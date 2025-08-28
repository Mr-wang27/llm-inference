# 该目录下还有其他的目录，源文件在这些目录下
# 所以这个CMakeList.txt只需要继续构建子目录即可

add_subdirectory(weights)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(utils)
add_subdirectory(models)

