# 该目录下还有其他的目录，源文件在这些目录下
# 所以这个CMakeList.txt只需要继续构建子目录即可
add_subdirectory(ffn)
add_subdirectory(attention)
add_subdirectory(decoder)
