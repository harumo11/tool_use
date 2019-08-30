cmake_minimum_required(VERSION 3.10)

add_executable(learning learning.cpp)

target_link_libraries(learning dynet fanda gpop)
