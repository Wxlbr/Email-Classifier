cmake_minimum_required(VERSION 3.20)
project(email-classifier)

include_directories(inc)

file(GLOB_RECURSE SOURCE "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCE})