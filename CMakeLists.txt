cmake_minimum_required(VERSION 2.8.11)
project(MLP)

add_custom_target(doc_files SOURCES README.md)

#############################################################################
# EXTERNALS
#############################################################################
set(EXTERNALS_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext)
file(MAKE_DIRECTORY ${EXTERNALS_DIR})   # ensure "ext" directory exists

add_custom_target(update)

function(update_repo repo dir)
    if(NOT EXISTS ${EXTERNALS_DIR}/${dir})  # if exists then "git pull" else "git clone"
        execute_process(
            COMMAND             git clone ${repo} ${dir}
            WORKING_DIRECTORY   ${EXTERNALS_DIR})
    else()
        add_custom_command(TARGET update
            COMMAND             git pull
            WORKING_DIRECTORY   ${EXTERNALS_DIR}/${dir}
            COMMENT "Updating ${dir} repo")
    endif()
endfunction()

# Eigen
update_repo(git@github.com:eigenteam/eigen-git-mirror.git   eigen)

#############################################################################
# DEMO
#############################################################################
include_directories(${EXTERNALS_DIR}/eigen)

add_compile_options(-std=c++11 -pedantic)

add_executable(demo demo.cpp MLP.h)
