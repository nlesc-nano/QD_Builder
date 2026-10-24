cmake_minimum_required(VERSION 3.18...3.31)
project(qdb_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/core.cpp)

# No fused multiply-add: keeps the arithmetic close to NumPy's. Exactness
# does not rely on it (values near a threshold are re-checked in Python).
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(_core PRIVATE -ffp-contract=off)
elseif (MSVC)
  target_compile_options(_core PRIVATE /fp:precise)
endif()

install(TARGETS _core LIBRARY DESTINATION qdb_native)
