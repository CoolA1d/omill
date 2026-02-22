# Obfuscated test DLL — compiled with O2 optimizations using xorstr + lazy_importer.
# Gated behind OMILL_ENABLE_REMILL (same as e2e tests).

add_library(obf_test SHARED
  src/dllmain.cpp
  src/xorstr_funcs.cpp
  src/lazy_funcs.cpp
  src/combined_funcs.cpp
  src/mixed_funcs.cpp
  src/cloakwork_str_funcs.cpp
  src/cloakwork_import_funcs.cpp
  src/cloakwork_combined_funcs.cpp
  src/cmut_funcs.cpp
)

# Include paths for obfuscation headers.
target_include_directories(obf_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/xorstr-master/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lazy_importer-master/include
  ${CMAKE_CURRENT_SOURCE_DIR}/cloakwork
  ${CMAKE_CURRENT_SOURCE_DIR}/cmut
)

# Per-file compile options: dllmain at -O0, everything else at -O2.
set_source_files_properties(src/dllmain.cpp PROPERTIES COMPILE_OPTIONS "-O0")
set_source_files_properties(
  src/xorstr_funcs.cpp
  src/lazy_funcs.cpp
  src/combined_funcs.cpp
  src/mixed_funcs.cpp
  PROPERTIES COMPILE_OPTIONS "-O2"
)

# Cloakwork requires C++20 (constinit, consteval) and -O2 for optimization.
set_source_files_properties(
  src/cloakwork_str_funcs.cpp
  src/cloakwork_import_funcs.cpp
  src/cloakwork_combined_funcs.cpp
  PROPERTIES COMPILE_OPTIONS "-O2;-std=c++20"
)

# cmut requires C++20 (<bit> for std::rotl/std::rotr) and -O2 for inlining.
set_source_files_properties(
  src/cmut_funcs.cpp
  PROPERTIES COMPILE_OPTIONS "-O2;-std=c++20"
)

# Disable AVX for xorstr to keep SIMD at SSE2 level.
target_compile_definitions(obf_test PRIVATE JM_XORSTR_DISABLE_AVX_INTRINSICS)

# Undo the global UNICODE definitions that LLVM adds — they break narrow-char
# Windows API forward declarations in the test sources.
target_compile_options(obf_test PRIVATE -UUNICODE -U_UNICODE)

# Link kernel32.lib for DllMain linkage.
target_link_libraries(obf_test PRIVATE kernel32)

# Output name without lib prefix.
set_target_properties(obf_test PROPERTIES
  PREFIX ""
  OUTPUT_NAME "obf_test"
)

# --------------------------------------------------------------------------
# OLLVM-obfuscated test DLL — C++ → bitcode → obfuscate → object → DLL.
# --------------------------------------------------------------------------

# Step 1: C++ → bitcode (at -O2 to get clean IR before obfuscation).
option(OMILL_OLLVM_OBF_FLATTEN "Enable OLLVM CFF flattening for test samples" ON)
option(OMILL_OLLVM_OBF_SUBSTITUTE "Enable OLLVM MBA substitution for test samples" ON)
option(OMILL_OLLVM_OBF_STRING_ENCRYPT "Enable OLLVM string encryption for test samples" ON)
option(OMILL_OLLVM_OBF_CONST_UNFOLD "Enable OLLVM const unfold for test samples" ON)
option(OMILL_OLLVM_OBF_VECTORIZE "Enable OLLVM scalar-to-vector transform for test samples" ON)
option(OMILL_OLLVM_OBF_VECTORIZE_DATA "Enable OLLVM vectorized stack data mutation for test samples" ON)
option(OMILL_OLLVM_OBF_VECTORIZE_BITWISE "Enable OLLVM vectorized bitwise add/sub mutation for test samples" OFF)
set(OMILL_OLLVM_OBF_SEED "2976579765" CACHE STRING "Base RNG seed for ollvm-obf test artifact generation")

set(_ollvm_obf_flags "")
list(APPEND _ollvm_obf_flags --seed=${OMILL_OLLVM_OBF_SEED})
if(OMILL_OLLVM_OBF_FLATTEN)
  list(APPEND _ollvm_obf_flags --flatten)
endif()
if(OMILL_OLLVM_OBF_SUBSTITUTE)
  list(APPEND _ollvm_obf_flags --substitute)
endif()
if(OMILL_OLLVM_OBF_STRING_ENCRYPT)
  list(APPEND _ollvm_obf_flags --string-encrypt)
endif()
if(OMILL_OLLVM_OBF_CONST_UNFOLD)
  list(APPEND _ollvm_obf_flags --const-unfold)
endif()
if(OMILL_OLLVM_OBF_VECTORIZE)
  list(APPEND _ollvm_obf_flags --vectorize)
  if(OMILL_OLLVM_OBF_VECTORIZE_DATA)
    list(APPEND _ollvm_obf_flags --vectorize-data)
  else()
    list(APPEND _ollvm_obf_flags --vectorize-data=false)
  endif()
  if(OMILL_OLLVM_OBF_VECTORIZE_BITWISE)
    list(APPEND _ollvm_obf_flags --vectorize-bitwise)
  endif()
endif()

# --------------------------------------------------------------------------
# Add OLLVM-obfuscated algorithm exports to obf_test.dll.
# --------------------------------------------------------------------------

set(_obf_algo_src "${CMAKE_CURRENT_SOURCE_DIR}/src/obf_algo_funcs.cpp")
set(_obf_algo_clean_bc "${CMAKE_CURRENT_BINARY_DIR}/obf_algo_funcs.bc")
set(_obf_algo_obf_bc "${CMAKE_CURRENT_BINARY_DIR}/obf_algo_funcs_obf.bc")
set(_obf_algo_obf_obj "${CMAKE_CURRENT_BINARY_DIR}/obf_algo_funcs_obf.obj")

add_custom_command(
  OUTPUT ${_obf_algo_clean_bc}
  COMMAND ${CMAKE_CXX_COMPILER} -c -emit-llvm -O2
    -target x86_64-pc-windows-msvc
    ${_obf_algo_src}
    -o ${_obf_algo_clean_bc}
  DEPENDS ${_obf_algo_src}
  COMMENT "Compiling obf_algo_funcs.cpp to bitcode"
)

add_custom_command(
  OUTPUT ${_obf_algo_obf_bc}
  COMMAND $<TARGET_FILE:ollvm-obf>
    ${_obf_algo_clean_bc}
    -o ${_obf_algo_obf_bc}
    ${_ollvm_obf_flags}
  DEPENDS ${_obf_algo_clean_bc} ollvm-obf
  COMMENT "Obfuscating obf_algo_funcs.bc with selected OLLVM transforms: ${_ollvm_obf_flags}"
)

add_custom_command(
  OUTPUT ${_obf_algo_obf_obj}
  COMMAND ${CMAKE_CXX_COMPILER} -c -O0
    -target x86_64-pc-windows-msvc
    ${_obf_algo_obf_bc}
    -o ${_obf_algo_obf_obj}
  DEPENDS ${_obf_algo_obf_bc}
  COMMENT "Compiling obfuscated algorithm bitcode to object"
)

set_source_files_properties(
  ${_obf_algo_obf_obj}
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
)
target_sources(obf_test PRIVATE ${_obf_algo_obf_obj})
add_dependencies(obf_test ollvm-obf)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs.bc
  COMMAND ${CMAKE_CXX_COMPILER} -c -emit-llvm -O2
    -target x86_64-pc-windows-msvc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ollvm_funcs.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs.bc
  DEPENDS src/ollvm_funcs.cpp
  COMMENT "Compiling ollvm_funcs.cpp to bitcode"
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix.bc
  COMMAND ${CMAKE_CXX_COMPILER} -c -emit-llvm -O2
    -target x86_64-pc-windows-msvc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ollvm_vec_const_mix.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix.bc
  DEPENDS src/ollvm_vec_const_mix.cpp
  COMMENT "Compiling ollvm_vec_const_mix.cpp to bitcode"
)

# Step 2: obfuscate bitcode with ollvm-obf.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.bc
  COMMAND $<TARGET_FILE:ollvm-obf>
    ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs.bc
    -o ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.bc
    ${_ollvm_obf_flags}
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs.bc ollvm-obf
  COMMENT "Obfuscating ollvm_funcs.bc with selected OLLVM transforms: ${_ollvm_obf_flags}"
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.bc
  COMMAND $<TARGET_FILE:ollvm-obf>
    ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix.bc
    -o ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.bc
    --seed=${OMILL_OLLVM_OBF_SEED}
    --const-unfold
    --vectorize
    --vectorize-percent=100
    --vectorize-bitwise
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix.bc ollvm-obf
  COMMENT "Obfuscating ollvm_vec_const_mix.bc with const-unfold + vectorize only"
)

# Step 3: obfuscated bitcode → object (O0 to prevent LLVM undoing obfuscation).
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.obj
  COMMAND ${CMAKE_CXX_COMPILER} -c -O0
    -target x86_64-pc-windows-msvc
    -msse4.2
    -mno-avx
    ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.bc
    -o ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.obj
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.bc
  COMMENT "Compiling obfuscated bitcode to object"
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.obj
  COMMAND ${CMAKE_CXX_COMPILER} -c -O0
    -target x86_64-pc-windows-msvc
    -msse4.2
    -mno-avx
    ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.bc
    -o ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.obj
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.bc
  COMMENT "Compiling obfuscated vector/const sample to object"
)

# Step 4: Link DLL from obfuscated object + dllmain.
add_library(ollvm_test SHARED
  src/dllmain.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.obj
  ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.obj
)
set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/ollvm_funcs_obf.obj
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
)
set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/ollvm_vec_const_mix_obf.obj
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
)
set_source_files_properties(src/dllmain.cpp
  PROPERTIES COMPILE_OPTIONS "-O0"
)
target_compile_options(ollvm_test PRIVATE -UUNICODE -U_UNICODE)
target_link_libraries(ollvm_test PRIVATE kernel32)
set_target_properties(ollvm_test PROPERTIES PREFIX "" OUTPUT_NAME "ollvm_test")
add_dependencies(ollvm_test ollvm-obf)

# ==========================================================================
# Deobfuscation round-trip test DLLs — one DLL per OLLVM pass (+ combined).
# Pipeline: C++ -> bitcode (-O2) -> ollvm-obf [per-pass] -> obj (-O0) -> DLL
# ==========================================================================

set(_rt_src "${CMAKE_CURRENT_SOURCE_DIR}/src/deobf_roundtrip_funcs.cpp")
set(_rt_seed ${OMILL_OLLVM_OBF_SEED})

# Step 1: C++ -> clean bitcode (base for all variants).
set(_rt_base_bc "${CMAKE_CURRENT_BINARY_DIR}/rt_funcs_base.bc")
add_custom_command(
  OUTPUT ${_rt_base_bc}
  COMMAND ${CMAKE_CXX_COMPILER} -c -emit-llvm -O2
    -target x86_64-pc-windows-msvc
    ${_rt_src}
    -o ${_rt_base_bc}
  DEPENDS ${_rt_src}
  COMMENT "[RT] Compiling deobf_roundtrip_funcs.cpp to clean bitcode"
)

# Helper macro: build one DLL from a single set of ollvm-obf flags.
# Usage: rt_add_obf_dll(<name> <flag1> [flag2 ...])
# - For the clean variant, pass CLEAN as the sole flag (no obfuscation).
macro(rt_add_obf_dll NAME)
  set(_rt_${NAME}_bc "${CMAKE_CURRENT_BINARY_DIR}/rt_funcs_${NAME}.bc")
  set(_rt_${NAME}_obj "${CMAKE_CURRENT_BINARY_DIR}/rt_funcs_${NAME}.obj")

  if("${ARGN}" STREQUAL "CLEAN")
    # No obfuscation — copy clean bitcode directly.
    add_custom_command(
      OUTPUT ${_rt_${NAME}_bc}
      COMMAND ${CMAKE_COMMAND} -E copy ${_rt_base_bc} ${_rt_${NAME}_bc}
      DEPENDS ${_rt_base_bc}
      COMMENT "[RT] Copying clean bitcode for rt_${NAME}"
    )
  else()
    add_custom_command(
      OUTPUT ${_rt_${NAME}_bc}
      COMMAND $<TARGET_FILE:ollvm-obf>
        ${_rt_base_bc}
        -o ${_rt_${NAME}_bc}
        --seed=${_rt_seed}
        ${ARGN}
      DEPENDS ${_rt_base_bc} ollvm-obf
      COMMENT "[RT] Obfuscating rt_funcs for rt_${NAME}: ${ARGN}"
    )
  endif()

  add_custom_command(
    OUTPUT ${_rt_${NAME}_obj}
    COMMAND ${CMAKE_CXX_COMPILER} -c -O0
      -target x86_64-pc-windows-msvc
      -msse4.2 -mno-avx
      ${_rt_${NAME}_bc}
      -o ${_rt_${NAME}_obj}
    DEPENDS ${_rt_${NAME}_bc}
    COMMENT "[RT] Compiling rt_${NAME} bitcode to object"
  )

  set_source_files_properties(
    ${_rt_${NAME}_obj}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
  )

  add_library(rt_${NAME} SHARED
    src/dllmain.cpp
    ${_rt_${NAME}_obj}
  )
  set_source_files_properties(src/dllmain.cpp
    PROPERTIES COMPILE_OPTIONS "-O0"
  )
  target_compile_options(rt_${NAME} PRIVATE -UUNICODE -U_UNICODE)
  target_link_libraries(rt_${NAME} PRIVATE kernel32)
  set_target_properties(rt_${NAME} PROPERTIES
    PREFIX "" OUTPUT_NAME "rt_${NAME}"
  )
  add_dependencies(rt_${NAME} ollvm-obf)
endmacro()

# --- Per-pass variants ---
rt_add_obf_dll(clean           CLEAN)
rt_add_obf_dll(flatten         --flatten)
rt_add_obf_dll(substitute      --substitute)
rt_add_obf_dll(bogus_cf        --bogus-control-flow)
rt_add_obf_dll(opaque_pred     --opaque-predicates)
rt_add_obf_dll(const_unfold    --const-unfold)
rt_add_obf_dll(string_encrypt  --string-encrypt)
rt_add_obf_dll(vectorize       --vectorize --vectorize-data)
rt_add_obf_dll(all             ${_ollvm_obf_flags})