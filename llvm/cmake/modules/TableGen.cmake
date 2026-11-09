# LLVM_TARGET_DEFINITIONS must contain the name of the .td file to process,
# while LLVM_TARGET_DEPENDS may contain additional file dependencies.
# Extra parameters for `tblgen' may come after `ofn' parameter.
# Adds the name of the generated file to TABLEGEN_OUTPUT.
include(LLVMDistributionSupport)

# Capture helper paths before entering caller-owned functions.
set(_LLVM_TABLEGEN_DEPFILE_REWRITER
  "${CMAKE_CURRENT_LIST_DIR}/RewriteTableGenDepfile.cmake")
set(_LLVM_TABLEGEN_OUTPUT_RECOVERY
  "${CMAKE_CURRENT_LIST_DIR}/RecoverTableGenOutput.cmake")

# Record successful generation separately from the artifact's content mtime.
# Make does not restore missing BYPRODUCTS, so it additionally needs a recovery
# rule per artifact. A custom-target dependency orders those rules after the
# stamp without making an unchanged artifact perpetually older than the stamp.
function(_llvm_tablegen_command ofn)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "DEPFILE;PRIMARY_INPUT" "COMMAND;DEPENDS;EXTRA_OUTPUTS")
  # Several upstream callers spell option values as -key="Value". Before
  # VERBATIM these quotes belonged to the shell, not the tool argument. Keep
  # that established argv while quoting actual path arguments correctly.
  set(_command)
  foreach(_argument IN LISTS ARG_COMMAND)
    if(_argument MATCHES "^(-[^=]+)=\"([^\"]*)\"$")
      set(_argument "${CMAKE_MATCH_1}=${CMAKE_MATCH_2}")
    endif()
    list(APPEND _command "${_argument}")
  endforeach()
  set(ARG_COMMAND ${_command})
  # The recovery script receives the command as arguments, where CMake cannot
  # perform its usual executable-target substitution. Normalize it once for
  # both primary generation and recovery, preserving cross-tool emulators.
  list(GET ARG_COMMAND 0 _executable)
  if(TARGET "${_executable}")
    get_target_property(_kind "${_executable}" TYPE)
    if(_kind STREQUAL "EXECUTABLE")
      list(REMOVE_AT ARG_COMMAND 0)
      list(PREPEND ARG_COMMAND "$<TARGET_FILE:${_executable}>")
      if(CMAKE_CROSSCOMPILING)
        get_target_property(_emulator "${_executable}" CROSSCOMPILING_EMULATOR)
        if(_emulator)
          list(PREPEND ARG_COMMAND ${_emulator})
        endif()
      endif()
    endif()
  endif()
  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/${ofn}.stamp")
  set(_outputs "${CMAKE_CURRENT_BINARY_DIR}/${ofn}" ${ARG_EXTRA_OUTPUTS})
  # Build-tree consumers need the complete artifact inventory before the first
  # native build. This is output metadata, not a second scheduling authority.
  set_property(GLOBAL APPEND PROPERTY LLVM_TABLEGEN_GENERATED_OUTPUTS ${_outputs})
  set(_byproducts)
  if(NOT CMAKE_GENERATOR MATCHES "Makefiles")
    set(_byproducts BYPRODUCTS ${_outputs})
  endif()
  set(_depfile_arguments)
  set(_rewrite)
  if(ARG_DEPFILE)
    set(_depfile_arguments DEPFILE "${ARG_DEPFILE}")
    set(_rewrite COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${ARG_DEPFILE}" "-DOUTPUT=${_stamp}"
      "-DPRIMARY_INPUT=${ARG_PRIMARY_INPUT}"
      -P "${_LLVM_TABLEGEN_DEPFILE_REWRITER}")
  endif()
  add_custom_command(OUTPUT "${_stamp}"
    ${_byproducts}
    COMMAND ${ARG_COMMAND}
    ${_rewrite}
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
    DEPENDS ${ARG_DEPENDS}
      "${_LLVM_TABLEGEN_DEPFILE_REWRITER}"
      "${_LLVM_TABLEGEN_OUTPUT_RECOVERY}"
    ${_depfile_arguments}
    ${LLVM_TABLEGEN_JOB_POOL}
    COMMENT "Building ${ofn}..."
    VERBATIM)
  if(CMAKE_GENERATOR MATCHES "Makefiles")
    string(SHA256 _identity "${_stamp}")
    set(_owner "tablegen-stamp-${_identity}")
    add_custom_target(${_owner} DEPENDS "${_stamp}")
    set_property(GLOBAL APPEND PROPERTY LLVM_TABLEGEN_GENERATION_TARGETS ${_owner})
    foreach(_output IN LISTS _outputs)
      add_custom_command(OUTPUT "${_output}"
        COMMAND "${CMAKE_COMMAND}"
          "-DREQUIRED_OUTPUT=${_output}" "-DSTAMP=${_stamp}"
          "-DDEPFILE=${ARG_DEPFILE}"
          "-DPRIMARY_INPUT=${ARG_PRIMARY_INPUT}"
          "-DDEPFILE_REWRITER=${_LLVM_TABLEGEN_DEPFILE_REWRITER}"
          -P "${_LLVM_TABLEGEN_OUTPUT_RECOVERY}" -- ${ARG_COMMAND}
        DEPENDS ${_owner}
        COMMENT "Restoring ${ofn}..."
        VERBATIM)
    endforeach()
  endif()
  set_source_files_properties(${_outputs} PROPERTIES
    GENERATED 1 LLVM_TABLEGEN_STAMP "${_stamp}"
    LLVM_TABLEGEN_OUTPUTS "${_outputs}")
endfunction()

# Preserve TABLEGEN_OUTPUT as the single caller-owned output list. In
# particular its existing reset behavior must also reset generation ordering;
# non-TableGen outputs (e.g. mixed PDLL libraries) remain real dependencies.
function(_llvm_tablegen_dependencies result)
  set(_dependencies)
  foreach(_output IN LISTS ARGN)
    get_source_file_property(_stamp "${_output}" LLVM_TABLEGEN_STAMP)
    if(_stamp)
      list(APPEND _dependencies "${_stamp}")
    endif()
    get_source_file_property(_outputs "${_output}" LLVM_TABLEGEN_OUTPUTS)
    if(_outputs)
      list(APPEND _dependencies ${_outputs})
    else()
      list(APPEND _dependencies "${_output}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _dependencies)
  set(${result} "${_dependencies}" PARENT_SCOPE)
endfunction()

function(tablegen project ofn)
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})

  # Override ${project} with ${project}_TABLEGEN_PROJECT
  if(NOT "${${project}_TABLEGEN_PROJECT}" STREQUAL "")
    set(project ${${project}_TABLEGEN_PROJECT})
  endif()

  # Validate calling context.
  if(NOT ${project}_TABLEGEN_EXE)
    message(FATAL_ERROR "${project}_TABLEGEN_EXE not set")
  endif()

  # Set the include directories
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(PREPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  list(PREPEND tblgen_includes ${CMAKE_CURRENT_SOURCE_DIR})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Use depfiles instead of globbing arbitrary *.td(s) for Ninja and Makefiles.
  # Makefile generators support DEPFILE since CMake 3.20. We force
  # CMake versions older than v3.30 on Windows to use the fallback behavior
  # due to a depfile parsing bug on Windows paths in versions prior to 3.30.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/25943
  # CMake versions older than v3.23 on other platforms use the fallback
  # behavior as v3.22 and earlier fail to parse some depfiles that get
  # generated, and this behavior was fixed in CMake commit
  # e04a352cca523eba2ac0d60063a3799f5bb1c69e.
  cmake_policy(GET CMP0116 cmp0116_state)
  if(CMAKE_GENERATOR MATCHES "Ninja|Makefiles" AND cmp0116_state STREQUAL NEW
     AND NOT (CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_LESS 3.30)
     AND NOT (CMAKE_VERSION VERSION_LESS 3.23))
    # CMake emits build targets as relative paths but Ninja doesn't identify
    # absolute path (in *.d) as relative path (in build.ninja). Post CMP0116,
    # CMake handles this discrepancy for us, otherwise we use the fallback
    # logic.
    set(additional_cmdline
      -o ${ofn}
      -d ${ofn}.d
      )
    set(global_tds)
    set(_tablegen_depfile_enabled ON)
  else()
    set(include_td_dirs "${tblgen_includes}")
    list(TRANSFORM include_td_dirs APPEND "/*.td")
    # Includes may be nested arbitrarily below an include root. Reconfigure
    # on additions as well, so a newly included file remains a dependency on
    # subsequent builds. This conservative path is only for hosts/generators
    # that cannot consume the precise depfile above.
    file(GLOB_RECURSE global_tds CONFIGURE_DEPENDS ${include_td_dirs})
    set(additional_cmdline
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      )
    set(_tablegen_depfile_enabled OFF)
  endif()

  if (LLVM_ENABLE_DAGISEL_COV AND "-gen-dag-isel" IN_LIST ARGN)
    list(APPEND LLVM_TABLEGEN_FLAGS "-instrument-coverage")
  endif()
  if (LLVM_ENABLE_GISEL_COV AND "-gen-global-isel" IN_LIST ARGN)
    list(APPEND LLVM_TABLEGEN_FLAGS "-instrument-gisel-coverage")
    list(APPEND LLVM_TABLEGEN_FLAGS "-gisel-coverage-file=${LLVM_GISEL_COV_PREFIX}all")
  endif()
  if (LLVM_OMIT_DAGISEL_COMMENTS AND "-gen-dag-isel" IN_LIST ARGN)
    list(APPEND LLVM_TABLEGEN_FLAGS "-omit-comments")
  endif()

  set(EXTRA_OUTPUTS)
  if("-gen-register-info" IN_LIST ARGN)
    cmake_path(GET ofn STEM OUTPUT_BASENAME)
    list(APPEND EXTRA_OUTPUTS
         ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BASENAME}Enums.inc
         ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BASENAME}Header.inc
         ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BASENAME}MCDesc.inc
         ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BASENAME}TargetDesc.inc)
  endif()

  # MSVC can't support long string literals ("long" > 65534 bytes)[1], so if there's
  # a possibility of generated tables being consumed by MSVC, generate arrays of
  # char literals, instead. If we're cross-compiling, then conservatively assume
  # that the source might be consumed by MSVC.
  # [1] https://docs.microsoft.com/en-us/cpp/cpp/compiler-limits?view=vs-2017
  # Don't pass this flag to mlir-src-sharder, since it doesn't support the
  # flag, and it doesn't need it.
  if (MSVC AND NOT "${project}" STREQUAL "MLIR_SRC_SHARDER")
    list(APPEND LLVM_TABLEGEN_FLAGS "--long-string-literals=0")
  endif()
  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio has problems with llvm-tblgen's native --write-if-changed
    # behavior. Since it doesn't do restat optimizations anyway, just don't
    # pass --write-if-changed there.
    set(tblgen_change_flag)
  else()
    set(tblgen_change_flag "--write-if-changed")
  endif()

  # Don't pass this flag to mlir-src-sharder, since it doesn't support the
  # flag, and it doesn't need it.
  if (NOT LLVM_ENABLE_WARNINGS AND NOT "${project}" STREQUAL "MLIR_SRC_SHARDER")
    list(APPEND LLVM_TABLEGEN_FLAGS "-no-warn-on-unused-template-args")
  endif()

  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append this file and its includes to the compile commands file.
  # This file is used by the TableGen LSP Language Server (tblgen-lsp-server).
  file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
      "--- !FileInfo:\n"
      "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
      "  includes: \"${tblgen_includes}\"\n"
  )

  # Prepend each include entry with -I for arguments.
  list(TRANSFORM tblgen_includes PREPEND -I)

  # We need both _TABLEGEN_TARGET and _TABLEGEN_EXE in the  DEPENDS list
  # (both the target and the file) to have .inc files rebuilt on
  # a tablegen change, as cmake does not propagate file-level dependencies
  # of custom targets. See the following ticket for more information:
  # https://cmake.org/Bug/view.php?id=15858
  # The dependency on both, the target and the file, produces the same
  # dependency twice in the result file when
  # ("${${project}_TABLEGEN_TARGET}" STREQUAL "${${project}_TABLEGEN_EXE}")
  # but lets us having smaller and cleaner code here.
  set(tablegen_exe ${${project}_TABLEGEN_EXE})
  set(tablegen_depends ${${project}_TABLEGEN_TARGET} ${tablegen_exe})

  if(LLVM_PARALLEL_TABLEGEN_JOBS)
    set(LLVM_TABLEGEN_JOB_POOL JOB_POOL tablegen_job_pool)
  else()
    set(LLVM_TABLEGEN_JOB_POOL "")
  endif()

  set(_depfile)
  if(_tablegen_depfile_enabled)
    set(_depfile "${CMAKE_CURRENT_BINARY_DIR}/${ofn}.d")
  endif()
  _llvm_tablegen_command(${ofn}
    COMMAND ${tablegen_exe} ${ARG_UNPARSED_ARGUMENTS}
      ${tblgen_includes} ${LLVM_TABLEGEN_FLAGS}
      ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
      ${tblgen_change_flag} ${additional_cmdline}
    DEPFILE "${_depfile}"
    EXTRA_OUTPUTS ${EXTRA_OUTPUTS}
    DEPENDS ${ARG_DEPENDS} ${tablegen_depends} ${global_tds}
      ${LLVM_TARGET_DEFINITIONS_ABSOLUTE} ${LLVM_TARGET_DEPENDS})

  # `make clean' must remove all those generated files:
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})

  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES
    GENERATED 1)
endfunction()

# Creates a target for publicly exporting tablegen dependencies.
function(add_public_tablegen_target target)
  if(NOT TABLEGEN_OUTPUT)
    message(FATAL_ERROR "Requires tablegen() definitions as TABLEGEN_OUTPUT.")
  endif()
  _llvm_tablegen_dependencies(_dependencies ${TABLEGEN_OUTPUT})
  add_custom_target(${target} DEPENDS ${_dependencies})
  set(_artifacts)
  foreach(_output IN LISTS TABLEGEN_OUTPUT)
    get_source_file_property(_outputs "${_output}" LLVM_TABLEGEN_OUTPUTS)
    if(_outputs)
      list(APPEND _artifacts ${_outputs})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _artifacts)
  set_property(TARGET ${target} PROPERTY LLVM_TABLEGEN_ARTIFACTS "${_artifacts}")
  set_property(GLOBAL APPEND PROPERTY LLVM_TABLEGEN_PUBLIC_TARGETS ${target})
  if(LLVM_COMMON_DEPENDS)
    add_dependencies(${target} ${LLVM_COMMON_DEPENDS})
  endif()
  get_subproject_title(subproject_title)
  set_target_properties(${target} PROPERTIES FOLDER "${subproject_title}/Tablegenning")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${target} PARENT_SCOPE)
endfunction()

macro(add_tablegen target project)
  cmake_parse_arguments(ADD_TABLEGEN "" "DESTINATION;EXPORT" "" ${ARGN})

  set(${target}_OLD_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
  set(LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS} TableGen)

  add_llvm_executable(${target} DISABLE_LLVM_LINK_LLVM_DYLIB
    ${ADD_TABLEGEN_UNPARSED_ARGUMENTS})
  set(LLVM_LINK_COMPONENTS ${${target}_OLD_LLVM_LINK_COMPONENTS})

  set(${project}_TABLEGEN_DEFAULT "${target}")
  if (LLVM_NATIVE_TOOL_DIR)
    if (EXISTS "${LLVM_NATIVE_TOOL_DIR}/${target}${LLVM_HOST_EXECUTABLE_SUFFIX}")
      set(${project}_TABLEGEN_DEFAULT "${LLVM_NATIVE_TOOL_DIR}/${target}${LLVM_HOST_EXECUTABLE_SUFFIX}")
    endif()
  endif()

  # FIXME: Quick fix to reflect LLVM_TABLEGEN to llvm-min-tblgen
  if("${target}" STREQUAL "llvm-min-tblgen"
      AND NOT "${LLVM_TABLEGEN}" STREQUAL ""
      AND NOT "${LLVM_TABLEGEN}" STREQUAL "llvm-tblgen")
    set(${project}_TABLEGEN_DEFAULT "${LLVM_TABLEGEN}")
  endif()

  if(ADD_TABLEGEN_EXPORT)
    set(${project}_TABLEGEN "${${project}_TABLEGEN_DEFAULT}" CACHE
      STRING "Native TableGen executable. Saves building one when cross-compiling.")
  else()
    # Internal tablegen
    set(${project}_TABLEGEN "${${project}_TABLEGEN_DEFAULT}")
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()

  # Effective tblgen executable to be used:
  set(${project}_TABLEGEN_EXE ${${project}_TABLEGEN} PARENT_SCOPE)
  set(${project}_TABLEGEN_TARGET ${${project}_TABLEGEN} PARENT_SCOPE)

  if(LLVM_USE_HOST_TOOLS)
    if( ${${project}_TABLEGEN} STREQUAL "${target}" )
      # The NATIVE tablegen executable *must* depend on the current target one
      # otherwise the native one won't get rebuilt when the tablgen sources
      # change, and we end up with incorrect builds.
      build_native_tool(${target} ${project}_TABLEGEN_EXE DEPENDS ${target})
      set(${project}_TABLEGEN_EXE ${${project}_TABLEGEN_EXE} PARENT_SCOPE)

      add_custom_target(${target}-host DEPENDS ${${project}_TABLEGEN_EXE})
      get_subproject_title(subproject_title)
      set_target_properties(${target}-host PROPERTIES FOLDER "${subproject_title}/Native")
      set(${project}_TABLEGEN_TARGET ${target}-host PARENT_SCOPE)

      # If we're using the host tablegen, and utils were not requested, we have no
      # need to build this tablegen.
      if ( NOT LLVM_BUILD_UTILS )
        set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL ON)
      endif()
    endif()
  endif()

  if (ADD_TABLEGEN_DESTINATION AND NOT LLVM_INSTALL_TOOLCHAIN_ONLY AND
      (LLVM_BUILD_UTILS OR ${target} IN_LIST LLVM_DISTRIBUTION_COMPONENTS))
    set(export_arg)
    if(ADD_TABLEGEN_EXPORT)
      get_target_export_arg(${target} ${ADD_TABLEGEN_EXPORT} export_arg)
    endif()
    install(TARGETS ${target}
            ${export_arg}
            COMPONENT ${target}
            RUNTIME DESTINATION "${ADD_TABLEGEN_DESTINATION}")
    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets("install-${target}"
                               DEPENDS ${target}
                               COMPONENT ${target})
    endif()
  endif()
  if(ADD_TABLEGEN_EXPORT)
    string(TOUPPER ${ADD_TABLEGEN_EXPORT} export_upper)
    set_property(GLOBAL APPEND PROPERTY ${export_upper}_EXPORTS ${target})
  endif()
endmacro()

# Make sure 'tablegen_compile_commands.yml' is only deleted once the very
# first time this file is included.
include_guard(GLOBAL)

# Clear out any pre-existing compile_commands file before processing. This
# allows for generating a clean compile_commands on each configure.
file(REMOVE ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml)
