include_guard(GLOBAL)

function(omill_resolve_qt6)
  set(_components ${ARGV})
  if(_components STREQUAL "")
    message(FATAL_ERROR "omill_resolve_qt6 requires at least one Qt component")
  endif()

  find_package(Qt6 COMPONENTS ${_components} QUIET)
  if(Qt6_FOUND)
    message(STATUS "Found Qt6: ${Qt6_DIR}")
    return()
  endif()

  if(NOT OMILL_AUTO_FETCH_QT)
    message(FATAL_ERROR
      "Qt6 not found. Set Qt6_DIR/CMAKE_PREFIX_PATH or enable OMILL_AUTO_FETCH_QT.")
  endif()

  if(NOT WIN32)
    message(FATAL_ERROR
      "OMILL_AUTO_FETCH_QT currently supports Windows only.")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(_qt_root "${CMAKE_BINARY_DIR}/_deps/qt6")
  set(_qt_config "")

  file(GLOB_RECURSE _qt_configs
    "${_qt_root}/**/Qt6Config.cmake")
  if(_qt_configs)
    list(SORT _qt_configs)
    list(GET _qt_configs -1 _qt_config)
  endif()

  if(NOT EXISTS "${_qt_config}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m aqt version
      RESULT_VARIABLE _aqt_version_result
      OUTPUT_QUIET
      ERROR_QUIET
    )

    if(NOT _aqt_version_result EQUAL 0)
      message(STATUS "Installing Python package: aqtinstall")
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pip install --user --upgrade aqtinstall
        RESULT_VARIABLE _aqt_pip_result
        OUTPUT_VARIABLE _aqt_pip_out
        ERROR_VARIABLE _aqt_pip_err
      )
      if(NOT _aqt_pip_result EQUAL 0)
        message(FATAL_ERROR
          "Failed to install aqtinstall.\n${_aqt_pip_out}\n${_aqt_pip_err}")
      endif()
    endif()

    message(STATUS
      "Auto-fetching Qt ${OMILL_QT_VERSION} (${OMILL_QT_ARCH}) to ${_qt_root}")

    set(_aqt_cmd
      "${Python3_EXECUTABLE}" -m aqt install-qt
      windows desktop "${OMILL_QT_VERSION}" "${OMILL_QT_ARCH}"
      --outputdir "${_qt_root}"
      --timeout "${OMILL_QT_DOWNLOAD_TIMEOUT}"
    )

    if(OMILL_QT_MODULES)
      list(APPEND _aqt_cmd --modules)
      foreach(_module IN LISTS OMILL_QT_MODULES)
        if(NOT _module STREQUAL "")
          list(APPEND _aqt_cmd "${_module}")
        endif()
      endforeach()
    endif()

    execute_process(
      COMMAND ${_aqt_cmd}
      RESULT_VARIABLE _aqt_result
      OUTPUT_VARIABLE _aqt_out
      ERROR_VARIABLE _aqt_err
    )
    if(NOT _aqt_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to auto-fetch Qt6 via aqt.\n${_aqt_out}\n${_aqt_err}")
    endif()
  endif()

  if(NOT EXISTS "${_qt_config}")
    file(GLOB_RECURSE _qt_configs
      "${_qt_root}/**/Qt6Config.cmake")
    if(_qt_configs)
      list(SORT _qt_configs)
      list(GET _qt_configs -1 _qt_config)
    endif()
  endif()

  if(NOT EXISTS "${_qt_config}")
    message(FATAL_ERROR
      "Qt auto-fetch completed but Qt6Config.cmake is missing at ${_qt_config}")
  endif()

  get_filename_component(_qt_dir "${_qt_config}" DIRECTORY)

  set(Qt6_DIR "${_qt_dir}" CACHE PATH
    "Path to Qt6Config.cmake" FORCE)

  find_package(Qt6 COMPONENTS ${_components} REQUIRED)
  message(STATUS "Using auto-fetched Qt6 from ${Qt6_DIR}")
endfunction()
