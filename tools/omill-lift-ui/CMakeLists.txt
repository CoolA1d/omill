if(NOT WIN32)
  message(FATAL_ERROR "omill-lift-ui currently supports Windows x64 only")
endif()

include("${PROJECT_SOURCE_DIR}/cmake/Qt6AutoFetch.cmake")
omill_resolve_qt6(Core Gui Widgets)

add_executable(omill-lift-ui
  main.cpp
)

target_link_libraries(omill-lift-ui
  PRIVATE
    omill
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

target_compile_features(omill-lift-ui PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(omill-lift-ui PRIVATE /W3 /EHsc)
else()
  target_compile_options(omill-lift-ui PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Make build output runnable by staging Qt runtime DLLs/plugins next to exe.
get_target_property(_qt_qmake_executable Qt6::qmake IMPORTED_LOCATION)
if(NOT _qt_qmake_executable)
  get_target_property(_qt_qmake_executable Qt6::qmake IMPORTED_LOCATION_RELEASE)
endif()
if(NOT _qt_qmake_executable)
  message(FATAL_ERROR "Unable to resolve Qt6::qmake location for deployment.")
endif()

get_filename_component(_qt_bin_dir "${_qt_qmake_executable}" DIRECTORY)
find_program(OMILL_WINDEPLOYQT_EXECUTABLE
  NAMES windeployqt
  HINTS "${_qt_bin_dir}"
  REQUIRED
)

add_custom_command(TARGET omill-lift-ui POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E env
          "PATH=${_qt_bin_dir};$ENV{PATH}"
          "${OMILL_WINDEPLOYQT_EXECUTABLE}"
          --dir "$<TARGET_FILE_DIR:omill-lift-ui>"
          --no-translations
          "$<TARGET_FILE:omill-lift-ui>"
  COMMENT "Deploying Qt runtime for omill-lift-ui"
)
