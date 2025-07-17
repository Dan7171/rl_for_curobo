# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_franka_dual_arm_subgroups_no_ee_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED franka_dual_arm_subgroups_no_ee_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(franka_dual_arm_subgroups_no_ee_FOUND FALSE)
  elseif(NOT franka_dual_arm_subgroups_no_ee_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(franka_dual_arm_subgroups_no_ee_FOUND FALSE)
  endif()
  return()
endif()
set(_franka_dual_arm_subgroups_no_ee_CONFIG_INCLUDED TRUE)

# output package information
if(NOT franka_dual_arm_subgroups_no_ee_FIND_QUIETLY)
  message(STATUS "Found franka_dual_arm_subgroups_no_ee: 0.3.0 (${franka_dual_arm_subgroups_no_ee_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'franka_dual_arm_subgroups_no_ee' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${franka_dual_arm_subgroups_no_ee_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(franka_dual_arm_subgroups_no_ee_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${franka_dual_arm_subgroups_no_ee_DIR}/${_extra}")
endforeach()
