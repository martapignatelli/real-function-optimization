# Install script for directory: /home/aless/challenge1-male/muparserx

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/aless/challenge1-male/muparserx/libmuparserx.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/aless/challenge1-male/muparserx/muparserx.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/muparserx" TYPE FILE FILES
    "/home/aless/challenge1-male/muparserx/cmake/muparserxConfig.cmake"
    "/home/aless/challenge1-male/muparserx/muparserxConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/muparserx" TYPE FILE FILES
    "/home/aless/challenge1-male/muparserx/parser/mpDefines.h"
    "/home/aless/challenge1-male/muparserx/parser/mpError.h"
    "/home/aless/challenge1-male/muparserx/parser/mpFuncCmplx.h"
    "/home/aless/challenge1-male/muparserx/parser/mpFuncCommon.h"
    "/home/aless/challenge1-male/muparserx/parser/mpFuncMatrix.h"
    "/home/aless/challenge1-male/muparserx/parser/mpFuncNonCmplx.h"
    "/home/aless/challenge1-male/muparserx/parser/mpFuncStr.h"
    "/home/aless/challenge1-male/muparserx/parser/mpFwdDecl.h"
    "/home/aless/challenge1-male/muparserx/parser/mpICallback.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIOprt.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIOprtBinShortcut.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIPackage.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIPrecedence.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIToken.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIValReader.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIValue.h"
    "/home/aless/challenge1-male/muparserx/parser/mpIfThenElse.h"
    "/home/aless/challenge1-male/muparserx/parser/mpMatrix.h"
    "/home/aless/challenge1-male/muparserx/parser/mpMatrixError.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtBinAssign.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtBinCommon.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtBinShortcut.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtCmplx.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtIndex.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtMatrix.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtNonCmplx.h"
    "/home/aless/challenge1-male/muparserx/parser/mpOprtPostfixCommon.h"
    "/home/aless/challenge1-male/muparserx/parser/mpPackageCmplx.h"
    "/home/aless/challenge1-male/muparserx/parser/mpPackageCommon.h"
    "/home/aless/challenge1-male/muparserx/parser/mpPackageMatrix.h"
    "/home/aless/challenge1-male/muparserx/parser/mpPackageNonCmplx.h"
    "/home/aless/challenge1-male/muparserx/parser/mpPackageStr.h"
    "/home/aless/challenge1-male/muparserx/parser/mpPackageUnit.h"
    "/home/aless/challenge1-male/muparserx/parser/mpParser.h"
    "/home/aless/challenge1-male/muparserx/parser/mpParserBase.h"
    "/home/aless/challenge1-male/muparserx/parser/mpParserMessageProvider.h"
    "/home/aless/challenge1-male/muparserx/parser/mpRPN.h"
    "/home/aless/challenge1-male/muparserx/parser/mpScriptTokens.h"
    "/home/aless/challenge1-male/muparserx/parser/mpStack.h"
    "/home/aless/challenge1-male/muparserx/parser/mpStringConversionHelper.h"
    "/home/aless/challenge1-male/muparserx/parser/mpTest.h"
    "/home/aless/challenge1-male/muparserx/parser/mpTokenReader.h"
    "/home/aless/challenge1-male/muparserx/parser/mpTypes.h"
    "/home/aless/challenge1-male/muparserx/parser/mpValReader.h"
    "/home/aless/challenge1-male/muparserx/parser/mpValue.h"
    "/home/aless/challenge1-male/muparserx/parser/mpValueCache.h"
    "/home/aless/challenge1-male/muparserx/parser/mpVariable.h"
    "/home/aless/challenge1-male/muparserx/parser/suSortPred.h"
    "/home/aless/challenge1-male/muparserx/parser/suStringTokens.h"
    "/home/aless/challenge1-male/muparserx/parser/utGeneric.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/aless/challenge1-male/muparserx/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
