# SPDX-License-Identifier: Apache-2.0
#
# Copyright IBM Corp. 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

AC_INIT([libzdnn], [0.4.0])

objdir=`pwd -P`

if test ! "`cd $srcdir; pwd -P`" = "$objdir"; then
   AC_MSG_ERROR([Configuring is only allowed in source directory! If needed, please propose a patch!])
fi

. ./config.zdnn || exit 1

# Currently objdir is equal to srcdir, but it is used in "make install"
# in order to distinguish sources and binaries.
AC_SUBST(objdir)

# Check for requirements
AC_PROG_CC
AC_PROG_CXX

AC_CONFIG_HEADERS([config.h])

AC_ARG_VAR(CC)
AC_ARG_VAR(CXX)
AC_ARG_VAR(LD)
AC_ARG_VAR(AR)
AC_ARG_VAR(ARFLAGS)
AC_ARG_VAR(CFLAGS)
AC_ARG_VAR(CFLAGS_INIT)
AC_ARG_VAR(CFLAGS_DEBUG)
AC_ARG_VAR(CFLAGS_SHARED)
AC_ARG_VAR(CFLAGS_ASM)
AC_ARG_VAR(CFLAGS_NOSEARCH)
AC_ARG_VAR(CXXFLAGS)
AC_ARG_VAR(CXXFLAGS_ASM)
AC_ARG_VAR(CPP_SYMCHECK_FLAGS)
AC_ARG_VAR(SODIR)
AC_ARG_VAR(LIBNAME)
AC_ARG_VAR(LIBSONAME)
AC_ARG_VAR(LIBNAME_PRIVATE)
AC_ARG_VAR(LIBSONAME_PRIVATE)
AC_ARG_VAR(LDFLAGS)
AC_ARG_VAR(LDFLAGS_SHARED)
AC_ARG_VAR(LDFLAGS_SHARED_EXPORTALL)
AC_ARG_VAR(LDFLAGS_TEST)
AC_ARG_VAR(ECHOFLAGS)
AC_ARG_VAR(AWK)
AC_ARG_VAR(READELF)

AC_SUBST(CFLAGS_QUOTE)
AC_SUBST(CFLAGS_QUOTE_INIT)

AC_SUBST(zdnn_config_debug, 0)
AC_SUBST(zdnn_config_no_rpath, 0)
AC_SUBST(ZDNN_TMAKE_FILES)
AC_SUBST(ZDNN_MAKE_TARGETS)
AC_SUBST(ZDNN_INSTALL_TARGETS)
AC_SUBST(LD_PATH_VAR)

AC_CHECK_PROG(AWK, awk, awk, false)
AS_IF([test x"$AWK" = x"false"], [AC_MSG_ERROR([Please install awk before configuring.])])

AC_CHECK_TOOL(READELF, readelf, "")
AS_IF([test x"$READELF" = x"false"], [AC_MSG_WARN([readelf is required for checking the exported symbols. Check will be skipped.])])

AC_ARG_ENABLE([nnpa], AS_HELP_STRING([--disable-nnpa], [Skip all NNPA facility instructions.]))
AS_IF([test "x$enable_nnpa" = "xno"], [
  AC_DEFINE(ZDNN_CONFIG_NO_NNPA, 1, [Skip all NNPA facility instructions.])
])

AC_ARG_ENABLE([debug], AS_HELP_STRING([--enable-debug], [Enable additional checking, error reporting, disable compiler optimizations, and add debug information]))
AS_IF([test "x$enable_debug" = "xyes"], [
  zdnn_config_debug=1
  CFLAGS="${CFLAGS} ${CFLAGS_DEBUG}"
  CFLAGS_INIT="${CFLAGS_INIT} ${CFLAGS_DEBUG}"
  CXXFLAGS="${CXXFLAGS} ${CFLAGS_DEBUG}"
  AC_DEFINE(ZDNN_CONFIG_DEBUG, 1, [Enable additional checking, error reporting, disable compiler optimizations, and add debug information])
])

AC_ARG_ENABLE([listings], AS_HELP_STRING([--enable-listings], [Make 'make all' generate assembler listings]))
AS_IF([test "x$enable_listings" = "xyes"], [
  ZDNN_MAKE_TARGETS="${ZDNN_MAKE_TARGETS} listings"
])

AC_ARG_ENABLE([test-rpath], AS_HELP_STRING([--disable-test-rpath], [Don't set the rpath in the test binaries to keep them relocatable]))
AS_IF([test "x$enable_test_rpath" = "xno"], [zdnn_config_no_rpath=1])

AC_ARG_ENABLE([expensive-optimizations], AS_HELP_STRING([--disable-expensive-optimizations], [Disable expensive compiler optimizations.]))
AS_IF([test "x$enable_expensive_optimizations" != "xno"], [ CFLAGS="${CFLAGS} ${CFLAGS_OPT_EXPENSIVE}" ])


# Generate output
AC_CONFIG_FILES([config.make])

AC_OUTPUT
