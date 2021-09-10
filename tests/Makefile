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

_dummy := $(shell mkdir -p obj)

THIRDPARTY_ROOT = third_party
UNITY_ROOT = $(THIRDPARTY_ROOT)/Unity

INCDIR := -I ../zdnn -I $(UNITY_ROOT)/src
OBJDIR := obj

include ../config.make

ifneq ($(CC),xlc)
ifneq ($(no_rpath),1)
LDFLAGS := $(LDFLAGS) -Wl,-rpath=\$$ORIGIN/../../zdnn/${SODIR}
endif
endif

C_TEST_SUPPORTFILES := $(UNITY_ROOT)/src/unity.c testsupport.c $(wildcard common_*.c)
CXX_TEST_SUPPORTFILES := $(wildcard *.cpp)
TEST_FILES := $(wildcard testDriver*.c)

C_TEST_SUPPORTOBJ := $(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(C_TEST_SUPPORTFILES)))
CXX_TEST_SUPPORTOBJ := $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(CXX_TEST_SUPPORTFILES)))
TEST_OBJ        := $(patsubst %.c,$(OBJDIR)/%.o,$(TEST_FILES))
TEST_BINARIES   := $(patsubst %.c,$(OBJDIR)/%,$(TEST_FILES))
TEST_RESULTS    := $(patsubst %.c,$(OBJDIR)/%.txt,$(TEST_FILES))

PARSED_RESULT:= `python3 resources/testresult_parser.py`

all: test

.PHONY: test

test: $(TEST_RESULTS) $(TEST_BINARIES) $(TEST_OBJ) $(C_TEST_SUPPORTOBJ) $(CXX_TEST_SUPPORTOBJ)
	@echo $(ECHOFLAGS) ${PARSED_RESULT}

# Compile
$(OBJDIR)/%.o: $(UNITY_ROOT)/src/%.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o $@ $<

$(OBJDIR)/%.o: $(ARGTABLE3_ROOT)/src/%.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o $@ $<

$(OBJDIR)/%.o: %.c
	$(CC) $(CFLAGS) $(INCDIR) -c -o $@ $<

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCDIR) -c -o $@ $<

# Link
$(OBJDIR)/testDriver_%: $(OBJDIR)/testDriver_%.o $(C_TEST_SUPPORTOBJ) $(CXX_TEST_SUPPORTOBJ)
	$(CXX) $(CXXFLAGS) $(INCDIR) -o $@ $< $(C_TEST_SUPPORTOBJ) $(CXX_TEST_SUPPORTOBJ) $(LDFLAGS) $(LDFLAGS_TEST)

# Run testcase
$(OBJDIR)/%.txt: $(OBJDIR)/%
	-$(LD_PATH_VAR)=../zdnn/$(SODIR) ZDNN_LOGLEVEL=off ./$< > $@

.PHONY: clean

clean:
	$(RM) $(OBJDIR)/* *~ core
