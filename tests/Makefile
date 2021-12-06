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

THIRDPARTY_ROOT = third_party
UNITY_ROOT = $(THIRDPARTY_ROOT)/Unity

OBJDIR := obj
BINDIR := bin

_dummy := $(shell mkdir -p $(OBJDIR); mkdir -p $(BINDIR))

include ../config.make

INCDIR := $(CFLAGS_NOSEARCH) -I ../zdnn -I $(UNITY_ROOT)/src

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
TEST_BINARIES   := $(patsubst %.c,$(BINDIR)/%,$(TEST_FILES))
TEST_RESULTS    := $(patsubst %.c,$(BINDIR)/%.txt,$(TEST_FILES))

PARSED_RESULT:= `python3 resources/testresult_parser.py`

all: test

.PHONY: test

test: $(TEST_RESULTS) $(TEST_BINARIES) $(TEST_OBJ) $(C_TEST_SUPPORTOBJ) $(CXX_TEST_SUPPORTOBJ)
	@echo $(ECHOFLAGS) ${PARSED_RESULT}

# Compile
$(OBJDIR)/%.o: $(UNITY_ROOT)/src/%.c
	$(CC) $(INCDIR) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(ARGTABLE3_ROOT)/src/%.c
	$(CC) $(INCDIR) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.c
	$(CC) $(INCDIR) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(INCDIR) $(CXXFLAGS) -c -o $@ $<

# Link
$(BINDIR)/testDriver_%: $(OBJDIR)/testDriver_%.o $(C_TEST_SUPPORTOBJ) $(CXX_TEST_SUPPORTOBJ)
	$(CXX) $(INCDIR) $(CXXFLAGS) -o $@ $< $(C_TEST_SUPPORTOBJ) $(CXX_TEST_SUPPORTOBJ) $(LDFLAGS) $(LDFLAGS_TEST)

# Run testcase
$(BINDIR)/%.txt: $(BINDIR)/%
	-$(LD_PATH_VAR)=../zdnn/$(SODIR) ZDNN_LOGLEVEL=off ./$< > $@

.PHONY: clean

clean:
	$(RM) $(OBJDIR)/* *~ core
	$(RM) $(BINDIR)/* *~ core
