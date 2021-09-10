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
INCDIR := -I ../zdnn -I .
OBJDIR := obj

include ../config.make

_dummy2 := $(shell mkdir -p $(SODIR))

CFLAGS := $(CFLAGS) $(INCDIR)
CXXFLAGS := $(CXXFLAGS) $(INCDIR)

H_FILES      := $(filter-out $(wildcard convert*.h), $(wildcard *.h))
C_SRCFILES   := $(wildcard *.c)
CXX_SRCFILES := $(wildcard *.cpp)

# Add the configure generated header
H_FILES += ../config.h

C_OBJFILES          := $(patsubst %.c,$(OBJDIR)/%.o,$(C_SRCFILES))
CXX_OBJFILES        := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CXX_SRCFILES))

$(OBJDIR)/%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_SHARED) -c $< -o $@

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS_SHARED) -c $< -o $@

all: $(SODIR)/$(LIBNAME).so $(SODIR)/$(LIBNAME_PRIVATE).so $(ZDNN_MAKE_TARGETS)

include $(ZDNN_TMAKE_FILES)


$(SODIR)/$(LIBNAME).so: $(C_OBJFILES) $(CXX_OBJFILES) $(H_FILES)
	$(LD) $(LDFLAGS_SHARED) -o $(SODIR)/$(LIBNAME).so $(C_OBJFILES) $(CXX_OBJFILES)

.PHONY: clean

clean:
	$(RM) $(OBJDIR)/*.o $(OBJDIR)/*.lst $(OBJDIR)/*.d *~ core $(SODIR)/* \
	*.so* ../zdnn/zdnn_private.map \
	zdnn_preprocessed.h zdnn.dynsyms

.PHONY: install

install: all install_shared $(ZDNN_INSTALL_TARGETS)

.PHONY: install_shared

install_shared:
	$(INSTALL) -d $(DESTDIR)$(libdir)
	$(INSTALL) -d $(DESTDIR)$(includedir)
	$(INSTALL) -t $(DESTDIR)$(libdir) $(SODIR)/$(LIBNAME).so
	$(INSTALL_DATA) -t $(DESTDIR)$(includedir) zdnn.h
