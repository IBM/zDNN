# SPDX-License-Identifier: Apache-2.0
#
# Copyright IBM Corp. 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_dummy := $(shell mkdir -p obj)
OBJDIR := obj

include ../config.make

INCDIR := $(CFLAGS_NOSEARCH) -I ../zdnn -I .

_dummy2 := $(shell mkdir -p $(SODIR))

CFLAGS := $(INCDIR) $(CFLAGS)
CXXFLAGS := $(INCDIR) $(CXXFLAGS)

H_FILES        := $(filter-out $(wildcard convert*.h), $(wildcard *.h))
INIT_SRCFILE := zdnn_init.c
SRCFILES     := $(filter-out $(INIT_SRCFILE), $(wildcard *.c) $(wildcard *.cpp))

# Add the configure generated header
H_FILES += ../config.h

OBJFILES          := $(patsubst %.c,$(OBJDIR)/%.o,$(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCFILES)))
INIT_OBJFILE      := $(patsubst %.c,$(OBJDIR)/%.o,$(INIT_SRCFILE))

all: $(SODIR)/$(LIBNAME).so $(SODIR)/$(LIBNAME_PRIVATE).so $(ZDNN_MAKE_TARGETS)

$(INIT_OBJFILE): $(INIT_SRCFILE)
	$(CC) $(CFLAGS_INIT) $(CFLAGS_SHARED) -c $< -o $@

$(OBJDIR)/%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_SHARED) -c $< -o $@

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS_SHARED) -c $< -o $@

include $(ZDNN_TMAKE_FILES)


$(SODIR)/$(LIBNAME).so: $(INIT_OBJFILE) $(OBJFILES) $(H_FILES)
	$(LD) $(LDFLAGS_SHARED) -o $(SODIR)/$(LIBNAME).so $(INIT_OBJFILE) $(OBJFILES)

.PHONY: clean

clean:
	$(RM) $(OBJDIR)/*.o $(OBJDIR)/*.lst $(OBJDIR)/*.d *~ core $(SODIR)/* \
	*.so* ../zdnn/zdnn_private.map \
	zdnn.i zdnn.dynsyms symcheck

.PHONY: install

install: all install_shared $(ZDNN_INSTALL_TARGETS)

.PHONY: install_shared

install_shared:
	$(INSTALL) -d $(DESTDIR)$(libdir)
	$(INSTALL) -d $(DESTDIR)$(includedir)
	$(INSTALL) -t $(DESTDIR)$(libdir) $(SODIR)/$(LIBNAME).so
	$(INSTALL_DATA) -t $(DESTDIR)$(includedir) zdnn.h
