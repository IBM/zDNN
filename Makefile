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

.PHONY: all
all: config.make
	$(MAKE) all -C zdnn
	$(MAKE) all -C tests

.PHONY: build
build: config.make
	$(MAKE) all -C zdnn

.PHONY: test
test: config.make
	$(MAKE) all -C tests

.PHONY: clean
clean: config.make
	$(MAKE) clean -C tests
	$(MAKE) clean -C zdnn

.PHONY: distclean
distclean: clean
	rm -f config.log config.status config.make config.h


.PHONY: install
install: build
	$(MAKE) install -C zdnn

config.make:
# Use this additional check to allow make invocation "make -B build" in jenkins.
ifeq ($(wildcard config.make),)
	@echo "Please use configure first";
	exit 1
endif
