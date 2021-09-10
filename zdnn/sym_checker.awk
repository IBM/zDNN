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

# The sym_checker.awk file cross checks the symbols declared in zdnn.h,
# zdnn.map and the exported symbols of the shared library.
#
# Usage:
# awk [ -v debug=<DBG> ] -f <ZDNN_H_PREPROCESSED> <ZDNN_MAP> <ZDNN_DYNSYMS>
# <ZDNN_H_PREPROCESSED>: gcc -E -c <src>/zdnn/zdnn.h -o zdnn_preprocessed.h
# <ZDNN_MAP>: <src>/zdnn/zdnn.map
# <ZDNN_DYNSYMS>: readelf -W --dyn-syms libzdnn.so
# <DBG>: To get debug-output, run with debug > 0
#        (increase number to get more verbose output)
#
# In case of errors, those are dumped and return-code is != 0

function dbg(level, msg) {
    if (debug >= level)
	print "#"fi"."FNR": "msg
}

function fail(msg) {
    fails++
    print "ERROR #"fails" in "curr_file":"FNR": "msg
}

BEGIN {
    if (debug == "") debug=0

    # The input lines are dumped starting at this debug level
    # see dbg(debug_line, ...)
    debug_line=3

    # file-index: 1 is the first input-file
    fi=0
    curr_file="BEGIN"
    use_argv_file=0
    fi_hdr=1
    hdr_file="zdnn.h"
    fi_map=2
    map_file="zdnn.map"
    fi_dyn=3
    dyn_file=".dynsym-symbol-table"

    dbg(0, "BEGIN of sym_checker.awk")

    hdr_line=""
    hdr_in_zdnn_h=0

    map_version_node_reset()
    map_tasks_cnt=0

    fails=0
}

FNR == 1 {
    fi++
    if (use_argv_file) {
	curr_file=ARGV[fi]
	if (fi == fi_hdr)
	    hdr_file=curr_file
	else if (fi == fi_map)
	    map_file=curr_file
	else if (fi == fi_dyn)
	    dyn_file=curr_file
    } else {
	if (fi == fi_hdr)
	    curr_file=hdr_file
	else if (fi == fi_map)
	    curr_file=map_file
	else if (fi == fi_dyn)
	    curr_file=dyn_file
    }
    dbg(0, "Processing file "ARGV[fi]" => "curr_file)
}

################################################################################
# Processing <ZDNN_H_PREPROCESSED>: variable-, function-prefix=hdr
################################################################################
function hdr_process_line(line, _line_words, _sym) {
    # Skip typedefs
    if (substr(line, 1, length("typedef")) == "typedef") {
	dbg(1, "=> Skip typedef")
	return
    }

    # Remove "((*))", "(*)" and "[*]"
    sub(/\(\(.*\)\)/, "", line)
    sub(/\(.*\)/, "", line)
    sub(/\[.*\]/, "", line)

    # Remove trailing ";" and whitespaces before
    sub(/ *;$/, "", line)

    # Get the last word
    split(line, _line_words)
    _sym=_line_words[length(_line_words)]

    # Remove leading *
    sub(/^\*/, "", _sym)

    if (_sym in hdr_syms) {
	fail("Found a duplicate symbol "_sym" in "hdr_file)
    } else {
	dbg(1, "Add to hdr_syms: "_sym)
	# 0 / 1: if found as default symbol (@@ZDNN) in .dynsym
	hdr_syms[_sym]=0
    }
}

fi == fi_hdr && /^#/ {
    dbg(debug_line, "hdr-comment: "$0)
    # Matching lines:
    # # 28 "/usr/include/inttypes.h" 2 3 4
    # # 23 "../zdnn/zdnn.h" 2
    hdr_basename=$3

    # '"/PATH/TO/zdnn.h"' => 'zdnn.h"'
    sub(".*/", "", hdr_basename)

    # 'zdnn.h"' => 'zdnn.h'
    sub("\"$", "", hdr_basename)

    if (hdr_basename == "zdnn.h")
	hdr_in_zdnn_h=1
    else
	hdr_in_zdnn_h=0

    dbg(debug_line, "Next lines come from: "hdr_basename" (hdr_in_zdnn_h="hdr_in_zdnn_h"; "$3")")
    next
}
fi == fi_hdr && NF > 0 {
    dbg(debug_line, "hdr(hdr_in_zdnn_h="hdr_in_zdnn_h"): "$0)
    if (hdr_in_zdnn_h != 0) {

    # unless hdr_line is empty, otherwise concat the next line with a space in between
    # so case like this will work:
    #
    # void
    # bar(char c);
    if (hdr_line != "") {
        hdr_line=hdr_line " " $0
    } else {
        hdr_line=hdr_line $0
    }

	if (index(hdr_line, ";") != 0) {
	    dbg(debug_line, "zdnn.h: "hdr_line)

	    # E.g. for structs, we need to ensure that we have the whole
	    # declaration. Compare the number of '{' and '}'.
	    # Note: structs use ";" for each member!
	    hdr_nr_brace_begin = sub(/{/, "{", hdr_line)
	    hdr_nr_brace_end = sub(/}/, "}", hdr_line)
	    dbg(2, "hdr_nr_brace_begin="hdr_nr_brace_begin"; hdr_nr_brace_end="hdr_nr_brace_end)
	    if (hdr_nr_brace_begin != hdr_nr_brace_end)
		next

	    hdr_process_line(hdr_line)
	    hdr_line=""
	}
    }
}

################################################################################
# Processing <ZDNN_MAP>: variable-, function-prefix=map
################################################################################
function map_version_node_reset() {
    map_curr_version_node=""
    map_curr_version_node_scope=""
    map_curr_version_node_global_symbols=-1
}

fi == fi_map && ( NF == 0 || /^#/ ) {
    dbg(debug_line, "map-empty-or-comment: "$0)
    if ($1 == "#" && $2 == "Task:") {
	map_tasks[map_tasks_cnt]=FNR": "$0
	map_tasks_cnt++
    }
    # Skip comments or empty lines
    next
}

fi == fi_map && /^ZDNN_.*{$/ {
    dbg(debug_line, "map-version-node-begin: "$0)
    # Matching lines:
    #  ZDNN_1.0 {

    map_version_node_reset()
    map_curr_version_node=$1
    next
}

fi == fi_map && map_curr_version_node != "" && $1 ~ /.*:$/ {
    dbg(debug_line, "map-version-node-scope: "$0)
    # Matching lines:
    #    global:
    #    local: *;

    map_curr_version_node_scope=$1
    if ($1 == "global:") {
	if (map_curr_version_node_global_symbols != -1)
	    fail("Found duplicate 'global:' scope in version-node "map_curr_version_node)
	else
	    map_curr_version_node_global_symbols=0
    } else if ($1 == "local:") {
	if ($2 != "*;")
	    fail("As we list all global-scoped symbols in "map_file", only 'local: *;' is allowed!")
    } else {
	fail("Found invalid map-version-node-scope: "$1)
	map_version_node_reset()
    }
    next
}

fi == fi_map && map_curr_version_node != "" && /^};$/ {
    dbg(debug_line, "map-version-node-end: "$0)
    # Matching lines:
    # };

    if (map_curr_version_node_global_symbols <= 0)
	fail("No global-scoped symbols found in version-node "map_curr_version_node" (Either remove the empty version-node or add a global-scoped symbol)")

    map_version_node_reset()
    next
}

fi == fi_map && map_curr_version_node != "" && map_curr_version_node_scope == "global:" {
    dbg(debug_line, "map-global-symbol: "$0)
    # Matching lines:
    #      zdnn_init;

    map_name=$1
    if (sub(/;$/, "", map_name) == 0) {
	fail("Failed to remove ';' at end of map-global-symbol: "map_name)
    } else {
	map_key=map_name"@"map_curr_version_node
	if (map_key in map_syms) {
	    fail("Found a duplicate symbol "map_name" in version node "map_curr_version_node" in "map_file)
	} else {
	    dbg(1, "Add to map_syms: "map_key)
	    map_curr_version_node_global_symbols++
	    # 0 / 1: if found as symbol (@@ZDNN or @ZDNN) in .dynsym
	    map_syms[map_key]=0
	}
    }
    next
}

fi == fi_map {
    dbg(debug_line, "map-unhandled: "$0)
    fail("Found map-unhandled line (Perhaps a local-scope symbol?): "$0)
}

################################################################################
# Processing <ZDNN_DYNSYMS>: variable-, function-prefix=dyn
################################################################################
fi == fi_dyn && NF >= 8 {
    # Process lines like:
    #    Num:    Value          Size Type    Bind   Vis      Ndx Name
    #     26: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND zdnn_get_status_str
    #     54: 00000000000083b0    56 FUNC    GLOBAL DEFAULT   12 zdnn_sub@@ZDNN_1.0
    # Note: Skip lines where Ndx is "Ndx" or e.g. "UND" or "ABS"
    dbg(debug_line, "dyn: "$0)
    dyn_name=$8 # e.g. zdnn_sub@@ZDNN_1.0

    if ($7 == "UND") {
	if (dyn_name ~ /^zdnn/)
	    fail("Found undefined symbol in "dyn_file": "dyn_name ". Better provide an implementation for this symbol in libzdnn.so?")
	else
	    dbg(1, "Skipping UND symbol: "dyn_name)
	next
    } else if ($7 ~ /[0-9]+/) {
	# 'zdnn_sub@@ZDNN_1.0' => 'zdnn_sub' and return > 0 if "@" was found
	dyn_sym=dyn_name
	if (sub("@.*", "", dyn_sym) == 0) {
	    fail("Found unversioned symbol in "dyn_file": "dyn_name)
	} else {
	    dyn_ver=substr(dyn_name, length(dyn_sym) + 1)

	    if (substr(dyn_ver, 1, 2) == "@@") {
		if (dyn_sym in hdr_syms) {
		    dbg(1, "Found default symbol "dyn_name" in "dyn_file", which is included in "hdr_file)
		    hdr_syms[dyn_sym]=1
		} else {
		    fail("Default symbol "dyn_name" from "dyn_file" was not found in "hdr_file". All exported default symbols should belong to the zdnn.h header file!")
		}
		dyn_ver=substr(dyn_ver, 3)
	    } else {
		dyn_ver=substr(dyn_ver, 2)
	    }

	    dyn_key=dyn_sym"@"dyn_ver
	    if (dyn_key in map_syms) {
		dbg(1, "Found symbol "dyn_name" in "dyn_file" in version-node "dyn_ver" in "map_file)
		map_syms[dyn_key]=1
	    } else {
		fail("Symbol "dyn_name" from "dyn_file" was not found in "map_file". Please list the symbol in the corresponding version-node "dyn_ver)
	    }
	}
    }
}

################################################################################
# Perform checks at the end
################################################################################
END {
    curr_file="END"
    fi++
    FNR=0

    dbg(0, "Processing END of sym_checker.awk")
    if (fi != 4)
	fail("Please pass all files as seen in Usage of this awk file.")


    FNR++
    dbg(1, "Symbols found in "hdr_file" (val=1 if found as default symbol (@@ZDNN) in .dynsym):")
    for (key in hdr_syms) {
	val=hdr_syms[key]
	dbg(1, "hdr_syms: key="key"; val="val)
	if (val == 0) {
	    fail("For symbol "key" from "hdr_file", there is no default symbol in "dyn_file". A program/library can't link against this symbol! For new symbols, you have to add it to a new version-node in "map_file)
	}
    }

    FNR++
    dbg(1, "Symbols found in "map_file" (val=1 if found as symbol (@@ZDNN or @ZDNN) in .dynsym):")
    for (key in map_syms) {
	val=map_syms[key]
	dbg(1, "map_syms: key="key"; val="val)

	if (val == 0) {
	    fail("For symbol "key" from "map_file", there is no symbol in "dyn_file". Please provide an [old] implementation for this symbol!")
	}
    }

    FNR++
    if (fails > 0) {
	print ""
	print "Please also have a look at the tasks described in "map_file":"
	for (i = 0; i < map_tasks_cnt; i++)
	    print map_tasks[i]
	print ""
	dbg(0, "END of sym_checker.awk: "fails" ERRORs")
	exit 1
    } else {
	dbg(0, "END of sym_checker.awk: SUCCESS")
    }
}
