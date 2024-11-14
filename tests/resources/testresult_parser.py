# SPDX-License-Identifier: Apache-2.0
#
# Copyright IBM Corp. 2021, 2024
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
#
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", type=str, default="bin/", help="directory with test results"
)
args = parser.parse_args()
results_dir = args.dir
if not results_dir.endswith("/"):
    results_dir = f"{results_dir}/"

results_path = f"{results_dir}testDriver*.txt"

num_passes = 0
num_ignores = 0
num_fails = 0

# accumulative pass/ignore/fail messages
pass_txt = ""
ignore_txt = ""
fail_txt = ""

# other things we care about (crashes, etc.)
notes_txt = ""

# escaped newline for make
NL = "\\n"

for filename in glob.glob(results_path):

    if os.stat(filename).st_size == 0:
        notes_txt = notes_txt + filename + " is a 0 byte file.  Likely crashed." + NL
    else:
        for line in open(filename, "r"):

            line = line.strip()
            if ":PASS" in line or ":IGNORE" in line or ":FAIL" in line:
                test_file, line_num, test_name, status = line.split(":", 3)

                test_file = test_file.strip()
                line_num = line_num.strip()
                test_name = test_name.strip()
                status = status.strip()

                if "PASS" in status:
                    num_passes = num_passes + 1
                    pass_txt = pass_txt + test_file + ":" + test_name + NL

                if "IGNORE" in status:
                    num_ignores = num_ignores + 1
                    ignore_txt = (
                        ignore_txt + test_file + ":" + test_name + ":" + status + NL
                    )

                if "FAIL" in status:
                    num_fails = num_fails + 1
                    fail_txt = (
                        fail_txt + test_file + ":" + test_name + ":" + status + NL
                    )

        # Unity prints a "final status" text at the end.  If the last line isn't either
        # of these then likely the testDriver crashed
        if line != "FAIL" and line != "OK":
            notes_txt = notes_txt + filename + " did not finish.  Likely crashed." + NL

# print the whole report as one big string so that make won't random insert a space
# in between every print()
print(
    f"-----------------------{NL}PASSES{NL}-----------------------{NL}"
    + pass_txt
    + f"{NL}-----------------------{NL}IGNORES{NL}-----------------------{NL}"
    + ignore_txt
    + f"{NL}-----------------------{NL}FAILURES{NL}-----------------------{NL}"
    + fail_txt
    + f"{NL}------------------------------------------------------------{NL}"
    + f"total = {num_passes + num_ignores + num_fails}, num_passes = {num_passes},"
    + f" num_ignores = {num_ignores}, num_fails = {num_fails}{NL}"
    + f"{NL}------------------------------------------------------------{NL}"
    + notes_txt
)
