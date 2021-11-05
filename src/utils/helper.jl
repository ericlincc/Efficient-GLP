# Copyright 2021 The CLVR Authors
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


# This file contains gerenal helper functions.

"""Read results CSV file into a Results object."""
function csv_to_results(filepath)

    csv_reader = CSV.File(filepath)

    results = Results()
    for row in csv_reader
        logresult!(results, row.iterations, row.times, row.fvaluegaps, row.metricLPs)
    end
    return results
end
