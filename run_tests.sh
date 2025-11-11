#!/bin/bash

#
# Run all the tests for vibehdf5
#

python tests/test_csv_import.py
python tests/test_filter_export.py
python tests/test_filter_persistence.py
python tests/test_filter_workflow.py
python tests/test_new_file_feature.py
python tests/test_syntax_highlighting.py