#!/bin/bash

generate_tests() {
  local input_path=$1
  local output_dir="tests/phmdoctest"
  local base=$(basename "$input_path" .md)
  local output_file="$output_dir/test_${base}.py"

  echo "Generating test for $input_path to $output_file"
  python -m phmdoctest "$input_path" --outfile "$output_file"
}

echo "Removing Previous Tests"
rm -rf ./tests/phmdoctest
mkdir ./tests/phmdoctest

echo "Processing README.md"
generate_tests "README.md"

echo "Processing Markdown files in ./docs/docs"
find ./docs/docs -name "*.md" -print0 | while IFS= read -r -d '' file; do
  generate_tests "$file"
done

echo "Don't Test Changelog"
rm ./tests/phmdoctest/test_changelog.py
