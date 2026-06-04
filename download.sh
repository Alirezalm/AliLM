#!/usr/bin/env bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"

download_if_missing() {
	local url="$1"
	local output="$2"

	if [[ -f "$output" ]]; then
		echo "Skipping $output; already exists"
		return 0
	fi

	wget -O "$output" "$url"
}

mkdir -p "$data_dir"
cd "$data_dir"

download_if_missing \
	"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
	"TinyStoriesV2-GPT4-train.txt"
download_if_missing \
	"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" \
	"TinyStoriesV2-GPT4-valid.txt"

download_if_missing \
	"https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz" \
	"owt_train.txt.gz"
if [[ ! -f "owt_train.txt" && -f "owt_train.txt.gz" ]]; then
	gunzip -f owt_train.txt.gz
fi

download_if_missing \
	"https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz" \
	"owt_valid.txt.gz"
if [[ ! -f "owt_valid.txt" && -f "owt_valid.txt.gz" ]]; then
	gunzip -f owt_valid.txt.gz
fi