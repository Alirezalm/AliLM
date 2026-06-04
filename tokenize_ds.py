from utils import tokenize_dataset


def main():
    data_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    tokenize_dataset(
        data_path,
        "./data/ts_valid",
        100_000_000,
    )


if __name__ == "__main__":
    main()
