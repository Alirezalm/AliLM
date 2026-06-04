from utils import tokenize_dataset


def main():
    data_path = "./data/owt_train.txt"
    tokenize_dataset(
        data_path,
        "ow_train",
        100_000_000,
    )


if __name__ == "__main__":
    main()
