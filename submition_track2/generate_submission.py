import pathlib as path

PATH_TO_TEST_DIRS = [path.Path('tests/private_test'), path.Path('tests/publiс_test')]

def main(*args, **kwargs):

    # ur code here

    dict_pred = ...
    # save data via the scheme
    data_frame = pd.DataFrame(dict_pred, columns=["id", "classification_predictions", "regression_predictions"])
    data_frame.to_csv('submission.csv', index=False, header=True)


if __name__ == "__main__":
    main()
