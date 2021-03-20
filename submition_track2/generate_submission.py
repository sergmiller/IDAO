PATH_TO_TEST_DIRS = ["dataset/private_test", "dataset/publiс_test"]

def main(*args, **kwargs):

    # ur code here

    dict_pred = ...
    # save data via the scheme
    data_frame = pd.DataFrame(dict_pred, columns=["id", "classification_predictions", "regression_predictions"])
    data_frame.to_csv('submission.csv', index=False, header=True)


if __name__ == "__main__":
    main()
