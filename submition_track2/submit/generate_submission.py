import sys
sys.path.append('..')
from utils import dataset, pipe

PATH_TO_TEST_DIRS = ['tests/private_test', 'tests/publiс_test']


def main(*args, **kwargs):
    public_test_dataset = dataset.build_dataset(path=kwargs.get("path2public"), limit=kwargs.get("limit"))
    private_test_dataset = dataset.build_dataset(path=kwargs.get('path2private'), limit=kwargs.get('limit'))

    model1 = kwargs.get('model1')
    model2 = kwargs.get('model2')

    public_test_predictions = pipe.apply_all_models_to_test_dataset(
        public_test_dataset, model1, model2, 'public_test')
    private_test_predictions = pipe.apply_all_models_to_test_dataset(
        private_test_dataset, model1, model2, 'private_test')

    test_submit = dataset.LabeledDataset.merge(
        public_test_predictions, private_test_predictions)

    dataset.dataset2submit_csv(test_submit, "submission.csv")

# debug start #
DEBUG_PATH_TO_TRAIN_DIR = '../../data/test_few_data/'
DEBUG_DATA_LIMIT = 6

class FakeModelWithProjection:
    def __init__(self):
        pass
    def predict(self, *args, **kwargs):
        return [1 for x in range(DEBUG_DATA_LIMIT)]
    def predict_proba(self, *args, **kwargs):
        return [0.5 for x in range(DEBUG_DATA_LIMIT)]
# debug end #


if __name__ == "__main__":
    main(
        model1=FakeModelWithProjection(),
        model2=FakeModelWithProjection(),
        path2public=DEBUG_PATH_TO_TRAIN_DIR,
        path2private=DEBUG_PATH_TO_TRAIN_DIR)
