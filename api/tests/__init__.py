from api.tests.dataset_tests import TestStockDataset
from api.tests.experiment_tests import TestExperiment
from api.tests.loss_tests import TestLoss


def run_all():
    logs = []
    # NOTE: there is a memory leak in predict. if you get a segmentation fault, run the tests one by one
    for Tester in [TestStockDataset, TestLoss, TestExperiment]:
        logs.extend(Tester().run_all(verbose=False))

    print(*logs, sep='\n')

