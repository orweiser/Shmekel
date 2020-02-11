from api.tests.dataset_tests import TestStockDataset
from api.tests.experiment_tests import TestExperiment
from api.tests.loss_tests import TestLoss


def run_all():
    logs = sum([Tester().run_all(verbose=False) for Tester in [
        TestStockDataset, TestExperiment, TestLoss
    ]], [])

    print(*logs, sep='\n')

