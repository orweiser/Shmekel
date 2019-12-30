from api.tests.experiment_tests import TestExperiment
from api.tests.dataset_tests import TestStockDataset

TestExperiment().run_all()
TestStockDataset().run_all()