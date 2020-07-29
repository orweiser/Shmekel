from api.tests.experiment_tests import TestExperiment
from api.tests.dataset_tests import TestStockDataset
from api.tests.loss_tests import TestLoss

TestExperiment().run_all()
TestStockDataset().run_all()
TestLoss().run_all()