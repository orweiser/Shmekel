""" Using the API """

""" 1. what is the API? """
# This API allows you to conduct a large amount of user's
# configurable experiments, while taking care creating,
# running, saving, loading and more.

# the api is based on a number of modules, defined
# under the 'core' sub-library:
#     Experiment
#     Dataset
#     Loss
#     Model
#     Results
#     Trainer
#     BackupHandler

""" 2. basic flow """

from api.core import Experiment
my_exp = Experiment(name='my_exp')

my_exp.print()

# >>> Experiment : my_exp
# >>> --------------------------------------------------
# >>> Model: fully_connected-depth_1-width_32
# >>> Loss: categorical_crossentropy
# >>> Train Dataset: mnist_train
# >>> Val Dataset: mnist_val>>>
#
# >>> status: 'initialized'

my_exp.run()

# >>> ... keras progress bar ...
# >>> Training Done.
# >>> my_exp--Results
# >>> Metrics: val_loss val_acc loss acc
# >>> Number of Epochs: 5
# >>> Best Epoch:
# >>> Epoch 4
# >>> --------------------------------------------------
# >>> --> val_loss ...................... 0.219
# >>> --> val_acc ....................... 0.938
# >>> --> loss .......................... 0.221
# >>> --> acc ........................... 0.936

print(my_exp.status)

# >>> 'done'

# at this point, you had run an experiment. you can
# find the experiment's backup at "../Shmekel_results/default_project/my_exp/"

# there's a special class to handle the results of an experiment:

results = my_exp.results

# check out the following methods:

# results.summary()
# results.plot()
#
# best_epoch = results.get_best_epoch(metric='val_acc')
# epoch_2 = results[2]
# train_acc = results['acc']

""" 3. customize experiments (exploring hoes...) """

in section 2. we used the default experiment
here we'll explore how to customize your experiments

let's explore how the model's depth affects the experiment's results


fc_3 = Experiment(name='fc_3', model_config=dict(model='FullyConnected', depth=3))
fc_5 = Experiment(name='fc_5', model_config=dict(model='FullyConnected', depth=5))

fc_3.model.summary()
fc_5.model.summary()

we created a Model subclass, named


""" 2. The 'core' sub-library: """

""" 3. the Experiment module """
"""
the Experiment module is the main module in the API.
it can be thought of as a container of the experiment's 
different parts. 

 
"""



