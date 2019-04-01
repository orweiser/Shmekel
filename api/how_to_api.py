""" Using the API """

""" 1. what is the API? """

"""
This API allows you to conduct a large amount of user's configurable experiments, while taking care of creating,
running, saving, loading and more.

the API is based on a number of modules, defined under api.core:

    Experiment
    Dataset
    Loss
    Model
    Results
    Trainer
    BackupHandler

Some possible future modules:
    
    Metrics
    Augmentations
    Optimizer

"""


""" 2. basic flow """

# You may run this file in console and see the expected behavior by yourself.

from api.core import Experiment
# my_exp = Experiment(name='my_exp')
# my_exp.print()

"""
>>> Experiment : my_exp
>>> --------------------------------------------------
>>> Model: fully_connected-depth_1-width_32
>>> Loss: categorical_crossentropy
>>> Train Dataset: mnist_train
>>> Val Dataset: mnist_val
>>>
>>> status: 'initialized'
"""

# my_exp.run()

"""
>>> ... keras progress bar ...
>>> Training Done.
>>> my_exp--Results
>>> Metrics: val_loss val_acc loss acc
>>> Number of Epochs: 5
>>> Best Epoch:
>>> Epoch 5
>>> --------------------------------------------------
>>> --> val_loss ...................... 0.219
>>> --> val_acc ....................... 0.938
>>> --> loss .......................... 0.221
>>> --> acc ........................... 0.936
"""

# print(my_exp.status)

"""
>>> 'done'

at this point, you had run an experiment. you can find the experiment's backup
at "../Shmekel_results/default_project/my_exp/"

there's a special class to handle the results of an experiment:
"""

# results = my_exp.results

"""
check out the following methods:

results.summary()
results.plot()

best_epoch = results.get_best_epoch(metric='val_acc')
epoch_2 = results[2]
train_acc = results['acc']
"""

""" 3. customize experiments """

"""
In section 2. we used the default experiment. Here you'll learn specifically, how to customize your experiments.
But first, let's explore how the model's depth affects the experiment's results by creating two experiments with 
fully connected models, and different depth.
"""

# fc_3 = Experiment(name='fc_3', model_config=dict(model='FullyConnected', depth=3))
# fc_5 = Experiment(name='fc_5', model_config=dict(model='FullyConnected', depth=5))
lstm_exp = Experiment(name='LSTM!', model_config=dict(model='LSTM', input_shape=(1, 4), output_shape=(2,)),
                      train_dataset_config=dict(dataset='StocksDataset', val_mode=False),
                      val_dataset_config=dict(dataset='StocksDataset', val_mode=True))

# print(fc_3.model)
# print(fc_5.model)
print(lstm_exp.model)

lstm_exp.start()

"""
>>> fully_connected-depth_3-width_32
>>> fully_connected-depth_5-width_32

Since we've implemented custom fully_connected model we can now run the experiments
and view results as described above, but we can also add a print function that fits our needs, or implement a custom
Results class that can do anything that suits us (prints, plots etc.).

Similarly, we could have customized other properties of the experiment
found at "api/datasets", "api/losses" etc.

You should understand these parameters before you can create a custom experiment:

    - When creating an Experiment, a *unique* name must be provided via the argument 'name' to prevent API collisions

The Experiment's sub-modules are specified by the following arguments (all dictionaries):
    - model_config: configure the architecture of the neural network in use.
        it creates an instance of 'api.core.model.Model' according to the config
        and the available architectures found under 'api.models'
        
            ** config must contain a key 'model' with the value being the name of the
                architecture you want ('FullyConnected', 'LSTM', etc.)
                go to 'api.models' to see available architectures
    
    - train_config: configure your training routine, i.e. number of epochs, batch size.
        Note: currently, you can configure the optimizer and data augmentations via the train_config,
            but it will be replaced by other sub-modules in the future
            
        for more information: "api.core.trainer.Trainer" ( Good luck :) )
        
    - backup_config: configure saving locations on your computer
        "api.core.backup_handler.DefaultLocal"
     
    - train_dataset_config: 
            ** config must contain a key 'dataset' with the value being the name of the
                dataset you want ('MNIST', 'StockDataset', etc.)
                go to 'api.datasets' to see available datasets
     
    - val_dataset_config: same as 'train_dataset', usually you would need to set key 'val_mode' equal True
    
    - loss_config: configure the Loss for the Experiment by including the key 'loss' with the value being the name 
        of the Loss. Supports all keras.losses
        
 
each parameters affects the appropriate module and can be expanded. A detailed guide is in the next chapter.
"""

""" 4. Creating your own modules """

"""
If you've gone this far, you probably ask yourself (because you need/want to) how can I contribute to this magnificent 
well documented beast of an API. 

You might need to add a model. In order to do so, you need to follow these instructions on how to implement a sub class
of api.core.model.Model (see api.models.lstm), 
 
    - Create a python file with the name of your model. as convention, 
        use lower case letters with underscores to separate words, e.g 'my_model'
    
    - Create your sub-class. for convention, use capital letters at words beginning and no spaces:
            from api.core.model import Model
            class MyModel(Model):
        
    - You now need to implement the abstract methods.
            def init(self, *args, **kwargs):
                pass
        
            def get_input_output_tensors(self):
                pass
        
            def __str__(self):
                pass
        
            def get_default_config(self) -> dict:
                pass
        
        ** go to api.core.model and see description of those methods.
        
    - When you're done implementing, make sure to add your model to the models getter.
        go to api.models.__init__
        import your model: "from api.models.my_model import Model"
        add it to the getter:
            "if model == 'MyModel': return MyModel(model=model, **kwargs)"
            
            happy coding :)

"""