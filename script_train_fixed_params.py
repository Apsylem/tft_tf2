# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

# %%
import os
import sys

from sklearn import multiclass
pathProject = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#pathProject = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pathProject

try:
    os.chdir(pathProject)
except:
    os.chdir('/app')
    pathProject = '/app'
    

sys.path.insert(0,pathProject)
import argparse
import datetime as dte

import tensorflow.compat.v1 as tf


"""# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'"""


import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import pickle



ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)
# %%

def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False,
         modeling_type = 'regression'):
    """Trains tft based on defined model params.

    Args:
      expt_name: Name of experiment
      use_gpu: Whether to run tensorflow with GPU operations
      model_folder: Folder path where models are serialized
      data_csv_path: Path to csv file containing data
      data_formatter: Dataset-specific data fromatter (see
        expt_settings.dataformatter.GenericDataFormatter)
      use_testing_mode: Uses a smaller models and data sizes for testing purposes
        only -- switch to False to use original default settings
    """
    # %%
    #expt_name = 'kidfail'
    #output_folder = '/home/alexander/time_fusion_transformer/tft_outputs'
    #output_folder = '/app/tft_outputs'
    #config = ExperimentConfig(expt_name, output_folder)
    #formatter = config.make_data_formatter()
    #from util.general_util import dev_pickle
    #dev_pickle((expt_name,use_gpu,model_folder,data_csv_path,data_formatter,use_testing_mode,modeling_type),"main")
    #(expt_name,use_gpu,model_folder,data_csv_path,data_formatter,use_testing_mode,modeling_type) = dev_pickle(False,"main")
    
   
    #use_gpu=False
    #use_gpu=True
    #model_folder=os.path.join(config.model_folder, "fixed_klein1")
    #model_folder=os.path.join(config.model_folder, "fixed")
    #data_csv_path=config.data_csv_path
    #data_formatter=formatter
    #use_testing_mode=False
    
    # %%
    
    # %%
    num_repeats = 1
    """
    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))"""

    # Tensorflow setup
    default_keras_session = tf.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")
    print("use_gpu",use_gpu)
    print("*** Training from defined parameters for {} ***".format(expt_name))

    print("Loading & splitting data...")
    if not data_formatter.pre_split_data_available:
        raw_data = pd.read_csv(data_csv_path, index_col=0)
        train, valid, test = data_formatter.split_data(raw_data)
        
    if data_formatter.pre_split_data_available:
        train, valid, test = data_formatter.load_splitted_data()
    
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()
    
    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder
    params['modeling_type'] = modeling_type
    
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    #save the data formater
    pickle.dump(data_formatter,open(os.path.join(model_folder,"data_formatter.pkl"),'wb'))
    
    
    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))
   
    best_loss = np.Inf
    for _ in range(num_repeats):

        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

            tf.keras.backend.set_session(sess)
            
            params = opt_manager.get_next_parameters()
            
            #from util.general_util import dev_pickle
            #dev_pickle((params),"params")
            #(params) = dev_pickle(False,"params")
            model = ModelClass(params, use_cudnn=use_gpu)
            
            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())
            import time
            before_fit = time.time()
            model.fit()
            time_to_fit = time.time()-before_fit
            

            val_loss = model.evaluate()

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

            tf.keras.backend.set_session(default_keras_session)
    """# %% 
    model.model
    # %%
    from util.general_util import dev_pickle
    #dev_pickle((data),"data_from_predict", False)
    (data) = dev_pickle(False,"data_from_predict")
    data['inputs'].shape
    # %%
    #tf.reset_default_graph()
    g = tf.Graph()
    sess = tf.Session(graph = tf.Graph(),config=tf_config) 
    #with tf.Session(graph=g,config=tf_config).as_default() as sess:
    tf.keras.backend.set_session(sess)
    with g.as_default():
    
    #tf.keras.backend.set_session(sess)
        
            #tf.compat.v1.disable_eager_execution()
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)
            
        model.load(opt_manager.hyperparam_folder)
        tf.keras.backend.set_session(sess)
    # %%
    #with g.as_default(), tf.Session(config=tf_config) as sess:
    #with tf.Session(config=tf_config) as sess:
    #with tf.Session(graph=g,config=tf_config).as_default() as sess:
    with g.as_default():
        tf.keras.backend.set_session(sess)
        #tf.compat.v1.disable_eager_execution()
        #model.model(data['inputs']).mean()
        
        model.model.outputs[0].mean().eval(session = sess,feed_dict={model.model.inputs[0]:data['inputs']})
        tf.keras.backend.set_session(sess)
    # %%
    with g.as_default():
        sess.run(model.model.outputs[0].mean(), feed_dict = {model.model.inputs[0]:data['inputs']})
    # %%
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)
        
        model.load(opt_manager.hyperparam_folder)
        
        combined = model.model(data['inputs'])
        #X = tf.placeholder(tf.float32, [None, data['inputs'].shape[1],data['inputs'].shape[2]])
        #X = tf.placeholder(tf.float32, [None, 40,1])
        print("combined.mean()",combined.mean())
        print("combined.mean()",dir(combined.mean()))
  
        #mean = combined.mean().eval(session=sess,feed_dict={model.model.inputs[0]:data['inputs']})
        mean = model.model.outputs[0].mean().eval(feed_dict={model.model.inputs[0]:data['inputs']})
        print(type(mean))
        print("mean",mean)
        
    # %%
    X = tf.compat.v1.placeholder(tf.float32, [None, time_steps,combined_input_size])
    
    process_map = {
    'mean': combined.mean().eval(session=sess,feed_dict={X:inputs})    ,
    'mode': combined.mode().eval(session=sess,feed_dict={X:inputs})    
    }
    # %%
    """
    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)
    

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True, sess=sess)
        
        # %%
        #from util.general_util import dev_pickle
        #dev_pickle((output_map,data_formatter),"formatting", False)
        #(output_map,data_formatter) = dev_pickle(False,"formatting")
        
        # %%
        if modeling_type=='regression':
            targets = data_formatter.format_predictions(output_map["targets"])
            p50_forecast = data_formatter.format_predictions(output_map["p50"])
            p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            #Strips out forecast time and identifier columns.
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]
        if modeling_type=='regression':
            p50_loss = utils.numpy_normalised_quantile_loss(
                extract_numerical_data(targets), extract_numerical_data(p50_forecast),
                0.5)
            p90_loss = utils.numpy_normalised_quantile_loss(
                extract_numerical_data(targets), extract_numerical_data(p90_forecast),
                0.9)

        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
        
    if modeling_type=='regression':
        print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
            p50_loss.mean(), p90_loss.mean()))
    
    print(f"took {time_to_fit} seconds for {params['num_epochs']} epochs to fit with gpu {use_gpu}")

if __name__ == "__main__":
    def get_args():
        """Gets settings from command line."""

        experiment_names = ExperimentConfig.default_experiments
        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "-expt_name",  
            type=str,
            nargs="?",
            default="volatility",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "-output_folder",  
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "-use_gpu",  
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training.")
        parser.add_argument(
            "-use_testing_mode",  
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="yes",
            help="Whether to use testing.")
        parser.add_argument(
                "-klein",      
                type=str,
                nargs="?",
                choices=["yes", "no"],
                default="no",
                help="wether to use klein Dataset")
        parser.add_argument(
                "-num_encoder_steps",      
                type=int,
                nargs='?',
                default=56,
                help="how many intervalls should be given es as input")
        parser.add_argument(
                "-n_timesteps_forecasting",      
                type=int,
                nargs='?',
                default=20,
                help="how many intervalls should be forecasted")
        parser.add_argument(
                "-timeseries_interval",      
                type=int,
                nargs='?',
                default=6,
                help="how long one timeseries intervall should be")
        parser.add_argument(
                "-input_t_dim",      
                type=int,
                nargs='?',
                default=120,
                help="how long one timeseries intervall should be")
        parser.add_argument(
                "-num_epochs",      
                type=int,
                nargs='?',
                default=200,
                help="how many epochs to train")
        parser.add_argument(
                "-lr",      
                type=float,
                nargs='?',
                default=0.0001,
                help="base lr")
   
        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder
    
        return args.expt_name, root_folder, args.use_gpu == 'yes', \
            args.use_testing_mode == 'yes', args.klein == 'yes',\
            args.num_encoder_steps,args.n_timesteps_forecasting,args.timeseries_interval,args.input_t_dim,args.num_epochs, \
            args.lr

    # Load settings for default experiments
    name, folder, use_tensorflow_with_gpu, use_testing_mode, klein, num_encoder_steps,n_timesteps_forecasting,timeseries_interval,input_t_dim,num_epochs, lr = get_args()
    
    
    
    print("Using output folder {}".format(folder))
    
    config = ExperimentConfig(name, folder)
    formatter = config.make_data_formatter()
    formatter = formatter(root_folder=folder,
                        klein=klein,
                        num_encoder_steps = num_encoder_steps ,
                        n_timesteps_forecasting = n_timesteps_forecasting,
                        timeseries_interval = timeseries_interval,
                        input_t_dim = input_t_dim,
                        num_epochs = num_epochs,
                        lr = lr,
                        multiclass=False)
    # Customise inputs to main() for new datasets.
    
    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        model_folder=os.path.join(config.model_folder,f"binary_{use_testing_mode}_itd_{input_t_dim}_nes_{num_encoder_steps}_ntsf{n_timesteps_forecasting}_ti{timeseries_interval}_klein_{klein}_lr_{lr}_16"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=use_testing_mode,
        modeling_type='binary_classification',
        #modeling_type='multiclass_classification',
        #modeling_type='regression',
        )  # Change to false to use original default params
    # %%
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein yes -num_encoder_steps 56 -n_timesteps_forecasting 20 -timeseries_interval 6 -input_t_dim 120 -num_epochs 1
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein yes -num_encoder_steps 60 -n_timesteps_forecasting 40 -timeseries_interval 6 -input_t_dim 60 -num_epochs 1
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 56 -n_timesteps_forecasting 3 -timeseries_interval 24 -input_t_dim 60 -num_epochs 1000 -lr 0.00001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 20 -n_timesteps_forecasting 3 -timeseries_interval 24 -input_t_dim 60 -num_epochs 1000 -lr 0.00001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 10 -n_timesteps_forecasting 3 -timeseries_interval 24 -input_t_dim 60 -num_epochs 1000 -lr 0.000001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 10 -n_timesteps_forecasting 3 -timeseries_interval 24 -input_t_dim 60 -num_epochs 1000 -lr 0.0000001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 10 -n_timesteps_forecasting 6 -timeseries_interval 12 -input_t_dim 60 -num_epochs 1000 -lr 0.000001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 56 -n_timesteps_forecasting 3 -timeseries_interval 24 -input_t_dim 60 -num_epochs 1000 -lr 0.01
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 20 -n_timesteps_forecasting 3 -timeseries_interval 24 -input_t_dim 60 -num_epochs 1000 -lr 0.001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 10 -n_timesteps_forecasting 1 -timeseries_interval 48 -input_t_dim 60 -num_epochs 1000 -lr 0.001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 20 -n_timesteps_forecasting 4 -timeseries_interval 6 -input_t_dim 60 -num_epochs 1000 -lr 0.001
#python script_train_fixed_params.py -expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -use_testing_mode no -klein no -num_encoder_steps 60 -n_timesteps_forecasting 8 -timeseries_interval 6 -input_t_dim 60 -num_epochs 1000 -lr 0.001
#took 106.69269490242004 seconds for 1 epochs to fit with gpu True
#took 100.74724411964417 seconds for 1 epochs to fit with gpu False

# max while using gpu ram was 50 GB
# max whithout gpu ram was also 50 GB
