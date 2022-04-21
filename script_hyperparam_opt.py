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
"""Main hyperparameter optimisation script.

Performs random search to optimize hyperparameters on a single machine. For new
datasets, inputs to the main(...) should be customised.
"""
# %%
import os
import sys
pathProject = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#pathProject = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pathProject
# %%
try:
    os.chdir(pathProject)
except:
    os.chdir('/app')
    pathProject = '/app'
    
# %%
sys.path.insert(0,pathProject)
import argparse
import datetime as dte

import sys
sys.path.append(pathProject)
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)

def main(expt_name, use_gpu, restart_opt, model_folder, hyperparam_iterations,
         data_csv_path, data_formatter):
  """Runs main hyperparameter optimization routine.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    restart_opt: Whether to run hyperparameter optimization from scratch
    model_folder: Folder path where models are serialized
    hyperparam_iterations: Number of iterations of random search
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
  """
  
  #if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
  #  raise ValueError(
  #      "Data formatters should inherit from" +
  #      "AbstractDataFormatter! Type={}".format(type(data_formatter)))

  default_keras_session = tf.compat.v1.keras.backend.get_session()

  if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

  else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

  print("### Running hyperparameter optimization for {} ###".format(expt_name))
  print("Loading & splitting data...")
  if not data_formatter.pre_split_data_available:
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    
  if data_formatter.pre_split_data_available:
    train, valid, test = data_formatter.load_splitted_data()
    
    
  train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
  )

  # Sets up default params
  fixed_params = data_formatter.get_experiment_params()
  param_ranges = ModelClass.get_hyperparm_choices()
  fixed_params["model_folder"] = model_folder

  print("*** Loading hyperparm manager ***")
  opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

  success = opt_manager.load_results()
 
  if success and not restart_opt:
    print("Loaded results from previous training")
  else:
    print("Creating new hyperparameter optimisation")
    opt_manager.clear()

  print("*** Running calibration ***")
  while len(opt_manager.results.columns) < hyperparam_iterations:
    print("# Running hyperparam optimisation {} of {} for {}".format(
        len(opt_manager.results.columns) + 1, hyperparam_iterations, "TFT"))

    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:

      tf.compat.v1.keras.backend.set_session(sess)

      params = opt_manager.get_next_parameters()
      
      model = ModelClass(params, use_cudnn=use_gpu)

      if not model.training_data_cached():
        model.cache_batched_data(train, "train", num_samples=train_samples)
        model.cache_batched_data(valid, "valid", num_samples=valid_samples)

      sess.run(tf.compat.v1.global_variables_initializer())
      model.fit()

      val_loss = model.evaluate()

      if np.allclose(val_loss, 0.) or np.isnan(val_loss):
        # Set all invalid losses to infintiy.
        # N.b. val_loss only becomes 0. when the weights are nan.
        print("Skipping bad configuration....")
        val_loss = np.inf

      opt_manager.update_score(params, val_loss, model)

      tf.compat.v1.keras.backend.set_session(default_keras_session)

  print("*** Running tests ***")
  tf.compat.v1.reset_default_graph()
  with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    best_params = opt_manager.get_best_params()
    model = ModelClass(best_params, use_cudnn=use_gpu)

    model.load(opt_manager.hyperparam_folder)

    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)
    targets = data_formatter.format_predictions(output_map["targets"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    def extract_numerical_data(data):
      """Strips out forecast time and identifier columns."""
      return data[[
          col for col in data.columns
          if col not in {"forecast_time", "identifier"}
      ]]

    p50_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p50_forecast),
        0.5)
    p90_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p90_forecast),
        0.9)

    tf.compat.v1.keras.backend.set_session(default_keras_session)

  print("Hyperparam optimisation completed @ {}".format(dte.datetime.now()))
  print("Best validation loss = {}".format(val_loss))
  print("Params:")

  for k in best_params:
    print(k, " = ", best_params[k])
  print()
  print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
      p50_loss.mean(), p90_loss.mean()))
  import pickle
  pickle.dump(best_params,open(os.path.join(opt_manager.hyperparam_folder, "best_params_pickle.pkl"),'wb'))


if __name__ == "__main__":

  def get_args():
    """Returns settings from command line."""

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
        "-restart_hyperparam_opt",  
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="yes",
        help="Whether to re-run hyperparameter optimisation from scratch.")
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
#-expt_name kidfail -output_folder /app/tft_outputs -use_gpu yes -restart_hyperparam_opt no -klein yes -num_encoder_steps 56 -n_timesteps_forecasting 20 -timeseries_interval 5 -input_t_dim 120
    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder
    print("args.use_gpu",args.use_gpu)
    print("args.timeseries_interval",args.timeseries_interval)
    print("args.klein",args.klein)
    import sys
    sys.exit()
    return args.expt_name, root_folder, args.use_gpu == 'yes', \
        args.restart_hyperparam_opt == 'yes', args.klein == 'yes',\
        args.num_encoder_steps,args.n_timesteps_forecasting,args.timeseries_interval,args.input_t_dim,

  # Load settings for default experiments
  name, folder, use_tensorflow_with_gpu, restart, klein, num_encoder_steps,n_timesteps_forecasting,timeseries_interval,input_t_dim = get_args()

  
  config = ExperimentConfig(name, folder)
  formatter = config.make_data_formatter()
  formatter = formatter(config.root_folder,
                        klein=klein,
                        num_encoder_steps = num_encoder_steps ,
                        n_timesteps_forecasting = n_timesteps_forecasting,
                        timeseries_interval = timeseries_interval,
                        input_t_dim = input_t_dim)
  # Customise inputs to main() for new datasets.
  main(
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      restart_opt=restart,
      model_folder=os.path.join(config.model_folder, f"main_itd_{input_t_dim}_ntsf{n_timesteps_forecasting}_ti{timeseries_interval}_klein_{klein}"),
      hyperparam_iterations=config.hyperparam_iterations,
      data_csv_path=config.data_csv_path,
      data_formatter=formatter)