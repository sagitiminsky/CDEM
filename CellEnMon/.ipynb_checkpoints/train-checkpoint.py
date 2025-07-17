import math
import os.path
import time
import torch
import pandas as pd
import glob
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import models
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mpl_dates
import config
import torch.nn.functional as F
import numpy as np
from libs.visualize.visualize import Visualizer
plt.switch_backend('agg')  # RuntimeError: main thread is not in main loop

ENABLE_WANDB = True
GROUPS = {
    "DEBUG": {0: "DEBUG"},
    "DYNAMIC_ONLY": {0: "lower metrics", 1: "without RR", 2: "with RR and inv_dist", 3: "with RR only"},
    "Dymanic and Static": {0: "first try", 1: "with RR only", 2: "plot with un-norm. values"},
    "Dymanic and Static 4x1 <-> 1x4": {0: "first try"},
    "Dynamic only": {0: "first try"},
    "Dynamic and Static": {0: "first try", 1:"play around with configurations"},
    "Dynamic and Static Dutch": {0: "first try", 1:"play around with configurations", 2:"real fake gauge metric", 3:"fake gague is not too positioning"},
    "Dynamic and Static Israel": {0: "first try", 1:"play around with configurations"},
    "Dynamic Dutch": {0:"first try"},
    "Real Validation": {0:"last try", 1:"last last try"},
    "Last Try":{0:"last last try"}
}

SELECTED_GROUP_NAME = "Last Try"
SELECT_JOB = 0




DME_KEYS = {1: 'TMmax[dBm]', 2: 'TMmin[dBm]', 3: 'RMmax[dBm]', 4: 'RMmin[dBm]'}
IMS_KEYS = {1: 'RR[mm/h]'}


def toggle(t):
    if t == 'AtoB':
        return 'BtoA'
    else:
        return 'AtoB'


def min_max_inv_transform(x, mmin, mmax):
    return (x+1) * (mmax - mmin) * 0.5 + mmin

# DIRECTIONS
LEFT = (1, 0, 0, 0)
RIGHT = (0, 1, 0, 0)
UP = (0, 0, 1, 0)
DOWN = (0, 0, 0, 1)


def pad_with_respect_to_direction( A, B, dir, value_a, value_b):
    A = F.pad(input=A, pad=dir, mode='constant', value=value_a)
    B = F.pad(input=B, pad=dir, mode='constant', value=value_b)
    return A, B


if __name__ == '__main__':
    real_fake_gauge_metric={}
    dates_range = f"{config.start_date_str_rep_ddmmyyyy}_{config.end_date_str_rep_ddmmyyyy}"
    datetime_format='%Y-%m-%d %H:%M' if config.export_type=="israel" else '%d-%m-%Y %H:%M' # no seconds required
    train_opt = TrainOptions().parse()  # get training options
    validation_opt = TestOptions().parse()
    experiment_name = "only_dynamic" if train_opt.is_only_dynamic else "dynamic_and_static"
    v = Visualizer(experiment_name=experiment_name)
    print("Visualizer Initialized!")
    if ENABLE_WANDB:
        wandb.init(project=train_opt.name, entity='sagitiminsky',
                   group=f"exp_{SELECTED_GROUP_NAME}", job_type=GROUPS[SELECTED_GROUP_NAME][SELECT_JOB])
    print(f'💪Train💪')
    train_dataset = data.create_dataset(train_opt)  # create a train dataset given opt.dataset_mode and other options
    print(f"gauges in train dataset:{train_dataset.dataset.dataset.ims.db.keys()}")
    print(f"links in train dataset:{train_dataset.dataset.dataset.dme.db.keys()}")
    
    print(f'\n\n👀Validation👀')
    validation_dataset = data.create_dataset(
        validation_opt)  # create a train dataset given opt.dataset_mode and other options
    
    print(f"gauges in validation dataset:{validation_dataset.dataset.dataset.ims.db.keys()}")
    print(f"links in validation dataset:{validation_dataset.dataset.dataset.dme.db.keys()}")
    

    model = models.create_model(train_opt)  # create a model given opt.model and other options
    model.setup(train_opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    direction = train_opt.direction
    for epoch in range(train_opt.n_epochs + train_opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        agg_train_mse_A, agg_train_mse_B = 0, 0
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        print(f"Direction:{direction}")
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            
            
            
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(is_train=True)  # calculate loss functions, get gradients, update network weights
            
            # Training losses
            training_losses = model.get_current_losses(is_train=True)
            agg_train_mse_A += training_losses["Train/mse_A"]
            agg_train_mse_B += training_losses["Train/mse_B"]

            t_comp = (time.time() - iter_start_time) / train_opt.batch_size

            if total_iters % train_opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
                # model.save_networks(save_suffix)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, train_opt.n_epochs + train_opt.n_epochs_decay, time.time() - epoch_start_time))

        if epoch % 10 == 0:
            print("Validation in progress...")
            data_B= validation_dataset.dataset.dataset.ims
            data_norm_B = data_B.db_normalized["260"]
            data_A=validation_dataset.dataset.dataset.dme
            validation_gauge_full = torch.Tensor(np.array(list(data_norm_B['data'].values())))
            k=train_opt.slice_dist
            
            for link in data_A.db_normalized:
#                 print(f"links:{validation_dataset.dataset.dataset.dme.db_normalized.keys()}")
                real_fake_gauge_metric[f"{link}-{'260'}"]=0
                seq_len=0
                real_gauge_vec=np.array([])
                fake_gauge_vec=np.array([])
                T=np.array([])
                
                # calculate metric for test gauges
                to_add=0
                
                data_norm_A=data_A.db_normalized[link]
                validation_link_full=torch.Tensor(np.array(list(data_norm_A['data'].values())))
                for i in range(0, len(validation_link_full), k):
                    
                    
                    print(f"link:{link}:{i}/{len(validation_link_full)}")
                    print(f"gauge:{260}:{i}/{len(validation_gauge_full)}")
                    
                    try:
                        A=validation_link_full[i:i+k].reshape(k,4)
                        B=validation_gauge_full[i:i+k].reshape(k,1)
                        rain_slice=B
                    
                    except RuntimeError:
                        continue
                    
                    for a, b in zip(data_norm_A['norm_metadata'], data_norm_B['norm_metadata']):
                        A, B = pad_with_respect_to_direction(A, B, RIGHT, value_a=a, value_b=b)
                        
                    input={"link":link, "attenuation_sample":torch.unsqueeze(A.T,0), "gague":"260", "rain_rate_sample":torch.unsqueeze(B.T,0)}
                    
                                       
                    model.set_input(input,isTrain=False)
                    
#                     print(f"Slected link:{model.link} | Selected gauge:{model.gague}")
#                     print(f"Validation dataset B:{data_B.db_normalized.keys()}")
                    model.optimize_parameters(is_train=False)  # calculate loss functions
                    validation_losses = model.get_current_losses(is_train=False)

                    if ENABLE_WANDB:
                        # Visualize
                        plt.clf()
                        metadata=[0]*4
                        visuals = model.get_current_visuals()
                        fig, axs = plt.subplots(2, 3, figsize=(15, 15))
                        title = f'{link}<->{260}'

                        plt.title(title)

                        for ax, key in zip(axs.flatten(), visuals):

                            if train_opt.is_only_dynamic:
                                N = 4 if 'A' in key else 1
                            else:
                                N = 8 if 'A' in key else 5

                            # Plot Data
                            data = visuals[key][0].reshape(train_opt.slice_dist, N).cpu().detach().numpy() if train_opt.is_only_dynamic else visuals[key][0].T.cpu().detach().numpy()
                            for i in range(1, 5):
                                if 'A' in key:
                                    mmin = model.data_transformation['link']['min'][0].numpy()
                                    mmax = model.data_transformation['link']['max'][0].numpy()
                                    label = DME_KEYS[i]
                                    data_vector = data[:, i - 1]
                                else:
                                    mmin = model.data_transformation['gague']['min'][0].numpy()
                                    mmax = model.data_transformation['gague']['max'][0].numpy()
                                    label = IMS_KEYS[1]
                                    data_vector = data.T[0]


                                metadata_lat_max = float(model.metadata_transformation['metadata_lat_max'])
                                metadata_lat_min = float(model.metadata_transformation['metadata_lat_min'])
                                metadata_long_max = float(model.metadata_transformation['metadata_long_max'])
                                metadata_long_min = float(model.metadata_transformation['metadata_long_min'])

                                metadata_inv_zip = [
                                    (metadata_long_max, metadata_long_min),
                                    (metadata_lat_max, metadata_lat_min),
                                    (metadata_long_max, metadata_long_min),
                                    (metadata_lat_max, metadata_lat_min)
                                ]
                                metadata = ["{:.4f}".format(min_max_inv_transform(x, mmin=mmin, mmax=mmax)) for
                                            (mmin, mmax), x in
                                            zip(metadata_inv_zip, visuals[key][0][:,0][-4:].cpu().detach().numpy())]

                                ax.plot([mpl_dates.date2num(datetime.strptime(t[0], datetime_format)) for t in model.t],
                                        min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax),
                                        marker='o',
                                        linestyle='dashed',
                                        linewidth=0.0,
                                        markersize=4,
                                        label=label
                                        )

                                ax.set_title(key if train_opt.is_only_dynamic else f'{key} \n'
                                                                                   f' {metadata}', y=0.75, fontdict={'fontsize':6})

                            #Formatting Date
                            date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')
                            ax.xaxis.set_major_formatter(date_format)

                            # save in predict directory for generated data
                            start_date = config.start_date_str_rep_ddmmyyyy
                            end_date = config.end_date_str_rep_ddmmyyyy
                            produced_gauge_folder = f'CellEnMon/datasets/dme/{start_date}_{end_date}/predict/{experiment_name}' if 'A' in key else f'CellEnMon/datasets/ims/{start_date}_{end_date}/predict/{experiment_name}'
                            real_gauge_folder=f'CellEnMon/datasets/ims/{start_date}_{end_date}/processed'

                            if not os.path.exists(produced_gauge_folder):
                                os.makedirs(produced_gauge_folder)

                            if not os.path.exists(real_gauge_folder):
                                os.makedirs(real_gauge_folder)

                            if 'fake_B' == key:
    #                             if experiment_name=="only_dynamic":
    #                                 virtual_gauge_lat=f"{float(model.link_center_metadata['latitude']):.3f}"
    #                                 virtual_gauge_long=f"{float(model.link_center_metadata['longitude']):.3f}"
    #                             else:

                                virtual_gauge_lat=data_A.db[link]["metadata"][0]
                                virtual_gauge_long=data_A.db[link]["metadata"][1]

                                file_path = f'{produced_gauge_folder}/{link}-{"260"}_{virtual_gauge_lat}_{virtual_gauge_long}.csv'
#                                 for file_path in glob.glob(f'{produced_gauge_folder}/{link}-{"260"}_*.csv',recursive=True): 
#                                     try:
#                                         os.remove(file_path)
#                                     except OSError:
#                                         print("OSError or file does not exist")

                                with open(file_path, "w") as file:
                                    a=np.array([t[0] for t in model.t]).reshape(train_opt.slice_dist,1)
                                    b=min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax).reshape(train_opt.slice_dist,1)
                                    headers = ','.join(['Time']+list(DME_KEYS.values())) if 'A' in key else ','.join(['Time']+list(IMS_KEYS.values()))
                                    c=np.hstack((a,b))
                                    fmt = ",".join(["%s"]*(c.shape[1]))
                                    np.savetxt(file, c, fmt=fmt, header=headers, comments='')



    #                                 is_virtual_gauge_within_radius_with_link=v.is_within_radius(stations={
    #                                     "fake_longitude":virtual_gauge_long, 
    #                                     "fake_latitude":virtual_gauge_lat,
    #                                     "real_longitude": model.link_center_metadata['longitude'],
    #                                     "real_latitude": model.link_center_metadata['latitude']},
    #                                     radius=config.VALIDATION_RADIUS)
    

                                is_virtual_gauge_within_radius_with_real_gauge = v.is_within_radius(stations={
                                    "fake_longitude":virtual_gauge_long, 
                                    "fake_latitude":virtual_gauge_lat,
                                    "real_longitude":"5.180",
                                    "real_latitude":"52.097"},
                                    radius=config.VALIDATION_RADIUS)

                                if is_virtual_gauge_within_radius_with_real_gauge: #and is_virtual_gauge_within_radius_with_link
                                    print("Link is in range with real gauge...")
                                    path_to_real_gauge=f"{real_gauge_folder}/{'260_5.180_52.097.csv'}"  
                                    real_rain_add=min_max_inv_transform(rain_slice, mmin=mmin, mmax=mmax).cpu().detach().numpy()
                                    fake_rain_add=b
                                
                                    cond=np.array([True if c >= 2 else False for c in abs(real_rain_add-fake_rain_add)])
                                    
                                    
                                    to_add=np.sum((real_rain_add-fake_rain_add)[cond]**2)
                                    
                                    #to_add,seq_len_add,real_vec_add,fake_vec_add,T_add=v.real_and_fake_metric(path_to_real_gauge,file_path)
                            
                                    # print(f"to add:{to_add}")
                                    real_fake_gauge_metric[f"{link}-{'260'}"]+=to_add
                                    seq_len+=len(cond)
                                    real_gauge_vec=np.append(real_gauge_vec,real_rain_add[cond])
                                    fake_gauge_vec=np.append(fake_gauge_vec,fake_rain_add[cond])
                                    T=np.append(T,a[cond])
                                    
                                    
                                    


                    wandb.log({"Images": [wandb.Image(visuals[key], caption=key) for key in visuals]})
                    wandb.log({title: plt})
                    
                #visualization of line plots of fake and real rain


                if f"{link}-{'260'}" in real_fake_gauge_metric and real_fake_gauge_metric[f"{link}-{'260'}"]:
                    # print(f"len T:{len(T)}")
                    # print(f"len real_gauge_vec:{len(real_gauge_vec)}")
                    # print(f"len fake_gauge_vec:{len(fake_gauge_vec)}")
                    wandb.log({f"RMSE-{link}-{'260'}":np.sqrt(real_fake_gauge_metric[f"{link}-{'260'}"]/seq_len)})
                    wandb.log({f"{link}-{'260'}" : wandb.plot.line_series(xs=range(len(T)), ys=[real_gauge_vec, fake_gauge_vec],keys=["real", "fake"],title=f"{link}-{'260'}",xname="Timestamp")})
            

            
            


            
            wandb.log({**validation_losses, **training_losses})      
            path_to_html = f"{v.out_path}/{v.map_name}"
#             v.draw_cml_map()
#             wandb.log({"html": wandb.Html(open(path_to_html), inject=False)})