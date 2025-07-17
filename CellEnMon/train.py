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
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend 'Agg'
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mpl_dates
import config
import torch.nn.functional as F
import numpy as np
from libs.visualize.visualize import Visualizer
from preprocess import Preprocess
from collections import OrderedDict
plt.switch_backend('agg')  # RuntimeError: main thread is not in main loop
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from matplotlib import colors
from collections import defaultdict
import warnings

# Ignore a specific type of warning from a specific module
warnings.filterwarnings("ignore", message="'linear' x-axis tick spacing not even, ignoring mpl tick formatting.", module="plotly.matplotlylib")

# Alternatively, ignore all warnings from a specific module
warnings.filterwarnings("ignore", module="plotly.matplotlylib")

ENABLE_WANDB = bool(os.environ["ENABLE_WANDB"])
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
    "Last Try":{0:"last last try"},
    "Frontiers":{0:"first try", 1:"start training on Israel data"},
    "Lahav":{0:"GAN+Cycle", 1:"Cycle", 2:"Cycle+Detector"},
    "Neot Smadar":{0:"GAN+Cycle",1:"Cycle", 2:"Cycle+Detector"},
    "1L1G": {0:"Cycle+Detector"},
    #...
    "9L6G": {0:"Cycle+Detector"},

}




# gauges in train dataset:dict_keys(['ZOMET HANEGEV', 'BEER SHEVA', 'EZUZ', 'NEOT SMADAR', 'SEDE BOQER', 'PARAN'])
# links in train dataset:dict_keys(['a477-b379', 'a459-803b', '462d-c088', 'a479-b477', 'b465-d481', 'b451-a350', 'b459-a690', 'b480-a458', 'f350-e483'])

# gauges in validation dataset:dict_keys(['NEVATIM', 'LAHAV'])
# links in validation dataset:dict_keys(['c078-d088', 'a473-b119', 'b394-ts04'])


# Validation Matching
if config.export_type == "dutch":
    validation_link_to_gauge_matching ={
        "rrFO-WPMD":[],
        "QLPN-NBOl": [], #"260"
        "hOQe-gKVi": ["348"] #"348"


    }    
else:
    validation_link_to_gauge_matching ={
    #     "c078-d088": [], 
    #     "a473-b119": [], 
        "b394-ts04": [], #LAHAV
        "b459-a690": [], #"NEOT SMADAR",
        "c409-d063": [],
        "j033-261c": ["NIZZAN"]

    }


### TO RUN:
# ENABLE_GAN=1 THETA=1 LAMBDA=2.5 SELECTED_GROUP_NAME="Lahav" SELECT_JOB=2 ENABLE_WANDB=True DEBUG=0 threshold=0.3 probability_threshold=0.25 python3 CellEnMon/train.py
### >> LAMBDA
#1) first wascalculated by evaluation func_fit for train dataset with function a^e(-ax) --> a=LAMBDA
#2) cycle_A and cycle_B should be the same scale - mind the training/validation losses (!!!)
# So to make sure that the cycle_A and cycle_B are in the same scale select LAMBDA=5

# Environment Variables
threshold = float(os.environ["threshold"])
rec_probability_threshold = float(os.environ["rec_probability_threshold"]) #0.3 # a*e^(-bx)+c, ie. we consider a wet event over x=0.2 mm/h
fake_probability_threshold = float(os.environ["fake_probability_threshold"])
SELECTED_GROUP_NAME = os.environ["SELECTED_GROUP_NAME"]
SELECT_JOB = int(os.environ["SELECT_JOB"])
LAMBDA=float(os.environ["LAMBDA"])

# see: __getitem__ in cellenmon_dataset - We randomize the pair and the time
os.environ["NUMBER_OF_CML_GAUGE_RANDOM_SELECTIONS_IN_EACH_EPOCH"]="100"
ITERS_BETWEEN_VALIDATIONS=10

DME_KEYS = {0: 'PowerTLTMmax[dBm]', 1: 'PowerTLTMmin[dBm]', 2: 'PowerRLTMmax[dBm]', 3: 'PowerRLTMmin[dBm]'}
IMS_KEYS = {0: 'RainAmout[mm/h]'}


def toggle(t):
    if t == 'AtoB':
        return 'BtoA'
    else:
        return 'AtoB'


def min_max_inv_transform(x, mmin, mmax):
    return x #x * (mmax - mmin) + mmin

# DIRECTIONS
LEFT = (1, 0, 0, 0)
RIGHT = (0, 1, 0, 0)
UP = (0, 0, 1, 0)
DOWN = (0, 0, 0, 1)



def pad_with_respect_to_direction( A, B, dir, value_a, value_b):
    A = F.pad(input=A, pad=dir, mode='constant', value=value_a)
    B = F.pad(input=B, pad=dir, mode='constant', value=value_b)
    return A, B

NUMBER_OF_CML_GAUGE_RANDOM_SELECTIONS_IN_EACH_EPOCH = int(os.environ["NUMBER_OF_CML_GAUGE_RANDOM_SELECTIONS_IN_EACH_EPOCH"])
if __name__ == '__main__':
    real_fake_gauge_metric={}
    dates_range = f"{config.start_date_str_rep_ddmmyyyy}_{config.end_date_str_rep_ddmmyyyy}"
    datetime_format='%Y-%m-%d %H:%M'
    date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M')

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
        validation_opt)  # create a validation dataset given opt.dataset_mode and other options
    
    print(f"gauges in validation dataset:{validation_dataset.dataset.dataset.ims.db.keys()}")
    # print(f"norm min-max:{validation_dataset.dataset.dataset.data_dict_B['data_min']}-{validation_dataset.dataset.dataset.data_dict_B['data_max']}")

    print(f"links in validation dataset:{validation_dataset.dataset.dataset.dme.db.keys()}")
    # print(f"norm min-max:{train_dataset.data_dict_B['data_min']}-{train_dataset.data_dict_B['data_max']}")

    model = models.create_model(train_opt)  # create a model given opt.model and other options
    model.setup(train_opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    
    
    for epoch in range(train_opt.n_epochs + train_opt.n_epochs_decay):
        training_losses=defaultdict(float)
#         direction = "AtoB" if (epoch // 10) % 2 == 0 else "BtoA"
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        agg_train_mse_A, agg_train_mse_B, agg_train_bce_B = 0, 0, 0
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

#         print(f"Direction:{direction}")
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            
            #model.train()
            model.set_input(data, epoch)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(is_train=True)  # calculate loss functions, get gradients, update network weights
            
            # Training losses
            

            t_comp = (time.time() - iter_start_time) / train_opt.batch_size

            iter_data_time = time.time()
        
            current_losses=model.get_current_losses(is_train=True)
            for key in current_losses:
                training_losses[key] += current_losses[key]
        


            
            
            
        if epoch % ITERS_BETWEEN_VALIDATIONS == 0 and epoch>0: # VALIDATION

            print(f'End of training epoch:{epoch} | Remember! in each epoch we trained on {NUMBER_OF_CML_GAUGE_RANDOM_SELECTIONS_IN_EACH_EPOCH} randomly selected CML-Gauge Pairs')
            print(f"Validation in progress...")
            data_A=validation_dataset.dataset.dataset.dme
            data_B= validation_dataset.dataset.dataset.ims
            k=train_opt.slice_dist
            validation_links=data_A.db_normalized.keys()
            rain_fig, rain_axs = plt.subplots(2, len(validation_links)//2+1, figsize=(15, 15))
            rain_axs=rain_axs.flatten()
            
            
            for link_counter,link in enumerate(validation_links):
                for gauge in validation_link_to_gauge_matching[link]:
                    data_norm_B = data_B.db_normalized[gauge]
                    validation_gauge_full = torch.Tensor(np.array(list(data_norm_B['data'].values())))
                    real_fake_gauge_metric[f"{link}-{gauge}"]=0
                    seq_len=0
                    real_gauge_vec=np.array([])
                    fake_gauge_vec=np.array([])
                    rec_gauge_vec=np.array([])
                    fake_gauge_vec_det=np.array([])
                    rec_gauge_vec_det=np.array([])
                    T=np.array([])

                    # calculate metric for test gauges
                    to_add=0
                    A_delay=0
                    B_delay=0
                    num_samples=len(validation_gauge_full)
                    data_norm_A=data_A.db_normalized[link]
                    validation_link_full=torch.Tensor(np.array(list(data_norm_A['data'].values())))
#                     model.setup(validation_opt,isTrain=False)
                    
                    for batch_counter,i in enumerate(range(0, num_samples, k)): #len(validation_gauge_full)


                        try:
                            A=validation_link_full[i :  i + k].reshape(k,4)
                            B=validation_gauge_full[i : i + k].reshape(k,1)
                            
                            slice_time=data_norm_B['time'][i: i + k]
                    
                        except RuntimeError:
                            break
                        

                        attenuation_sample=torch.unsqueeze(A.T,0)
                        attenuation_sample_unnormalized = attenuation_sample
                        rain_sample=torch.unsqueeze(B.T,0)
                        rain_sample_unnormalized = rain_sample
                        loader={"link":link, "attenuation_sample":attenuation_sample,\
                         "gague":gauge,\
                         "rain_rate_sample":rain_sample,\
                         "Time": slice_time,\
                         "rain_rate_prob": config.func_fit(rain_sample_unnormalized, LAMBDA),
                         "distance": torch.tensor([3], device='cuda:0', dtype=torch.float64), # in KM
                         "slice_dist": train_opt.slice_dist
                        }                      
                        model.set_input(loader,epoch,isTrain=False)
                            
                        model.optimize_parameters(is_train=False)
                        visuals = model.get_current_visuals()
                        validation_losses = model.get_current_losses(is_train=False) # validation for each batch, i.e slice_dist samples


                        if ENABLE_WANDB:
                            # Visualize
                            for key, value in list(validation_losses.items()):
                                validation_losses[f"#{batch_counter}-{key}"] = value
                            wandb.log({**validation_losses})

                            metadata=[0]*4

            
                            with torch.no_grad():
                                visuals = OrderedDict(
                                [
                                    ('real_A', torch.unsqueeze(A.T,0)),\
                                    ('fake_B', model.fake_B),\
                                    ('rec_A', model.rec_A),\
                                    ('real_B',torch.unsqueeze(B.T,0)),\
                                    ('fake_A', model.fake_A),\
                                    ('rec_B', model.rec_B),\
                                    ('fake_B_det', model.fake_B_det),\
                                    ('rec_B_det', model.rec_B_det)
                                ]
                            )
                            

                            real_rain_add=np.round(visuals['real_B'][0].cpu().detach().numpy(), 2)
                            fake_rain_add=np.round(visuals['fake_B'][0].cpu().detach().numpy(), 2)
                            rec_rain_add=np.round(visuals['rec_B'][0].cpu().detach().numpy(), 2)
                            fake_detection_add=np.round(visuals['fake_B_det'][0].cpu().detach().numpy(), 2)
                            rec_detection_add=np.round(visuals['rec_B_det'][0].cpu().detach().numpy(), 2)
                            
                            real_gauge_vec=np.append(real_gauge_vec,real_rain_add)
                            fake_gauge_vec=np.append(fake_gauge_vec,fake_rain_add)
                            rec_gauge_vec=np.append(rec_gauge_vec,rec_rain_add)
                            fake_gauge_vec_det=np.append(fake_gauge_vec_det,fake_detection_add)
                            rec_gauge_vec_det=np.append(rec_gauge_vec_det, rec_detection_add)
                            T=np.append(T,slice_time)

                            # rec_A=visuals['rec_A'][0].cpu().detach().numpy()
                            # rec_A_unnorm=min_max_inv_transform(rec_A, mmin=-50.8, mmax=17)
                            # mmin_rec_A=np.min(rec_A_unnorm)
                            # mmax_rec_A=np.max(rec_A_unnorm)

                            # rec_B=visuals['rec_B'][0].cpu().detach().numpy()
                            # rec_B_unnorm=min_max_inv_transform(rec_B, mmin=0, mmax=3.3)
                            # mmin_rec_B=np.min(rec_B_unnorm)
                            # mmax_rec_B=np.max(rec_B_unnorm)


                            fig, axs = plt.subplots(2, 3, figsize=(15, 15))
                            title = f'{batch_counter}:{link}<->{gauge}'
                            
                            for ax, key in zip(axs.flatten(), visuals):
                                N = 4 if 'A' in key else 1

                                # Plot Data
                                data = visuals[key][0].cpu().detach().numpy()
                                assert(data.shape == (N,train_opt.slice_dist))
                                
                            
                                for i in range(4): #This is only validaiton
                                    if 'A' in key:
                                        # Normalization with global values will only work \
                                        # when using Relu or logistic_cdf
                                        # Linear normalization is saturating values
                                        # Relu does not normalize between 0 and 1
                                        mmin = -50.8 #mmin_rec_A if 'fake' in key else -50.8
                                        mmax = 17 #mmax_rec_A if 'fake' in key else 17
                                        label = DME_KEYS[i]
                                        data_vector = torch.tensor(data[i])
                                        
                                    else:
                                        mmin = 0 
                                        mmax = 3.3 #mmax_rec_B if 'fake' in key else 3.3
                                        mmin_B=mmin
                                        mmax_B=mmax
                                        label = IMS_KEYS[0]
                                        data_vector = torch.tensor(data[0])


                                    
                                    if key!="fake_B" and key!="rec_B":
                                        ax.plot(mpl_dates.date2num(slice_time),
                                                data_vector,
                                                marker='o',
                                                linestyle='dashed',
                                                linewidth=0.0,
                                                markersize=4,
                                                label=label
                                        )
                                    else:
                                        
                                        if key=="fake_B":
                                            mask=fake_detection_add[0]
                                            probability_threshold=fake_probability_threshold

                                        else:
                                            mask=rec_detection_add[0]
                                            probability_threshold=rec_probability_threshold

                                        
                                        mask=(mask >= probability_threshold).astype(int)
                                        cmap=["red" if m else "black" for m in mask]
                                        
                                        ax.scatter(mpl_dates.date2num(slice_time),
                                                data_vector,
                                                marker='o',
                                                linestyle='dashed',
                                                linewidth=0.0,
                                                c=cmap,
                                                label="RainAmout[mm/h]",
                                        )
                                        


                                    ax.set_title(key if train_opt.is_only_dynamic else f'{key} \n'
                                                                                       f' {metadata}', y=0.75, fontdict={'fontsize':6})

                                    
                                ax.xaxis.set_major_formatter(date_format)
                            




                        wandb.log({title: fig})
                        # with np.printoptions(threshold=np.inf):
                        #     print(f"Fake | batch #{batch_counter}:{np.round(fake_detection_add, 2)}")
                        #     print(f"Rec | batch#{batch_counter}:{np.round(rec_detection_add, 2)}")
                    

                    
                    
                    # Convert continuous values to binary class labels
                    real_gauge_vec_labels = (real_gauge_vec >= threshold).astype(int)
                    rec_gauge_vec_det_labels = ((rec_gauge_vec_det >= rec_probability_threshold)).astype(int)
                    fake_gauge_vec_det_labels = ((fake_gauge_vec_det >= fake_probability_threshold)).astype(int)
                    

                    p=Preprocess(link=link,gauge=gauge,epoch=epoch, T=T,\
                        real=real_gauge_vec,
                        fake=fake_gauge_vec,\
                        rec=rec_gauge_vec,\
                        fake_detections = fake_gauge_vec_det_labels,\
                        rec_detections = rec_gauge_vec_det_labels
                    )
                    
                    

                    
                    CM_rec=confusion_matrix(real_gauge_vec_labels, p.rec_det)
                    CM_fake=confusion_matrix(real_gauge_vec_labels, p.fake_det)
                    
                    # Create subplots for given confusion matrices
                    f, axes = plt.subplots(1, 1, figsize=(15, 15))

                    # Plot the first confusion matrix at position (0)
                    axes.set_title("Cycle", size=8)
                    # axes[1].set_title("Fake", size=8)

                    ConfusionMatrixDisplay(confusion_matrix=CM_rec, display_labels=["dry","wet"]).plot(
                        include_values=True, cmap="Blues", ax=axes, colorbar=False, values_format=".0f")
                    
                    # ConfusionMatrixDisplay(confusion_matrix=CM_fake, display_labels=["dry","wet"]).plot(
                    #     include_values=True, cmap="Blues", ax=axes[1], colorbar=False, values_format=".0f")

                    # Remove x-axis labels and ticks
                    axes.xaxis.set_ticklabels(['dry', 'wet'])
                    axes.yaxis.set_ticklabels(['dry', 'wet'])
                    # axes[1].xaxis.set_ticklabels(['dry', 'wet'])
                    # axes[1].yaxis.set_ticklabels(['dry', 'wet'])

                    
                    wandb.log({f"Confusion Matrices":f})
                    wandb.log({"f1-score real Gauges<->Cycle Gauges": f1_score(p.rec_det, real_gauge_vec_labels)})
                    # wandb.log({"f1-score real Gauges<->Fake Gauges": f1_score(p.fake_det, real_gauge_vec_labels)})
                    wandb.log({"Acc real Gauges<->Cycle Gauges": (CM_rec[0][0]+CM_rec[1][1])/(CM_rec[0][0]+CM_rec[0][1]+CM_rec[1][0]+CM_rec[1][1])})
                    # wandb.log({"Acc real Gauges<->Fake Gauges": (CM_fake[0][0]+CM_fake[1][1])/(CM_fake[0][0]+CM_fake[0][1]+CM_fake[1][0]+CM_fake[1][1])})
                    

        



                    preprocessed_time_wanb=np.array(mpl_dates.date2num(T))

                    
                    fig_preprocessed, axs_preprocessed = plt.subplots(1, 1, figsize=(15, 15))
                    
                    # >> REC
                    # axs_preprocessed.plot(preprocessed_time_wanb, p.fake_cumsum, 'r:' ,label="FAKE")
                    axs_preprocessed.plot(preprocessed_time_wanb, p.fake_dot_det_cumsum, 'm:', label="FAKE+Det")
                    
                    
                    # >> FAKE
                    # axs_preprocessed.plot(preprocessed_time_wanb, p.rec_cumsum, 'b-', label="REC")
                    axs_preprocessed.plot(preprocessed_time_wanb, p.rec_dot_det_cumsum, 'g-', label="REC+Det")
                    

                    # >> GT
                    axs_preprocessed.plot(preprocessed_time_wanb, p.real_cumsum, "--", label="GT", color='orange')
                    axs_preprocessed.grid()
                    fig_preprocessed.legend()
                    fig_preprocessed.tight_layout()


                    # Specify the number of ticks you want on the x-axis
                    num_ticks = 10

                    # Calculate the step size between ticks
                    step_size = len(preprocessed_time_wanb) // num_ticks

                    # Set the ticks on the x-axis
                    # axs_preprocessed.set_xticks(np.array(preprocessed_time_wanb[::step_size]))  # Setting x-ticks
                    # axs_preprocessed.set_xticklabels(preprocessed_time_wanb[::step_size], rotation=45)  # Setting x-tick labels with rotation
                    axs_preprocessed.xaxis.set_major_formatter(date_format)
                    
                        
                    wandb.log({f"Virtual (CML) vs Real (Gauge) - {link}-{gauge}":fig_preprocessed})
                    
                    #RMSSE
                    cond=[True if r >= threshold or f >= threshold else False for r,f in zip(p.real, p.rec)]
                    N=len(p.fake)
                    # wandb.log({f"RMSSE-REG-{link}-{gauge}":np.sqrt(np.sum((p.real - p.fake)**2)/N)})
                    wandb.log({f"RMSSE-FAKE+DET-{link}-{gauge}":np.sqrt(np.sum((p.real - p.fake_dot_det)**2)/N)})
                    wandb.log({f"RMSSE-REC+DET-{link}-{gauge}":np.sqrt(np.sum((p.real - p.rec_dot_det)**2)/N)})
        
                            
                    assert(len(T)==len(real_gauge_vec.flatten()))
                    assert(len(T)==len(fake_gauge_vec.flatten()))
                    

                
            if ENABLE_WANDB:
                for key in current_losses:
                    training_losses[key] = training_losses[key]/(ITERS_BETWEEN_VALIDATIONS*len(train_dataset))
            
                wandb.log({**training_losses})
                path_to_html = f"{v.out_path}/{v.map_name}"
#                 v.draw_cml_map()
#                 wandb.log({"html": wandb.Html(open(path_to_html), inject=False)})
            print(print(f"Validation cycle end..."))

        model.save_networks("latest")