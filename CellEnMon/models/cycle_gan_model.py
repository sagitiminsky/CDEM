import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from CellEnMon.util.image_pool import SignalPool
from .networks import define_G, define_D, GANLoss
import numpy as np
import os
import config

threshold = float(os.environ["threshold"])
rec_probability_threshold = float(os.environ["rec_probability_threshold"])
fake_probability_threshold = float(os.environ["fake_probability_threshold"])
THETA=float(os.environ["THETA"])


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)
    return _log_cosh(y_pred - y_true)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning CML-to-Gauge translation without paired data.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A||
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B||
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)  -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.epsilon=1e-12
        dataset_type_str="Train" if self.isTrain else "Validation"
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['cycle_A', 'cycle_B', 'mse_A', 'mse_B','bce_rec_B', "idt_A", "idt_B" ] #
        
        if int(os.environ["ENABLE_GAN"]):
            self.loss_names.append('D_A')
            self.loss_names.append('D_B')
            self.loss_names.append('G_A')
            self.loss_names.append('G_B_only')

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', "fake_B_det", "rec_B_det"]
        visual_names_B = ['real_B', 'fake_A', 'rec_B',"fake_B_det","rec_B_det"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = define_G(opt.input_nc_A, opt.output_nc_A, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,direction="AtoB")
        self.netG_B = define_G(opt.input_nc_B, opt.output_nc_B, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, direction="BtoA")

        if self.isTrain:  # define discriminators
            self.netD_A = define_D(opt.input_nc_A, opt.ndf, opt.netD,
                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = define_D(opt.input_nc_B, opt.ndf, opt.netD,
                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = SignalPool(opt.pool_size)  # create signal buffer to store previously generated signals
            self.fake_B_pool = SignalPool(opt.pool_size)  # create signal buffer to store previously generated signals
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss(reduction='none')
            self.criterionIdt = torch.nn.L1Loss()
            self.mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def weight_func(self, x, a):
        return 1/(a * torch.exp(-x*a))

    def set_input(self, input, epoch, isTrain=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.slice_dist=input['slice_dist']
        self.noise = torch.rand(self.slice_dist, device="cuda:0") * 0.1
        self.real_A = input['A' if AtoB else 'B'].to(self.device) if isTrain else input["attenuation_sample" if AtoB else 'rain_rate_sample'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device) if isTrain else input['rain_rate_sample' if AtoB else 'attenuation_sample'].to(self.device)
        self.gague = input['gague']
        self.link = input['link']
        self.t = input['Time']
        self.dataset_type="Train" if isTrain else "Validation"
        self.isTrain=isTrain
        self.rain_rate_prob = input['rain_rate_prob'].to(self.device)
        L=input['distance'].to(self.device)
        self.L=L+self.epsilon
        self.dist_func=1/(torch.log(1+(self.L/config.TRAIN_RADIUS)))
        self.epoch=epoch
        
        
        if isTrain:
            self.alpha=0.02
            self.metadata_A = input['metadata_A' if AtoB else 'metadata_B'].to(self.device)
            self.metadata_B = input['metadata_B' if AtoB else 'metadata_A'].to(self.device)
            self.attenuation_prob = input['attenuation_prob'].to(self.device)

            
            self.link_norm_metadata=input['link_norm_metadata']
            self.link_metadata=input['link_metadata']
            self.link_full_name=input['link_full_name'][0]
            self.link_center_metadata=input['link_center_metadata']


            self.gague_norm_metadata=input['gague_norm_metadata']
            self.gague_metadata=input['gague_metadata']
            self.gague_full_name=input['gague_full_name'][0]

            self.data_transformation = input['data_transformation']
            self.metadata_transformation = input['metadata_transformation']

    def dynamic_norm_zero_one(self,x, db_type): #
        epsilon=1e-6
        min_val = torch.min(x)
        max_val = torch.max(x)

        global_min = -50.8 if db_type=="dme" else 0
        global_max = 17 if db_type=="dme" else 3.3

        
        # Clipping min_val and max_val
        min_val = min_val if min_val < global_min else global_min
        max_val = max_val if max_val > global_max else global_max

        return (x - min_val) / (max_val - min_val + epsilon)

    def norm_zero_one(self,x):
        epsilon=1e-6
        min_val = torch.min(x)
        max_val = torch.max(x)

        return (x - min_val) / (max_val - min_val + epsilon)


    def logistic_cdf(self, x):
        return 1 / (1 + torch.exp(-x / 5))

    def norm_mean_std(self,x):
        return (x-x.mean())/(x.std()+self.epsilon)

        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""          
        activation = nn.ReLU() #nn.Identity() #nn.ReLU()
        
        ##############
        ## >> Fake ###
        ##############
        fake_B = self.netG_A(self.real_A, dir="AtoB")   # G_A(A)

        
        self.fake_B_det = torch.sigmoid(fake_B[1]) # <--Detection

        ## >> Regression
            # >> A
        self.fake_A = self.netG_B(self.real_B, dir="BtoA")  ## <<-- regression

            # >> B
        self.fake_B=activation(fake_B[0]) ## <<-- regression
        self.fake_B_dot_detection = self.fake_B * (self.fake_B_det >= fake_probability_threshold)

        ############
        ## >> Rec ##
        ############
        rec_B=self.netG_A(self.fake_A,dir="AtoB")

        self.rec_B_det_without_activation = rec_B[1]
        self.rec_B_det = torch.sigmoid(rec_B[1]) ### <-- detection
            
        # >> A
        self.rec_A = self.netG_B(self.fake_B_dot_detection, dir="BtoA")  ## <<-- regression

        # >> B
            ## >> rec Detection
        self.rec_B = activation(rec_B[0]) ## <<-- regression
        self.rec_B_dot_detection = self.rec_B * (self.rec_B_det >= rec_probability_threshold)


    def backward_D_basic(self, netD, real, fake, weight=1): #weight=torch.ones([1], device='cuda:0')
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        L2=nn.MSELoss(reduction='none')
        # Real
        pred_real = netD(real)
        target = torch.full_like(pred_real, 1.0).to(pred_real.device)
        loss_D_real = torch.mean(L2(pred_real, target))
        # Fake
        pred_fake = netD(fake.detach())
        target = torch.full_like(pred_fake, 0).to(pred_fake.device)
        loss_D_fake = torch.mean(L2(pred_fake, target))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)
        if self.isTrain:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = 10 * self.backward_D_basic(self.netD_A, self.real_A, self.fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = 10 * self.backward_D_basic(self.netD_B, self.real_B, self.fake_B_dot_detection) # self.fake_B_dot_detection

    def backward_G(self):
        """Calculate the losses"""



        if "DEBUG" in os.environ and int(os.environ["DEBUG"])==1:
            print(f"real A:{self.real_A.shape}")
            print(f"real B:{self.real_B.shape}")
            print(f"fake A:{self.fake_A.shape}")
            print(f"fake B:{self.fake_B.shape}")
            print(f"rec A: {self.rec_A.shape}")
            print(f"rec B:{self.rec_B.shape}")
            print(f"D(fake_B): {self.netD_B(self.fake_B).shape}")
            print(f"D(fake_A): {self.netD_A(self.fake_A).shape}")

            print(f"self.fake_B_det:{self.fake_B_det}")
            print(f"rr_prob: {self.rain_rate_prob.shape}")
            print(f"fake_B: {self.fake_B.shape}")
            print(f"fake_A: {self.fake_A.shape}")

            print(f"fake_B * rr_prob: {(self.fake_B * self.rain_rate_prob).shape}")
            print(f"rec_B * rr_prob: {(self.rec_B * self.rain_rate_prob).shape}")
            assert(False)

        lambda_A = 10
        lambda_B = 10
        #######################
        #### Identity loss ####
        #######################
        L1_idt=nn.L1Loss(reduction='none')
        self.loss_idt_A = torch.sum(L1_idt(self.fake_A, self.real_A))
        self.loss_idt_B = torch.sum(L1_idt(self.fake_B, self.real_B)) #* self.rain_rate_prob

        rec_bce_weight_loss = nn.BCEWithLogitsLoss(weight=self.rain_rate_prob) #  
        BCE = nn.BCELoss(weight=self.rain_rate_prob)

        targets=(self.real_B >= threshold).float()
        
        gamma = 0.1 if config.export_type=='israel' else 0.01

        # BCE for detector
        self.loss_bce_rec_B  = gamma * BCE(self.rec_B_det, targets)

        # if self.loss_bce_rec_B > 1000:
            # print(f"rec_B_det: {self.rec_B_det}")
            # print(f"targets: {targets}")
            # print(f"rain_rate_prob: {self.rain_rate_prob}")
            # Make sure to set LAMBDA properly,
            # a good place to start Israel: 1 | Ducth: 0.1



        L1=nn.L1Loss(reduction='none')
        L2=nn.MSELoss(reduction='none')


        # Backward cycle loss
        alpha_A=1 if config.export_type=='israel' else 100
        alpha_B=1000 if config.export_type=='israel' else 1

        self.loss_cycle_A = alpha_A * torch.mean(L2(self.rec_A, self.real_A))
        self.loss_cycle_B = alpha_B * torch.mean(L2(self.rec_B, self.real_B))

        # gamma=2        
        # residual = torch.abs(self.rec_B - self.real_B)  # L1 loss
        # modulating_factor = (1 - torch.exp(-residual)) ** gamma # Modulating factor
        # self.loss_cycle_B = torch.mean(modulating_factor * self.rain_rate_prob * residual)
        
        beta_A = 1 if config.export_type=="israel" else 10
        beta_B = 1 if config.export_type=="israel" else 10
        

        #######################
        ####### GAN loss ######
        #######################
        # GAN loss D_B(G_A(A))
        self.D_B=self.netD_B(self.fake_B_dot_detection) # self.fake_B_dot_detection
        targets = torch.full_like(self.D_B, 1.0).to(self.D_B.device)
        self.loss_G_B_only = beta_B * torch.mean(L2(self.D_B, targets))

        # GAN loss D_A(G_B(B))
        self.D_A=self.netD_A(self.fake_A)
        targets = torch.full_like(self.D_A, 1.0).to(self.D_A.device)
        self.loss_G_A = beta_A * torch.mean(L2(self.D_A, targets)) #weight=self.rr_norm.max(), weight=self.att_norm.mean()
        


        self.loss_mse_A = torch.mean(L2(self.fake_A, self.real_A))
        self.loss_mse_B = torch.mean(L2(self.fake_B, self.real_B))

        self.loss_G = \
            (     
                self.loss_cycle_A +\
                self.loss_cycle_B +\

                self.loss_bce_rec_B
            )

        GAN_LOSS=0
        if int(os.environ["ENABLE_GAN"]):
            ## >> As per original paper, cycle loss needs to be x10 the loss of the GAN
            GAN_LOSS =\
            (
                self.loss_G_B_only +\
                self.loss_G_A
            )

        self.loss_G = self.loss_G + GAN_LOSS

        if self.isTrain:
            self.loss_G.backward()

   
    
    def min_max_inv_transform(self,x, mmin, mmax):
        return x # x * (mmax - mmin) + mmin
    
    def optimize_parameters(self, is_train=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        
        dataset_type_str="Train" if is_train else "Validation"
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        
        
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        if self.isTrain:
            self.optimizer_G.step()  # update G_A and G_B's weights
        
        # # D_A and D_B
        # ## resetting attrs ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'mse_A', 'mse_B', 'bce_B','G_B_only']


        if int(os.environ["ENABLE_GAN"]):
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero        
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            if self.isTrain:
                self.optimizer_D.step()  # update D_A and D_B's weights
            
            setattr(self,f"loss_{self.dataset_type}_G_B_only",self.loss_G_B_only)
            setattr(self,f"loss_{self.dataset_type}_D_A",self.loss_D_A)
            setattr(self,f"loss_{self.dataset_type}_G_A",self.loss_G_A)
            setattr(self,f"loss_{self.dataset_type}_D_B",self.loss_D_B)

        setattr(self,f"loss_{self.dataset_type}_cycle_A",self.loss_cycle_A)
        setattr(self,f"loss_{self.dataset_type}_cycle_B",self.loss_cycle_B)
        setattr(self,f"loss_{self.dataset_type}_mse_A",self.loss_mse_A)
        setattr(self,f"loss_{self.dataset_type}_mse_B",self.loss_mse_B)
        setattr(self,f"loss_{self.dataset_type}_bce_rec_B",self.loss_bce_rec_B)
        setattr(self,f"loss_{self.dataset_type}_idt_A",self.loss_idt_A)
        setattr(self,f"loss_{self.dataset_type}_idt_B",self.loss_idt_B)