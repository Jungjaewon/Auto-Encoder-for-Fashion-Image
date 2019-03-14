from model import VarEncoder
from model import Encoder
from torchvision import transforms as T
from torchvision.utils import save_image
from reporting import Reporting
import torch
import torch.nn as nn
import os
import time
import datetime


class Trainer(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.image_size = config.image_size
        self.encoder_mode = config.encoder_mode
        self.encoder_last = config.encoder_last
        self.encoder_start_ch = config.encoder_start_ch
        self.encoder_target_ch = config.encoder_target_ch
        self.image_layer = config.image_layer

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.e_lr = config.e_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:' + config.gpu)
        self.email_address = config.email_address
        self.image_sending = config.image_sending


        # Directories.
        self.train_dir = config.train_dir
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.model_save_start = config.model_save_start
        self.lr_update_step = config.lr_update_step

        # Reporting
        self.reporting = Reporting(self.email_address, self.image_sending, os.path.join(self.train_dir,self.sample_dir))

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        if self.encoder_mode == 0:
            self.model = Encoder(self.encoder_start_ch,self.encoder_target_ch,last_layer=self.encoder_last,image_layer=self.image_layer)
            self.optimizer = torch.optim.Adam(self.model.parameters(),self.e_lr,[self.beta1,self.beta2])
            self.print_network(self.model, 'E')
        elif self.encoder_mode == 1:
            self.model = VarEncoder(self.encoder_start_ch, self.encoder_target_ch, image_layer=self.image_layer)
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.e_lr, [self.beta1, self.beta2])
            self.print_network(self.model, 'VE')
        else:
            raise Exception("encoder_mode error!!")

        self.model.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))

        if self.encoder_mode == 0:
            model_path = os.path.join(self.train_dir, self.model_save_dir,'{}-E.ckpt'.format(resume_iters))
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
        elif self.encoder_mode == 1:
            model_path = os.path.join(self.train_dir, self.model_save_dir, '{}-VE.ckpt'.format(resume_iters))
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            raise Exception("encoder_mode error!!")


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, e_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = e_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, fixedTargetImages, = next(data_iter)

        fixedTargetImages = fixedTargetImages.to(self.device)

        # Learning rate cache for decaying.
        e_lr = self.e_lr
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images
            try:
                _, targetImages = next(data_iter)
            except:
                data_iter = iter(data_loader)
                _, targetImages = next(data_iter)

            targetImages = targetImages.to(self.device)
            loss = {}

            # =================================================================================== #
            #                             2. Training the model                                   #
            # =================================================================================== #

            if self.encoder_mode == 0:
                restoredImages = self.model(targetImages)
                loss_recons = torch.mean(torch.abs(restoredImages - targetImages))

                self.reset_grad()
                loss_recons.backward()
                self.optimizer.step()

                if torch.isnan(loss_recons):
                    raise Exception('loss_recons is nan')
            elif self.encoder_mode == 1:

                restoredImages, mu, logvar = self.model(targetImages)
                loss_recons = torch.mean(torch.abs(restoredImages - targetImages))
                loss_dist = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss_total = loss_recons + loss_dist
                self.reset_grad()
                loss_total.backward()
                self.optimizer.step()

            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_list = list()
                    if self.encoder_mode == 0:
                        fixedRestoredimage = self.model(fixedTargetImages)
                    elif self.encoder_mode == 1:
                        fixedRestoredimage, _, _ = self.model(fixedTargetImages)
                    image_list.append(fixedRestoredimage)
                    image_list.append(fixedTargetImages)
                    result_image = torch.cat(image_list,dim=3)
                    sample_path = os.path.join(self.train_dir,self.sample_dir, '{}-images.jpg'.format(i + 1))
                    if self.image_layer == 'tanh':
                        save_image(self.denorm(result_image.data.cpu()), sample_path, nrow=1, padding=0)
                    elif self.image_layer == 'sigmoid':
                        save_image(result_image.data.cpu(), sample_path, nrow=1, padding=0)
                    else:
                        raise Exception('Please check the gen_last layer mode')

                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if ((i + 1) >= self.model_save_start) and ((i + 1) % self.model_save_step == 0):
                E_path = os.path.join(self.train_dir, self.model_save_dir, '{}-E.ckpt'.format(i + 1))
                torch.save(self.model.state_dict(), E_path)
                print('Saved model checkpoints into {}...'.format(os.path.join(self.train_dir,self.model_save_dir)))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                e_lr -= (self.e_lr / float(self.num_iters_decay))
                self.update_lr(e_lr)
                print ('Decayed learning rates, e_lr: {}.'.format(e_lr))

        #self.reporting.send_mail()

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.data_loader

        with torch.no_grad():
            for _, targetImages in enumerate(data_loader):
                image_list = list()
                if self.encoder_mode == 0:
                    restoredimage = self.model(targetImages)
                elif self.encoder_mode == 1:
                    restoredimage, _, _ = self.model(targetImages)
                image_list.append(restoredimage)
                image_list.append(targetImages)
                result_image = torch.cat(image_list, dim=3)
                sample_path = os.path.join(self.train_dir, self.sample_dir, '{}-images.jpg'.format(i + 1))
                if self.image_layer == 'tanh':
                    save_image(self.denorm(result_image.data.cpu()), sample_path, nrow=1, padding=0)
                elif self.image_layer == 'sigmoid':
                    save_image(result_image.data.cpu(), sample_path, nrow=1, padding=0)
                else:
                    raise Exception('Please check the gen_last layer mode')