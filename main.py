import os
import argparse
from trainer import Trainer
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(os.path.join(config.train_dir,config.log_dir)):
        os.makedirs(os.path.join(config.train_dir,config.log_dir))
    if not os.path.exists(os.path.join(config.train_dir,config.model_save_dir)):
        os.makedirs(os.path.join(config.train_dir,config.model_save_dir))
    if not os.path.exists(os.path.join(config.train_dir,config.sample_dir)):
        os.makedirs(os.path.join(config.train_dir,config.sample_dir))
    if not os.path.exists(os.path.join(config.train_dir,config.result_dir)):
        os.makedirs(os.path.join(config.train_dir,config.result_dir))

    # def get_loader(image_dir, train_json, test_json, num_items=5, crop_size=178, image_size=256,
    #                batch_size=16, batch_mode='', mode='train', num_workers=1):
    # Data loader.
    print('Start get_loader...')
    data_loader = get_loader(config.image_dir, config.train_json, config.test_json,
                             config.image_size, config.batch_size, config.mode,
                             config.num_workers, config.train_pickle, config.test_pickle)
    print('Finish get_loader...')


    # Solver for training and testing StarGAN.
    trainer = Trainer(data_loader, config)

    if config.mode == 'train':
        print('Netowrk training start')
        trainer.train()
    elif config.mode == 'test':
        print('Network testing start')
        trainer.test()

def print_parameters(config):

        txt_path = os.path.join(config.train_dir, 'parameter_setting.txt')

        with open(txt_path,'w') as fp:

            fp.write('# Model confiurations' + '\n')
            fp.write('  image size : ' +  str(config.image_size) + '\n')
            fp.write('  encoder_mode : ' +  str(config.encoder_mode) + '\n')
            fp.write('  encoder_last : ' +  str(config.encoder_last) + '\n')
            fp.write('  encoder_start_ch : ' +  str(config.encoder_start_ch) + '\n')
            fp.write('  encoder_target_ch : ' +  str(config.encoder_target_ch) + '\n')
            fp.write('\n')
            fp.write('# Training configuration' + '\n')
            fp.write('  dataset : ' + str(config.dataset) + '\n')
            fp.write('  batch_size : ' + str(config.batch_size) + '\n')
            fp.write('  num_iters : ' + str(config.num_iters) + '\n')
            fp.write('  num_iters_decay : ' + str(config.num_iters_decay) + '\n')
            fp.write('  e_lr : ' + str(config.e_lr) + '\n')
            fp.write('  beta1 : ' + str(config.beta1) + '\n')
            fp.write('  beta2 : ' + str(config.beta2) + '\n')
            fp.write('  resume_iters : ' + str(config.resume_iters) + '\n')
            fp.write('  train_pickle : ' + str(config.train_pickle) + '\n')
            fp.write('  test_pickle : ' + str(config.test_pickle) + '\n')
            fp.write('  gpu : ' + str(config.gpu) + '\n')
            fp.write('\n')
            fp.write('# Test configuration' + '\n')
            fp.write('  test_iters : ' + str(config.test_iters) + '\n')
            fp.write('\n')
            fp.write('# Miscellaneous' + '\n')
            fp.write('  num_workers : ' + str(config.num_workers) + '\n')
            fp.write('  mode : ' + str(config.mode) + '\n')
            fp.write('  use_tensorboard : ' + str(config.use_tensorboard) + '\n')
            fp.write('  email_address : ' + str(config.email_address) + '\n')
            fp.write('  image_sending : ' + str(config.image_sending) + '\n')
            fp.write('\n')
            fp.write('# Directories' + '\n')
            fp.write('  image_dir : ' + str(config.image_dir) + '\n')
            fp.write('  train_json : ' + str(config.train_json) + '\n')
            fp.write('  test_json : ' + str(config.test_json) + '\n')
            fp.write('  train_dir : ' + str(config.train_dir) + '\n')
            fp.write('  log_dir : ' + str(config.log_dir) + '\n')
            fp.write('  model_save_dir : ' + str(config.model_save_dir) + '\n')
            fp.write('  sample_dir : ' + str(config.sample_dir) + '\n')
            fp.write('  result_dir : ' + str(config.result_dir) + '\n')
            fp.write('\n')
            fp.write('# Step size' + '\n')
            fp.write('  log_step : ' + str(config.log_step) + '\n')
            fp.write('  sample_step : ' + str(config.sample_step) + '\n')
            fp.write('  model_save_start : ' + str(config.model_save_start) + '\n')
            fp.write('  model_save_step : ' + str(config.model_save_step) + '\n')
            fp.write('  lr_update_step : ' + str(config.lr_update_step) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--encoder_mode', type=int, default=0, help='0 : Vanilla encoder, 1 : Variant auto encoder')
    parser.add_argument('--encoder_last', type=str, default='max', choices=['max', 'conv','avg'], help='Vanilla encoder last layer setting')
    parser.add_argument('--encoder_start_ch', type=int, default=64, help='E, start conv ch')
    parser.add_argument('--encoder_target_ch', type=int, default=25, help='E, target conv ch')
    parser.add_argument('--image_layer', type=str, default='tanh',choices=['tanh','sigmoid'], help='lasy layer for image')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='Polyvore', help='outfit generation dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--e_lr', type=float, default=0.0005, help='learning rate for E')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--train_pickle', type=str, default='./train_datalist.pickle', help='train data list pickle file directory')
    parser.add_argument('--test_pickle', type=str, default='', help='test data list pickle file directory')
    parser.add_argument('--gpu', type=str, default='0', help='specify only one gpu number')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument('--email_address', type=str, default='woodcook486@naver.com')
    parser.add_argument('--image_sending', type=int, default='10')

    # Directories.
    parser.add_argument('--image_dir', type=str, default='/home/woodcook486/polyvore/data/images/')
    parser.add_argument('--train_json', type=str, default='data/cleaned_train_no_dup.json')
    parser.add_argument('--test_json', type=str, default='data/cleaned_test_no_dup.json')
    parser.add_argument('--train_dir', type=str, default='Encoder')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--sample_step', type=int, default=10000)
    parser.add_argument('--model_save_start', type=int, default=150000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=100000)

    config = parser.parse_args()
    print(config)
    main(config)
    print_parameters(config)
