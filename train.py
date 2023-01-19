import os
import pdb
import time
import yaml
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from avsr.utils.getter import get_criterion, get_optimizer
from avsr.utils.model_builder import build_model
from vocabulary.utils import KsponSpeechVocabulary
from dataset.dataset import prepare_dataset, AVcollator
from dataset.sampler import DistributedCurriculumSampler


def setup(rank, world_size, port):
    """
    world_size : number of processes
    rank : this should be a number between 0 and world_size-1
    """
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    '''
    Initialize the process group
    Rule of thumb choosing backend
    NCCL -> GPU training / Gloo -> CPU training
    check table here : https://pytorch.org/docs/stable/distributed.html
    '''
    
    import platform
    backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    
def cleanup():
    dist.destroy_process_group()


def save_checkpoint(model, checkpoint_path, epoch):
    if isinstance(epoch, int):
        torch.save(model.state_dict(), f"{checkpoint_path}/{epoch:05d}.pt")
    else:
        torch.save(model.state_dict(), f"{checkpoint_path}/{epoch}.pt")

    
def load_ddp_checkpoint(rank, model, checkpoint_path, epoch):
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(f"{checkpoint_path}/{epoch:05d}.pt", map_location=map_location)
    if rank==0: print(f"Loaded weights from {epoch} epoch")
    model.load_state_dict(state_dict, strict=False)


def show_description(epoch, total_epoch, it, total_it, lr, loss, mean_loss, _time, ctc_loss=None, att_loss=None, ctc_mean_loss=None, att_mean_loss=None, head=""):
    train_time = int(time.time() - _time)
    _sec = train_time % 60
    train_time //= 60
    _min = train_time % 60
    train_time //= 60
    _hour = train_time % 24
    _day = train_time // 24
    if ctc_loss:
        desc = f"{head}LOSS(Total/CTC/ATT) {loss:.4f}/{ctc_loss:.4f}/{att_loss:.4f} :: MEAN LOSS {mean_loss:.4f}/{ctc_mean_loss:.4f}/{att_mean_loss:.4f} :: LEARNING_RATE {lr:.8f} :: BATCH [{it}/{total_it}] :: EPOCH [{epoch}/{total_epoch}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    else:
        desc = f"{head}LOSS {loss:.4f} :: MEAN LOSS {mean_loss:.4f} :: LEARNING_RATE {lr:.8f} :: BATCH [{it}/{total_it}] :: EPOCH [{epoch}/{total_epoch}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    print(desc, end="\r")


def train(rank, world_size, config, vocab, trainset, validset, port):
    setup(rank, world_size, port)

    # define a loader
    train_sampler = DistributedCurriculumSampler(trainset, num_replicas=world_size, rank=rank, drop_last=False)
    val_sampler = DistributedCurriculumSampler(validset, num_replicas=world_size, rank=rank, drop_last=False)
    
    dataloader = DataLoader(dataset=trainset, batch_size=int(config['batch_size']/world_size),
                            collate_fn = AVcollator(
                                max_len = config['max_len'],
                                use_video = config['use_video'],
                                raw_video = config['raw_video'],), 
                            shuffle = False,
                            sampler = train_sampler,
                            num_workers=config['num_workers'])

    valloader = DataLoader(dataset=validset, batch_size=int(config['batch_size']/world_size),
                            collate_fn = AVcollator(
                                max_len = config['max_len'],
                                use_video = config['use_video'],
                                raw_video = config['raw_video'],), 
                            shuffle = False,
                            sampler = val_sampler,
                            num_workers=config['num_workers'])

    if rank==0:
        print(f'# of batch for each rank : {len(dataloader)}')
        
        if not os.path.exists('results/'+config['save_dir']):
            os.makedirs('results/'+config['save_dir'])

    # define a model
    model = build_model(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        architecture=config['architecture'],
        loss_fn=config['loss_fn'],
        front_dim=config['front_dim'],
        encoder_n_layer=config['encoder_n_layer'],
        encoder_d_model=config['encoder_d_model'],
        encoder_n_head=config['encoder_n_head'], 
        encoder_ff_dim=config['encoder_ff_dim'], 
        encoder_dropout_p=config['encoder_dropout_p'],
        decoder_n_layer=config['decoder_n_layer'],
        decoder_d_model=config['decoder_d_model'],
        decoder_n_head=config['decoder_n_head'], 
        decoder_ff_dim=config['decoder_ff_dim'], 
        decoder_dropout_p=config['decoder_dropout_p'],
        pass_visual_frontend= not config['raw_video'],
        verbose=rank==0,
    )
    # move the model to GPU with id rank
    model.to(rank)
    # load state dict
    if config['resume_epoch'] > -1:
        load_ddp_checkpoint(rank, model, checkpoint_path='results/'+config['save_dir'], epoch=config['resume_epoch'])

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # define a criterion
    criterion = get_criterion(loss_fn=config['loss_fn'], ignore_index=vocab.pad_id, blank_id=vocab.unk_id)
    steps_per_epoch = len(dataloader)
    optimizer, scheduler = get_optimizer(
                             ddp_model.parameters(), 
                             learning_rate = config['learning_rate'],
                             init_lr = config['init_lr'],
                             final_lr = config['final_lr'],
                             gamma = config['gamma_lr'],
                             epochs = config['epochs'],
                             warmup = config['warmup'],
                             steps_per_epoch = steps_per_epoch,
                             scheduler = config['scheduler'],
                           )
    
    if rank==0:
        for epoch in range(config['resume_epoch']+1):
            scheduler.step(on='epoch')
    
    train_start = time.time()
    last_save = time.time()
    for epoch in range(config['resume_epoch']+1, config['epochs']):
        dist.barrier()
        
        ## shuffle dataset
        train_sampler.set_epoch(epoch)
        
        # Train
        ddp_model.train()
        epoch_total_loss = 0
        epoch_ctc_loss = 0
        epoch_att_loss = 0
        for it, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in enumerate(dataloader):
            vids = vids.to(rank)
            seqs = seqs.to(rank)
            targets = targets.to(rank)
            target_lengths = target_lengths.to(rank)
            
            """
            Check input sizes
            
            print()
            print(vids.size())
            print(seqs.size())
            """
            
            optimizer.zero_grad()
            outputs = ddp_model(vids, vid_lengths,
                                seqs, seq_lengths,
                                targets[:,:-1], target_lengths) # drop eos_id
            
            # if isinstance(outputs, tuple): outputs, att = outputs
            # else: att = None
                                
            loss = criterion(outputs=outputs, targets=targets[:,1:], target_lengths=target_lengths) # drop sos_id
            if isinstance(loss, tuple):
                loss[0].backward()
                ctc_loss = loss[1]
                att_loss = loss[2]
                loss = loss[0].item()
                epoch_total_loss += loss
                epoch_ctc_loss += ctc_loss
                epoch_att_loss += att_loss
            else:
                loss.backward()
                ctc_loss = None
                att_loss = None
                loss = loss.item()
                epoch_total_loss += loss
            
            if config['max_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_norm"])

            optimizer.step()

            if rank == 0:
                cur_lr = scheduler.get_lr()[0]
                scheduler.step(on='step', step = epoch*steps_per_epoch + it)
                # show description
                show_description(
                    epoch=epoch, 
                    total_epoch=config['epochs'], 
                    it = it, 
                    total_it = len(dataloader), 
                    lr = cur_lr,
                    loss = loss, 
                    mean_loss = epoch_total_loss/(it+1), 
                    _time = train_start,
                    ctc_loss = ctc_loss,
                    att_loss = att_loss,
                    ctc_mean_loss = epoch_ctc_loss/(it+1), 
                    att_mean_loss = epoch_att_loss/(it+1), 
                    head = "Train      ::")
        if rank==0: print()

        # Validation
        ddp_model.eval()
        epoch_total_val_loss = 0
        epoch_ctc_val_loss = 0
        epoch_att_val_loss = 0
        with torch.no_grad():
            for it, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in enumerate(valloader):
                vids = vids.to(rank)
                seqs = seqs.to(rank)
                targets = targets.to(rank)
                target_lengths = target_lengths.to(rank)
                
                outputs = ddp_model(vids, vid_lengths,
                                    seqs, seq_lengths,
                                    targets[:,:-1], target_lengths) # drop eos_id
                                    
                loss = criterion(outputs=outputs, targets=targets[:,1:], target_lengths=target_lengths) # drop sos_id
                if isinstance(loss, tuple):
                    ctc_loss = loss[1]
                    att_loss = loss[2]
                    loss = loss[0].item()
                    epoch_total_val_loss += loss
                    epoch_ctc_val_loss += ctc_loss
                    epoch_att_val_loss += att_loss
                else:
                    ctc_loss = None
                    att_loss = None
                    loss = loss.item()
                    epoch_total_val_loss += loss

                if rank == 0:
                    # show description
                    show_description(
                        epoch=epoch, 
                        total_epoch=config['epochs'], 
                        it = it, 
                        total_it = len(valloader), 
                        lr = 0,
                        loss = loss, 
                        mean_loss = epoch_total_val_loss/(it+1), 
                        _time = train_start,
                        ctc_loss = ctc_loss,
                        att_loss = att_loss,
                        ctc_mean_loss = epoch_ctc_val_loss/(it+1), 
                        att_mean_loss = epoch_att_val_loss/(it+1), 
                        head = "Validation ::")

        if rank==0:
            scheduler.step(on='epoch', loss = epoch_total_val_loss)
            save_checkpoint(model, 'results/'+config['save_dir'], epoch)
            last_save = time.time()
            print()
            print()
        
    cleanup()


def main(args):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # fix the seed
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])

    vocab = KsponSpeechVocabulary(unit = config['tokenize_unit'])
    
    # load dataset
    dataset = prepare_dataset(
        transcripts_path = config['transcripts_path_train'],
        vocab = vocab,
        use_audio = config.get('use_audio', True),
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
        audio_sample_rate = config['audio_sample_rate'],
        audio_n_mels = config['audio_n_mels'],
        audio_frame_length = config['audio_frame_length'],
        audio_frame_shift = config['audio_frame_shift'],
        audio_normalize = config['audio_normalize'],
        spec_augment = config['spec_augment'],
        freq_mask_para = config['freq_mask_para'],
        freq_mask_num = config['freq_mask_num'],
        time_mask_num = config['time_mask_num'],
        noise_rate = config['noise_augment'],
        noise_path = config['noise_path'],
    )
    validset = prepare_dataset(
        transcripts_path = config['transcripts_path_valid'],
        vocab = vocab,
        use_audio = config.get('use_audio', True),
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
        audio_sample_rate = config['audio_sample_rate'],
        audio_n_mels = config['audio_n_mels'],
        audio_frame_length = config['audio_frame_length'],
        audio_frame_shift = config['audio_frame_shift'],
        audio_normalize = config['audio_normalize'],
    )
    print(f"# of data : {len(dataset)}")
    print(f"# of validation data : {len(validset)}")


    # train
    '''
    spawn nprocs processes that run fn with args
    process index passed to fn
    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
    '''
    world_size = args.world_size if args.world_size else torch.cuda.device_count()
    mp.spawn(train,
            args=(world_size, config, vocab, dataset, validset, args.port),
            nprocs=world_size,
            join=True)


def get_args():
    parser = argparse.ArgumentParser(description='option for AV training.')
    parser.add_argument('-w','--world_size',
                         default=0, type=int, help='Configurate the number of GPUs')
    parser.add_argument('-c','--config',
                         type=str, help='Configuration Path')
    parser.add_argument('-d','--data_folder',
                         action = 'store_true', help='Data folder path') 
    parser.add_argument('-p','--port',
                         default = '12355', type = str, help='Port number of multi process')
    args = parser.parse_args()
    return args
  

if __name__ == '__main__':
    args = get_args()
    main(args)
    
