from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',help='dataset name: regdb or sysu or llcm')
parser.add_argument('--lr', default=0.1, type=float,help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,help='network baseline: resnet50')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int,metavar='s', help='save model every 10 epochs')
parser.add_argument('--resume', '-r', default='', type=str,help='resume from checkpoint (can be absolute path)')
parser.add_argument('--model_path',default='./result/',type=str,help='model save path')
parser.add_argument('--log_path', default='result/SCTNet/log/',type=str, help='log save path')
parser.add_argument('--vis_log_path', default='result/SCTNet/log/vis_log/',type=str, help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',help='number of data loading workers')
parser.add_argument('--img_w', default=192, type=int,metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,metavar='m', help='method type: base or agw')
parser.add_argument('--backbone', default='AGW', type=str, help='backbone type: AGW or base')
parser.add_argument('--margin', default=0.3, type=float,metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,metavar='npos', help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,metavar='g', help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str,metavar='m', help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true',help='retrieve thermal to visible search on RegDB')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Data/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]  # [gallery_mode=1(visible), query_mode=2(thermal)]
elif dataset == 'regdb':
    data_path = './Data/RegDB/'
    n_class = 206
    test_mode = [2, 1]   # [gallery_mode=2(thermal), query_mode=1(visible)]
elif dataset == 'llcm':
    data_path = './Data/LLCM/' 
    n_class = 713
    test_mode = [2, 1]  # [gallery_mode=2(thermal), query_mode=2(NIR)] - NIR to Visible search

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048
print('==> Building model..')
if args.backbone == 'AGW':
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
elif args.method == 'awg':
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
else:
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
    
net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path

print('==> Loading data..')
# Data loading code

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(gall_loader, ngall):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, input, input, test_mode[0])
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader, nquery):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, input, input, test_mode[1])
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc


def normalize_features(features):
    """L2 normalize features"""
    return features / np.linalg.norm(features, axis=1, keepdims=True)


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.resume
        print('Loading model from:', model_path)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['net'])
            
            print('Model loaded successfully!')
            print(f'Model on device: {next(net.parameters()).device}')
            net.eval()
            print(f'Model in eval mode: {not net.training}')
            
            print('==> loaded checkpoint {} (epoch {})'
                  .format(model_path, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(model_path))
            exit()
    else:
        print('Warning: No checkpoint specified. Using untrained model.')

    print(f'\nTest Mode: {args.mode}')
    print(f'test_mode = {test_mode} (gallery_mode={test_mode[0]}, query_mode={test_mode[1]})')

    query_img, query_label, query_cam = process_query_sysu(
        data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(
        data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("\nDataset statistics:")
    print("  " + "-"*30)
    print("  subset   | # ids | # images")
    print("  " + "-"*30)
    print("  query    | {:5d} | {:8d}".format(
        len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(
        len(np.unique(gall_label)), ngall))
    print("  " + "-"*30)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(
        args.img_w, args.img_h))
    query_loader = data.DataLoader(
        queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery)
    
    query_feat_pool = normalize_features(query_feat_pool)
    query_feat_fc = normalize_features(query_feat_fc)
    
    print(f"\nQuery Feature Info:")
    print(f"  Query shape: {query_feat_fc.shape}")
    print(f"  FC Norm (mean): {np.mean(np.linalg.norm(query_feat_fc, axis=1)):.4f}")
    print(f"  FC Norm (min/max): {np.min(np.linalg.norm(query_feat_fc, axis=1)):.4f} / {np.max(np.linalg.norm(query_feat_fc, axis=1)):.4f}")
    print(f"  POOL Norm (mean): {np.mean(np.linalg.norm(query_feat_pool, axis=1)):.4f}")
    
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(
            data_path, mode=args.mode, trial=trial)

        ngall = len(gall_label)
        
        if trial == 0:
            print(f"\nGallery info (trial 0) - IDs: {len(np.unique(gall_label))}, Images: {len(gall_label)}")
        elif trial == 1:
            print(f"Gallery info (trial 1) - IDs: {len(np.unique(gall_label))}, Images: {len(gall_label)}")

        trial_gallset = TestData(
            gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(
            trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader, ngall)
        
        gall_feat_pool = normalize_features(gall_feat_pool)
        gall_feat_fc = normalize_features(gall_feat_fc)

        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        
        if trial == 0:
            print(f"\nDistance stats (POOL) - min: {-distmat_pool.min():.4f}, max: {-distmat_pool.max():.4f}, mean: {-distmat_pool.mean():.4f}")
        
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(
            -distmat_pool, query_label, gall_label, query_cam, gall_cam)

        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        
        if trial == 0:
            print(f"Distance stats (FC) - min: {-distmat.min():.4f}, max: {-distmat.max():.4f}, mean: {-distmat.mean():.4f}")
        
        cmc, mAP, mINP = eval_sysu(-distmat, query_label,
                                   gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

elif dataset == 'regdb':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.resume
        print('Loading model from:', model_path)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['net'])
            
            print('Model loaded successfully!')
            print(f'Model on device: {next(net.parameters()).device}')
            net.eval()
            print(f'Model in eval mode: {not net.training}')
            
            print('==> loaded checkpoint {} (epoch {})'
                  .format(model_path, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(model_path))
            exit()
    else:
        print('Error: --resume is required for testing!')
        exit()

    if args.tvsearch:
        print('\nTest Mode: Thermal to Visible (I2V)')
    else:
        print('\nTest Mode: Visible to Thermal (V2I)')

    for trial in range(10):
        test_trial = trial + 1

        if args.tvsearch:
            query_img, query_label = process_test_regdb(
                data_path, trial=test_trial, modal='thermal')
            gall_img, gall_label = process_test_regdb(
                data_path, trial=test_trial, modal='visible')
            current_test_mode = [1, 2]
        else:
            query_img, query_label = process_test_regdb(
                data_path, trial=test_trial, modal='visible')
            gall_img, gall_label = process_test_regdb(
                data_path, trial=test_trial, modal='thermal')
            current_test_mode = [2, 1]

        nquery = len(query_label)
        ngall = len(gall_label)

        print('\nTest Trial: {}'.format(test_trial))
        print("Dataset statistics:")
        print("  " + "-"*30)
        print("  subset   | # ids | # images")
        print("  " + "-"*30)
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  " + "-"*30)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(
            args.img_w, args.img_h))
        query_loader = data.DataLoader(
            queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(
            args.img_w, args.img_h))
        gall_loader = data.DataLoader(
            gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        test_mode_backup = test_mode
        test_mode = current_test_mode
        
        query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery)
        gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader, ngall)
        
        test_mode = test_mode_backup

        query_feat_pool = normalize_features(query_feat_pool)
        query_feat_fc = normalize_features(query_feat_fc)
        gall_feat_pool = normalize_features(gall_feat_pool)
        gall_feat_fc = normalize_features(gall_feat_fc)

        if trial == 0:
            print(f"\nFeature Info (Trial 0):")
            print(f"  Query shape: {query_feat_fc.shape}")
            print(f"  Query FC Norm (mean): {np.mean(np.linalg.norm(query_feat_fc, axis=1)):.4f}")
            print(f"  Query FC Norm (min/max): {np.min(np.linalg.norm(query_feat_fc, axis=1)):.4f} / {np.max(np.linalg.norm(query_feat_fc, axis=1)):.4f}")
            print(f"  Gallery FC Norm (mean): {np.mean(np.linalg.norm(gall_feat_fc, axis=1)):.4f}")

        if args.tvsearch:
            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
            
            if trial == 0:
                print(f"\nDistance stats (POOL) - min: {-distmat_pool.min():.4f}, max: {-distmat_pool.max():.4f}, mean: {-distmat_pool.mean():.4f}")
            
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

            distmat = np.matmul(gall_feat_fc, np.transpose(query_feat_fc))
            
            if trial == 0:
                print(f"Distance stats (FC) - min: {-distmat.min():.4f}, max: {-distmat.max():.4f}, mean: {-distmat.mean():.4f}")
            
            cmc, mAP, mINP = eval_regdb(-distmat, gall_label, query_label)
        else:
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            
            if trial == 0:
                print(f"\nDistance stats (POOL) - min: {-distmat_pool.min():.4f}, max: {-distmat_pool.max():.4f}, mean: {-distmat_pool.mean():.4f}")
            
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            
            if trial == 0:
                print(f"Distance stats (FC) - min: {-distmat.min():.4f}, max: {-distmat.max():.4f}, mean: {-distmat.mean():.4f}")
            
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print(
            'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

elif dataset == 'llcm':
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['net'])
            
            print('Model loaded successfully!')
            print(f'Model on device: {next(net.parameters()).device}')
            net.eval()
            print(f'Model in eval mode: {not net.training}')
            
            print('==> loaded checkpoint {} (epoch {})'
                  .format(model_path, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(model_path))
            exit()
    else:
        print('Error: --resume is required for testing!')
        exit()

    print(f'\ntest_mode = {test_mode} (gallery_mode={test_mode[0]}, query_mode={test_mode[1]})')

    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  " + "-"*30)
    print("  subset   | # ids | # images")
    print("  " + "-"*30)
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  " + "-"*30)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery)
    
    query_feat_pool = normalize_features(query_feat_pool)
    query_feat_fc = normalize_features(query_feat_fc)
    
    print(f"\nQuery Feature Info:")
    print(f"  Query shape: {query_feat_fc.shape}")
    print(f"  FC Norm (mean): {np.mean(np.linalg.norm(query_feat_fc, axis=1)):.4f}")
    print(f"  FC Norm (min/max): {np.min(np.linalg.norm(query_feat_fc, axis=1)):.4f} / {np.max(np.linalg.norm(query_feat_fc, axis=1)):.4f}")
    print(f"  POOL Norm (mean): {np.mean(np.linalg.norm(query_feat_pool, axis=1)):.4f}")
    
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        ngall = len(gall_label)
        
        if trial == 0:
            print(f"\nGallery info (trial 0) - IDs: {len(np.unique(gall_label))}, Images: {len(gall_label)}")
        elif trial == 1:
            print(f"Gallery info (trial 1) - IDs: {len(np.unique(gall_label))}, Images: {len(gall_label)}")
        
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader, ngall)
        
        gall_feat_pool = normalize_features(gall_feat_pool)
        gall_feat_fc = normalize_features(gall_feat_fc)

        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        
        if trial == 0:
            print(f"\nDistance stats (POOL) - min: {-distmat_pool.min():.4f}, max: {-distmat_pool.max():.4f}, mean: {-distmat_pool.mean():.4f}")
            print(f"Distance stats (FC) - min: {-distmat_fc.min():.4f}, max: {-distmat_fc.max():.4f}, mean: {-distmat_fc.mean():.4f}")

        a = 0.1
        distmat = distmat_pool + distmat_fc
        distmat_A = a * distmat_pool + (1 - a) * distmat_fc

        cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_A, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_att
            all_mAP_pool = mAP_att
            all_mINP_pool = mINP_att
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_att
            all_mAP_pool = all_mAP_pool + mAP_att
            all_mINP_pool = all_mINP_pool + mINP_att

        print('Test Trial: {}'.format(trial))
        print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC+POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

# ====== Print average results ======
cmc = all_cmc / 10
mAP = all_mAP / 10
mINP = all_mINP / 10

cmc_pool = all_cmc_pool / 10
mAP_pool = all_mAP_pool / 10
mINP_pool = all_mINP_pool / 10

print('\n' + '='*80)
print('All Average:')
print('='*80)
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
print('='*80)