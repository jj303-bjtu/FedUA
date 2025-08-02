import argparse
import torch
import numpy as np
from data.utils.loader import get_base_dataset
from servers.server_fedavg import ServerFedAvg
from servers.server_fedfda import ServerFedFDA
from servers.server_fedpac import ServerFedPac
from servers.server_fedfope import ServerFedFope
from servers.server_local import ServerLocal
from servers.server_fedtaylorcfl import ServerFedTaylorCFL
from servers.server_fedug import ServerFedUg
from models import model_dict

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset arguments
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "digit5", "tinyimagenet", "emnist"],
                        type=str)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--partition_path", default="cifar10_c100_dir05", type=str, help="name of partition folder")
    parser.add_argument("--augmented", action="store_true", help="whether or not to augment the first 50 clients (Only for CIFAR)")
    parser.add_argument("--taylor", action="store_true", help="whether use taylor protection")
    # generic training hyperparameters
    parser.add_argument("--global_rounds", default=200, type=int)
    parser.add_argument("--local_epochs", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float) # Consider 0.001 or 0.0005 if issues persist
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--eval_gap", default=1, type=float, help="Rounds Between Model Evaluation (set to 1 for frequent eval during debugging)") # Changed default for debugging
    parser.add_argument("--train_prop", default=1.0, type=float, help="Proportion of Training Data To Use")
    # FL/Server Setup
    parser.add_argument("--method", default="FedUg", type=str) # Changed to FedUg for testing this setup
    parser.add_argument("--num_clients", default=100, type=int)
    parser.add_argument("--sampling_prob", default=0.3, type=float, help="Client Participation Probability")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    # model architecture
    parser.add_argument("--model_name", default="cnn", type=str, help="Model Architecture", choices=["cnn", "resnet18", "cifaredl", "emnistnet", "imagenet"]) # Added cifaredl
    # method-specific hyperparameters
    parser.add_argument("--p_epochs", default=5, type=int, help="Number of Personalization Epochs")
    parser.add_argument("--single_beta", action="store_true", help="if we should only use a single beta term (pFedFDA)")
    parser.add_argument("--local_beta", action="store_true", help="if we should only use only local statistics (pFedFDA)")
    # logging/saving
    parser.add_argument("--exp_name", default="fedug_optimized_test", type=str, help="save file prefix") # Changed for clarity
    parser.add_argument('--beta_tau', type=float, default=50,
                   help='Beta temperature parameter')
    parser.add_argument('--min_samples', type=int, default=20,
                    help='Minimum samples for beta calculation')

    # taylor protection
    parser.add_argument('--in_channels', type=int, default=3,
                       help='in_channels')
    parser.add_argument('--D_h_taylor', type=int, default=128, help='Hidden dimension for TaylorMLP_Layer')
    parser.add_argument('--N_taylor', type=int, default=3, help='Order of Taylor expansion for TaylorMLP_Layer')
    parser.add_argument('--taylor_activation', type=str, default="gelu", help='Activation function for TaylorMLP_Layer (e.g., gelu)')
    parser.add_argument("--uncertainty", action="store_true", help="if we should use uncertainty (defaulted to True for FedUg)") # Defaulted to True for FedUg

    # For Adaptive Annealing (Prior Correction) - Retained from original file
    parser.add_argument('--adaptive_annealing_threshold', type=float, default=0.3, help='Skew threshold for slower annealing (fraction of total classes).')
    parser.add_argument('--adaptive_annealing_factor', type=float, default=1.5, help='Factor to slow down annealing for skewed clients.')

    # For OOD Regularization - Retained from original file
    parser.add_argument('--lambda_ood_reg', type=float, default=0.1, help='Weight for OOD regularization loss.')

    parser.add_argument('--default_prior_strength', type=float, default=None, # Will default to num_classes in server if None
                        help='Initial strength S0 for the global prior. If None, defaults to num_classes. (default: None)')
    # OPTIMIZATION: Added a default to target_global_prior_strength
    parser.add_argument('--target_global_prior_strength', type=float, default=50.0, # Example: num_classes * 5. Original was None.
                        help='Target strength S0 for the aggregated global prior. If None, server defaults. (default: 50.0)')

    parser.add_argument('--use_fedavg_momentum', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether server uses momentum for weight aggregation (FedAvgM) (default: True)')
    parser.add_argument('--server_momentum_beta', type=float, default=0.9,
                        help='Beta for server-side momentum in FedAvgM (default: 0.9)')
    parser.add_argument('--server_lr', type=float, default=1.0,
                        help='Server learning rate for FedAvgM (often 1.0 for applying momentum) (default: 1.0)')

    parser.add_argument('--client_alpha_report_sample_frac', type=float, default=0.2,
                        help='Fraction of client local data for computing average alpha report (1.0 for full) (default: 0.2)')
    parser.add_argument('--evaluate_personalized_models', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether to run personalization and evaluation during eval gaps (default: True)')
    
    # Gradient Clipping arguments (retained)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--clip_grad_value', type=float, default=None, help='Max value for gradient clipping')

    # Other arguments (retained from original file)
    parser.add_argument('--dir_alpha', type=float, default=0.5, help='dir_alpha')
    parser.add_argument('--eta_g', type=float, default=0.1, help='Global learning rate')
    parser.add_argument('--tau', type=float, default=0.8, help='Local update steps')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--whitening', type=bool, default=True, help='whitening') # Note: type=bool can be tricky with argparse. Use type=lambda x: (str(x).lower() == 'true')
    parser.add_argument('--grad_align_lambda', type=float, default=1.0, help='grad_align_lambda')
    
    # FedUG specific arguments
    parser.add_argument('--use_global_prior_regularization', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use global prior regularization')
    parser.add_argument('--edl_kl_global_prior_lambda', type=float, default=1.0, help='KL divergence weight for global prior')
    parser.add_argument('--edl_kl_lambda_personalization', type=float, default=0.5, help='KL divergence weight for personalization')
    parser.add_argument('--edl_annealing_epochs', type=int, default=100, help='Number of epochs for KL annealing')
    parser.add_argument('--clip_alpha_max_val', type=float, default=50.0, help='Maximum value for alpha clipping (reduced for better stability)')
    parser.add_argument('--use_ood_reg', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use OOD regularization')
    parser.add_argument('--ood_aug_strength', type=float, default=0.1, help='OOD augmentation strength')
    parser.add_argument('--lr_personalization', type=float, default=0.001, help='Learning rate for personalization (reduced for better stability)')
    parser.add_argument('--personalization_epochs_on_eval', type=int, default=5, help='Personalization epochs (reduced to avoid overfitting)')
    parser.add_argument('--use_data_augmentation', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use data augmentation for better generalization')

    ### SUGGESTION 2: ABLATION STUDY ARGUMENTS ###
    parser.add_argument('--global_consistency_weight', type=float, default=1.0, help='Weight for the KL-divergence term (lambda in the paper).')
    parser.add_argument('--prior_strength', type=float, default=5.0, help='Scale factor for converting softmax probabilities to Dirichlet alpha parameters in the client report.')
    parser.add_argument('--prior_report_strategy', type=str, default='softmax_prob', choices=['softmax_prob', 'raw_evidence'], help='Strategy for clients to report prior information.')
    parser.add_argument('--use_prior_smoothing', action='store_true', default=False, help='Enable smoothing of the global prior on the server.')
    parser.add_argument('--use_prior_clamping', action='store_true', default=False, help='Enable clamping of the global prior on the server.')

    ### SUGGESTION 1: UNCERTAINTY/OOD ARGUMENTS ###
    parser.add_argument('--ood_dataset', type=str, default='', help='Name of the OOD dataset to use for evaluation (e.g., "svhn"). Your data loader must handle this.')
    # 在您的主脚本中添加这些参数
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing for better calibration.')
    parser.add_argument('--use_adv_ood', action='store_true', default=False, help='Enable adversarial training for OOD detection.')
    parser.add_argument('--adv_epsilon', type=float, default=0.01, help='Epsilon for FGSM adversarial attack.')
    parser.add_argument('--adv_reg_weight', type=float, default=0.5, help='Weight for the adversarial OOD regularization loss.')
    parser.add_argument('--plot_dir', type=str, default='', help='')
    parser.add_argument('--temperature', type=float, default=1.2, 
                        help='Temperature parameter for calibrating predictions')

    # 添加FedUG V2敏感性分析相关参数
    parser.add_argument('--max_evidence', type=float, default=2.0, help='Maximum evidence value')
    parser.add_argument('--edl_weight', type=float, default=0.3, help='EDL loss weight')

    # 添加拜占庭攻击相关参数
    parser.add_argument('--byzantine_ratio', type=float, default=0.0, 
                        help='Ratio of Byzantine clients (0.0-1.0, default: 0.0)')
    parser.add_argument('--byzantine_attack_type', type=str, default='random',
                        choices=['random', 'label_flip', 'gaussian_noise', 'sign_flip'],
                        help='Type of Byzantine attack (default: random)')

    # 添加拜占庭鲁棒性参数
    parser.add_argument('--use_byzantine_robust', action='store_true', default=False, 
                       help='Use Byzantine-robust aggregation for alpha priors')
    parser.add_argument('--byzantine_detection_threshold', type=float, default=5.0,
                       help='Threshold for detecting Byzantine clients based on alpha variance')

    args = parser.parse_args()

    # numpy seed (ensures repeatable subsampling)
    np.random.seed(0) # For reproducibility

    # ensure arguments are correct
    if args.dataset in ["mnist", "emnist", "fmnist"]:
        args.in_channels = 1 # Corrected from in_channels to args.in_channels
        if args.model_name == "cnn": # Default model_name
            if args.uncertainty:
                args.model_name = "emnistnetedl"
            else:
                args.model_name = "emnistnet"
    else: # CIFAR, TinyImageNet
        args.in_channels = 3 # Corrected from in_channels to args.in_channels
        if args.model_name == "cnn": # Default model_name
            if args.uncertainty:
                args.model_name = "cifaredl" # Select EDL model if uncertainty is True
            else:
                args.model_name = "cifarnet" # Original cifarnet if not uncertainty

    if args.dataset == "emnist":
        args.batch_size = 16
        args.num_classes = 62 # Example for EMNIST ByClass, adjust if different split

    if args.dataset == "tinyimagenet":
        args.num_classes = 200
        if args.model_name not in ["resnet18", "imagenet"]: # imagenet is a custom resnet-like model
            if args.uncertainty:
                args.model_name = "imagenetedl"
            else:
                args.model_name = "imagenet" # Default to a suitable model for TinyImageNet

    if args.dataset == "cifar100":
        args.num_classes = 100
        if args.uncertainty and args.model_name == "cnn": # Ensure cifaredl for CIFAR-100 uncertainty
            args.model_name = "cifaredl"
        else:
            args.model_name = "cifarnet" # Original cifarnet if not uncertainty


    args.model = model_dict[args.model_name](num_classes=args.num_classes, in_channels=args.in_channels)
    args.base_dataset = get_base_dataset(args)
    return args

def main(args):
    if args.method == "FedAvg":
        server = ServerFedAvg(args)
    elif args.method == "Local":
        server = ServerLocal(args)
    elif args.method == "pFedFDA":
        server = ServerFedFDA(args)
    elif args.method == "FedPac":
        server = ServerFedPac(args)
    elif args.method == "FedFope":
        server = ServerFedFope(args)
    elif args.method == "FedTaylorCFL":
        server = ServerFedTaylorCFL(args)
    elif args.method == "FedUg":
        server = ServerFedUg(args)
    elif args.method == "FedUgV2":
        from servers.server_fedug_v2 import ServerFedUgV2
        server = ServerFedUgV2(args)
    elif args.method == "FedUgV3":
        from servers.server_fedug_v3 import ServerFedUgV3
        server = ServerFedUgV3(args)
    else:
        raise NotImplementedError

    print(f"Training {args.method} with model {args.model_name} on {args.dataset} for {args.num_clients} clients.")
    print(f"Hyperparameters: LR={args.lr}, WD={args.wd}, Local Epochs={args.local_epochs}, Global Rounds={args.global_rounds}")
    if args.method == "FedUg":
        print(f"FedUg Specifics: ClipAlphaMax={args.clip_alpha_max_val}, TargetGlobalPriorStrength={args.target_global_prior_strength}")
    
    # 添加拜占庭攻击信息打印
    if args.byzantine_ratio > 0:
        num_byzantine = int(args.byzantine_ratio * args.num_clients)
        print(f"Byzantine Attack: {args.byzantine_attack_type}, Ratio={args.byzantine_ratio} ({num_byzantine}/{args.num_clients} clients)")
    else:
        print("No Byzantine attack configured.")

    server.train() # Server's main training loop
    if hasattr(server, 'train_times') and server.train_times:
        print(f"Method took ({np.mean(server.train_times):.2f} +/- {np.std(server.train_times):.2f}) seconds per training iteration (client train + server agg).")
    if hasattr(server, 'round_times') and server.round_times:
        print(f"Method took ({np.sum(server.round_times):.2f}) total seconds for all rounds.")

if __name__ == "__main__":
    args = parse_args()
    # Ensure model_dict has 'cifaredl' and other models
    if 'cifaredl' not in model_dict:
        from models.uncertainty_cnn import CIFARNet_EDL # Assuming this is the path
        model_dict['cifaredl'] = CIFARNet_EDL
    # Add other models if not present, e.g., emnistnet, imagenet (custom resnet)

    # A simple way to add them if they are in a 'models' directory and follow a pattern:
    if args.model_name not in model_dict:
        if args.model_name == "emnistnet": # Example
            # from models.emnistnet import EMNISTNet # Replace with actual import
            # model_dict['emnistnet'] = EMNISTNet
            pass # Add imports as needed
        elif args.model_name == "imagenet": # Example for a custom ResNet-like model
            # from models.resnet_tinyimagenet import ResNet_TinyImageNet # Replace
            # model_dict['imagenet'] = ResNet_TinyImageNet
            pass # Add imports as needed

    if args.model_name not in model_dict and args.model_name == "cifarnet":
         from models.cnn import CIFARNet # Assuming cnn.py contains CIFARNet
         model_dict['cifarnet'] = CIFARNet
    
    # Make sure the selected model is in model_dict before creating it
    if args.model_name not in model_dict:
        raise ValueError(f"Model {args.model_name} not found in model_dict. Please define or import it.")
    
    args.model = model_dict[args.model_name](num_classes=args.num_classes, in_channels=args.in_channels)


    main(args)
