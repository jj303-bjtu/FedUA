# example launch script

# common arguments for all methods
# 添加拜占庭攻击相关参数
BASE_ARGS="--num_clients 100 --sampling_prob 0.3 --local_epochs 5 --global_rounds 200 --eval_gap 200 --use_adv_ood --adv_epsilon 0.1 --plot_dir ./results/ --uncertainty"

# 拜占庭攻击参数说明:
# --byzantine_ratio 0.1 --byzantine_attack_type random
# --byzantine_ratio: 拜占庭客户端比例 (0.1 = 10%)
# --byzantine_attack_type: 攻击类型 ['random', 'label_flip', 'gaussian_noise', 'sign_flip']

# data-scarcity arguments: e.g., train prop [0.25, 0.5, 0.75, 1.0]
# if we have created augmented datasets, we can also include --augmented to include covariate shift
# if we have add FedUA, we can also include --uncertainty
# if we have add ood_dataset, we can also include --ood_dataset svhn
# if we have add prior_report_strategy, we can also include --prior_report_strategy raw_evidence
# prior_report_strategy in ['softmax_prob', 'raw_evidence']
# add use_adv_ood
# add adv_epsilon 
# add plot_dir
# --global_consistency_weight 2.0
# --use_prior_smoothing
# --use_prior_clamping
DATA_ARGS="--train_prop 1.0 --use_prior_smoothing --use_prior_clamping --ood_dataset svhn --use_byzantine_robust"

# select model e.g., ['cnn', 'resnet18']
MODEL_ARGS="--model_name cnn"

# method-specific arguments
FEDAVG_ARGS="--method FedUgV2"
FEDFDA_ARGS="--method pFedFDA"

# specify dataset arguments
DATASET_ARGS="--dataset cifar10 --num_classes 10"

# specify dataset partition arguments
PARTITION_ARGS="--partition_path cifar10_c100_dir05_1"
# see data/partition/*.json name

export DATA_PATH="/home/wenkang_kong/Desktop/FedUA/data"

# FedAvg | FedAvgFT
nohup python main.py ${BASE_ARGS} ${FEDAVG_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> FedUA_multi_OOD_cifar10_no_bzt_random.log

# pFedFDA
# nohup python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> FedFope_graph_multi_c100_cifar10_5_100.log
# python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS}
# pFedMDG
# nohup python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> 1.log
# python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS}
