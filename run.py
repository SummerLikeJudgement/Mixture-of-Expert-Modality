import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import emoe
import sys

# 设置CUDA环境
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('EMOE')
torch.cuda.set_device(0)

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('EMOE')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def EMOE_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode = ''
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    # 模型配置目录
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    # 模型保存目录
    if model_save_dir == "":
        model_save_dir = Path.home() / "EMOE" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    # 结果保存目录
    if res_save_dir == "":
        res_save_dir = Path.home() / "EMOE" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    # 日志目录
    if log_dir == "":
        log_dir = Path.home() / "EMOE" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # 设置随机种子
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    # 设置logger
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    

    # 获取配置参数
    args = get_config_regression(model_name, dataset_name, config_file)
    args.mode = mode
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)# 分配GPU
    args['train_mode'] = 'regression' # 设置为回归任务
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)


    # 结果保存目录
    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    # 多次实验
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    # 蒸馏任务
    if args.is_distill:
        criterions = list(model_results[0].keys()) # 从model_results的第一个元素（字典）中获取所有的键
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2) # 计算该指标值的平均值
            std = round(np.std(values)*100, 2) # 计算该指标值的标准差
            res.append((mean, std))
        df.loc[len(df)] = res # 将res作为新行添加到DataFrame末尾
        df.to_csv(csv_file, index=None) # 保存csv文件
        logger.info(f"Results saved to {csv_file}.")

def _run(args, num_workers=4, is_tune=False, from_sena=False):
    # 初始化数据加载器
    dataloader = MMDataLoader(args, num_workers)
    
    print("training for EMOE")

    # 设置低层梯度大小和损失权重
    args.gd_size_low = 64 
    args.w_losses_low = [1, 10]
    args.metric_low = 'l1' # L1损失

    # 设置高层梯度大小和损失权重
    args.gd_size_high = 32
    args.w_losses_high = [1, 10]
    args.metric_high = 'l1' # L1损失

    from_idx = [0, 1, 2]
    assert len(from_idx) >= 1

    model = getattr(emoe, 'EMOE')(args).cuda()

    trainer = ATIO().getTrain(args)

    # 测试
    if args.mode == 'test':
        model.load_state_dict(torch.load('pt/mosi-aligned.pth')) # 加载预训练模型
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    # 训练
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model.load_state_dict(torch.load('pt/emoe.pth'))

        results = trainer.do_test(model, dataloader['test'], mode="TEST", f=1)

        # 清理内存
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results