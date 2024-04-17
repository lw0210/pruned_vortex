import argparse
import glob
import json
import os
import random
import subprocess

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from vortx import collate, data, lightningmodel, utils


# import torch_pruning as tp
import torch.nn.utils.prune as prune


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # if training needs to be resumed from checkpoint,
    # it is helpful to change the seed so that
    # the same data augmentations are not re-used
    pl.seed_everything(config["seed"])
    
    logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config)
    
    subprocess.call(
        [
            "zip",
            "-q",
            os.path.join(logger.experiment.dir, "code.zip"),
            "config.yml",
            *glob.glob("vortx/*.py"),
            *glob.glob("scripts/*.py"),
        ]
    )
    
    ckpt_dir = os.path.join(logger.experiment.dir, "ckpts")
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=ckpt_dir,
        verbose=True,
        save_top_k=2,
        monitor="val/loss",
    )
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"])]
    
    if config["use_amp"]:
        amp_kwargs = {"precision": 16}
    else:
        amp_kwargs = {}
    
    model = lightningmodel.LightningModel(config)
    print('###############################')
    print(model)
    parameters_to_prune = (##########全局剪（可以）     #用的是这个剪的2dcnn
        (model.vortx.cnn2d.conv0[3], 'weight'),
        (model.vortx.cnn2d.conv0[6], 'weight'),
        (model.vortx.cnn2d.conv0[8][0].layers[3], 'weight'),
        (model.vortx.cnn2d.conv0[8][1].layers[3], 'weight'),
        (model.vortx.cnn2d.conv0[8][2].layers[3], 'weight'),
        (model.vortx.cnn2d.conv1[0].layers[3], 'weight'),
        (model.vortx.cnn2d.conv1[1].layers[3], 'weight'),
        (model.vortx.cnn2d.conv1[2].layers[3], 'weight'),
        (model.vortx.cnn2d.conv2[0].layers[3], 'weight'),
        (model.vortx.cnn2d.conv2[1].layers[3], 'weight'),
        (model.vortx.cnn2d.conv2[2].layers[3], 'weight'))
    #########     (model.vortx.cnns3d.fine.stage1[0].net[0], 'kernel'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=0.25)  # 全局剪枝（global_unstructured）
    #
    # total_params = 0
    # pruned_params = 0
    # for name, module in model.vortx.cnn2d.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         total_params += module.weight.nelement()
    #         pruned_params += torch.sum(module.weight == 0).item()
    #
    # sparsity = pruned_params / total_params
    # print(total_params)
    # print(pruned_params)
    # print("剪枝后的模型稀疏度：{:.2f}%".format(sparsity * 100))
    # print("剪枝后的2dcnn参数数量：{} / {}".format(pruned_params, total_params))


    # prune.ln_structured(model.vortx.cnns3d.coarse.stage1[0].net[0], name="kernel", amount=0.25, n=2, dim=0)# #用的是这个剪的3dcnn
    # prune.ln_structured(model.vortx.cnns3d.coarse.stage1[1].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.coarse.stage1[2].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.coarse.stage2[0].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.coarse.stage2[1].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.coarse.stage2[2].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.medium.stage1[0].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.medium.stage1[1].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.medium.stage1[2].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.medium.stage1[0].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.medium.stage1[1].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.medium.stage1[2].net[0], name="kernel", amount=0.25, n=2, dim=0)
    #
    # prune.ln_structured(model.vortx.cnns3d.fine.stage1[0].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.fine.stage1[1].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.fine.stage1[2].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.fine.stage1[0].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.fine.stage1[1].net[0], name="kernel", amount=0.25, n=2, dim=0)
    # prune.ln_structured(model.vortx.cnns3d.fine.stage1[2].net[0], name="kernel", amount=0.25, n=2, dim=0)

    # total_params = 0
    # pruned_params = 0
    # for p in model.vortx.cnns3d.parameters():
    #     total_params += p.numel()
    #     pruned_params += torch.sum(p==0).item()
    #
    # # sparsity = pruned_params / total_params
    # print(total_params)
    # print(pruned_params)
    # print("剪枝后的模型稀疏度：{:.2f}%".format(sparsity * 100))
    # print("剪枝后的3dcnn参数数量：{} / {}".format(pruned_params, total_params))




    # for name, module in model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp.named_modules():###############全链接层剪枝（ok）
    # ##############如model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp中有fc1和fc2   用的就是这个剪
    #     # 对模型中所有的全链接执行ln_unstructured剪枝操作, 选取20%的参数剪枝
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
    #         prune.remove(module, 'weight')######应该要加一下吧
    # for name, module in model.vortx.mv_fusion.coarse.transformer.encoder_layers[1].mlp.named_modules():###############全链接层剪枝（ok）
    # ##############如model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp中有fc1和fc2
    #     # 对模型中所有的全链接执行ln_unstructured剪枝操作, 选取20%的参数剪枝
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
    #         prune.remove(module, 'weight')######应该要加一下吧
    # for name, module in model.vortx.mv_fusion.medium.transformer.encoder_layers[0].mlp.named_modules():###############全链接层剪枝（ok）
    # ##############如model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp中有fc1和fc2
    #     # 对模型中所有的全链接执行ln_unstructured剪枝操作, 选取20%的参数剪枝
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
    #         prune.remove(module, 'weight')######应该要加一下吧
    # for name, module in model.vortx.mv_fusion.medium.transformer.encoder_layers[1].mlp.named_modules():###############全链接层剪枝（ok）
    # ##############如model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp中有fc1和fc2
    #     # 对模型中所有的全链接执行ln_unstructured剪枝操作, 选取20%的参数剪枝
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
    #         prune.remove(module, 'weight')######应该要加一下吧
    # for name, module in model.vortx.mv_fusion.fine.transformer.encoder_layers[0].mlp.named_modules():###############全链接层剪枝（ok）
    # ##############如model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp中有fc1和fc2
    #     # 对模型中所有的全链接执行ln_unstructured剪枝操作, 选取20%的参数剪枝
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
    #         prune.remove(module, 'weight')######应该要加一下吧
    # for name, module in model.vortx.mv_fusion.fine.transformer.encoder_layers[1].mlp.named_modules():###############全链接层剪枝（ok）
    # ##############如model.vortx.mv_fusion.coarse.transformer.encoder_layers[0].mlp中有fc1和fc2
    #     # 对模型中所有的全链接执行ln_unstructured剪枝操作, 选取20%的参数剪枝
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
    #         prune.remove(module, 'weight')######应该要加一下吧
    # #
    # # 统计剪枝后的模型的稀疏度和参数数量
    # total_params = 0
    # pruned_params = 0
    #
    # for name, module in model.vortx.mv_fusion.fine.transformer.encoder_layers[1].mlp.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         total_params += module.weight.nelement()
    #         pruned_params += torch.sum(module.weight == 0).item()
    # #
    # sparsity = pruned_params / total_params
    # print(total_params)
    # print(pruned_params)
    # print("剪枝后的模型稀疏度：{:.2f}%".format(sparsity * 100))
    # print("剪枝后的参数数量：{} / {}".format(pruned_params, total_params))
    # pruned_params = 0
    #
    # for name, module in model.vortx.mv_fusion.fine.transformer.encoder_layers[1].mlp.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         total_params += module.weight.nelement()
    #         pruned_params += torch.sum(module.weight == 0).item()
    #
    # sparsity = pruned_params / total_params
    # print("剪枝后的模型稀疏度：{:.2f}%".format(sparsity * 100))
    # print("剪枝后的fc量：{} / {}".format(pruned_params, total_params))


#t统计总数
    # total_params1 = 0
    # pruned_params1 = 0
    # for p in model.vortx.parameters():
    #     # total_params += torch.sum(p == 0).item()
    #     total_params1 += p.numel()
    #     pruned_params1 += torch.sum(p == 0).item()
    #
    # # sparsity = pruned_params / total_params
    # print(total_params1)
    # print(pruned_params1)
##############################################################################################################################


    print('#############################################')
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        benchmark=True,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"],
        check_val_every_n_epoch=5,
        detect_anomaly=True,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,  # a hack so batch size can be adjusted for fine tuning
        **amp_kwargs,
    )
    trainer.fit(model, ckpt_path=config["ckpt"])
