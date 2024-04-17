import argparse
import json
import os
import yaml

import imageio
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import skimage.measure
import torch
import torchsparse
import tqdm

from vortx import data, lightningmodel, utils



import torch.nn.utils.prune as prune


def load_model(ckpt_file, use_proj_occ, config):
    model = lightningmodel.LightningModel.load_from_checkpoint(
        ckpt_file,
        config=config,
    )



#########################################################
    # print(model)
    # parameters_to_prune = (  ##########全局剪（可以）     #用的是这个剪的2dcnn
    #     (model.vortx.cnn2d.conv0[3], 'weight'),
    #     (model.vortx.cnn2d.conv0[6], 'weight'),
    #     (model.vortx.cnn2d.conv0[8][0].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv0[8][1].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv0[8][2].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv1[0].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv1[1].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv1[2].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv2[0].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv2[1].layers[3], 'weight'),
    #     (model.vortx.cnn2d.conv2[2].layers[3], 'weight'))
    # #########     (model.vortx.cnns3d.fine.stage1[0].net[0], 'kernel'))
    # prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.25)
    #
    # prune.ln_structured(model.vortx.cnns3d.coarse.stage1[0].net[0], name="kernel", amount=0.25, n=2,
    #                     dim=0)  # #用的是这个剪的3dcnn
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
    #
    #
    #
    #
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

##########################################################

    model.vortx.use_proj_occ = use_proj_occ
    model = model.cuda()
    model = model.eval()
    model.requires_grad_(False)
    return model


def load_scene(info_file):
    with open(info_file, "r") as f:
        info = json.load(f)

    rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
    depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
    pose = np.empty((len(info["frames"]), 4, 4), dtype=np.float32)
    for i, frame in enumerate(info["frames"]):
        pose[i] = frame["pose"]
    intr = np.array(info["intrinsics"], dtype=np.float32)
    return rgb_imgfiles, depth_imgfiles, pose, intr


def get_scene_bounds(pose, intr, imheight, imwidth, frustum_depth):
    frust_pts_img = np.array(
        [
            [0, 0],
            [imwidth, 0],
            [imwidth, imheight],
            [0, imheight],
        ]
    )
    frust_pts_cam = (
        np.linalg.inv(intr) @ np.c_[frust_pts_img, np.ones(len(frust_pts_img))].T
    ).T * frustum_depth
    frust_pts_world = (
        pose @ np.c_[frust_pts_cam, np.ones(len(frust_pts_cam))].T
    ).transpose(0, 2, 1)[..., :3]

    minbound = np.min(frust_pts_world, axis=(0, 1))
    maxbound = np.max(frust_pts_world, axis=(0, 1))
    return minbound, maxbound


def get_tiles(minbound, maxbound, cropsize_voxels_fine, voxel_size_fine):
    cropsize_m = cropsize_voxels_fine * voxel_size_fine

    assert np.all(cropsize_voxels_fine % 4 == 0)
    cropsize_voxels_coarse = cropsize_voxels_fine // 4
    voxel_size_coarse = voxel_size_fine * 4

    ncrops = np.ceil((maxbound - minbound) / cropsize_m).astype(int)
    x = np.arange(ncrops[0], dtype=np.int32) * cropsize_voxels_coarse[0]
    y = np.arange(ncrops[1], dtype=np.int32) * cropsize_voxels_coarse[1]
    z = np.arange(ncrops[2], dtype=np.int32) * cropsize_voxels_coarse[2]
    yy, xx, zz = np.meshgrid(y, x, z)
    tile_origin_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)
    y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)
    z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)
    yy, xx, zz = np.meshgrid(y, x, z)
    base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    tiles = []
    for origin_ind in tile_origin_inds:
        origin = origin_ind * voxel_size_coarse + minbound
        tile = {
            "origin_ind": origin_ind,
            "origin": origin.astype(np.float32),
            "maxbound_ind": origin_ind + cropsize_voxels_coarse,
            "voxel_inds": torch.from_numpy(base_voxel_inds + origin_ind),
            "voxel_coords": torch.from_numpy(
                base_voxel_inds * voxel_size_coarse + origin
            ).float(),
            "voxel_features": torch.empty(
                (len(base_voxel_inds), 0), dtype=torch.float32
            ),
            "voxel_logits": torch.empty((len(base_voxel_inds), 0), dtype=torch.float32),
        }
        tiles.append(tile)
    return tiles


def frame_selection(tiles, pose, intr, imheight, imwidth, n_imgs, rmin_deg, tmin):
    sparsified_frame_inds = np.array(utils.remove_redundant(pose, rmin_deg, tmin))

    if len(sparsified_frame_inds) < n_imgs:
        # after redundant frame removal we can end up with too few frames--
        # add some back in
        avail_inds = list(set(np.arange(len(pose))) - set(sparsified_frame_inds))
        n_needed = n_imgs - len(sparsified_frame_inds)
        extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)
        selected_frame_inds = np.concatenate((sparsified_frame_inds, extra_inds))
    else:
        selected_frame_inds = sparsified_frame_inds

    for i, tile in enumerate(tiles):
        if len(selected_frame_inds) > n_imgs:
            sample_pts = tile["voxel_coords"].numpy()
            cur_frame_inds, score = utils.frame_selection(
                pose[selected_frame_inds],
                intr,
                imwidth,
                imheight,
                sample_pts,
                tmin,
                rmin_deg,
                n_imgs,
            )
            tile["frame_inds"] = selected_frame_inds[cur_frame_inds]
        else:
            tile["frame_inds"] = selected_frame_inds
    return tiles


def get_img_feats(vortx, imheight, imwidth, proj_mats, rgb_imgfiles, cam_positions):
    imsize = np.array([imheight, imwidth])
    dims = {
        "coarse": imsize // 16,
        "medium": imsize // 8,
        "fine": imsize // 4,
    }
    feats_2d = {
        "coarse": torch.empty(
            (1, len(rgb_imgfiles), 80, *dims["coarse"]), dtype=torch.float16
        ),
        "medium": torch.empty(
            (1, len(rgb_imgfiles), 40, *dims["medium"]), dtype=torch.float16
        ),
        "fine": torch.empty(
            (1, len(rgb_imgfiles), 24, *dims["fine"]), dtype=torch.float16
        ),
    }
    cam_positions = torch.from_numpy(cam_positions).cuda()[None]
    for i in range(len(rgb_imgfiles)):
        rgb_img = data.load_rgb_imgs([rgb_imgfiles[i]], imheight, imwidth)
        rgb_img = torch.from_numpy(rgb_img).cuda()[None]
        cur_proj_mats = {k: v[:, i, None] for k, v in proj_mats.items()}
        cur_feats_2d = model.vortx.get_img_feats(
            rgb_img, cur_proj_mats, cam_positions[:, i, None]
        )
        for resname in feats_2d:
            feats_2d[resname][0, i] = cur_feats_2d[resname][0, 0].cpu()
    return feats_2d


def inference(model, info_file, outfile, n_imgs, cropsize):#model, info_file, outfile, args.n_imgs, cropsize
    rgb_imgfiles, depth_imgfiles, pose, intr = load_scene(info_file)
    test_img = imageio.imread(rgb_imgfiles[0])
    imheight, imwidth, _ = test_img.shape

    scene_minbound, scene_maxbound = get_scene_bounds(
        pose, intr, imheight, imwidth, frustum_depth=4
    )

    pose_w2c = np.linalg.inv(pose)
    tiles = get_tiles(
        scene_minbound,
        scene_maxbound,
        cropsize_voxels_fine=np.array(cropsize),
        voxel_size_fine=0.04,
    )

    # pre-select views for each tile
    tiles = frame_selection(
        tiles, pose, intr, imheight, imwidth, n_imgs=n_imgs, rmin_deg=15, tmin=0.1
    )

    # drop the frames that weren't selected for any tile, re-index the selected frame indicies
    selected_frame_inds = np.unique(
        np.concatenate([tile["frame_inds"] for tile in tiles])
    )
    all_frame_inds = np.arange(len(pose))
    frame_reindex = np.full(len(all_frame_inds), 100_000)
    frame_reindex[selected_frame_inds] = np.arange(len(selected_frame_inds))
    for tile in tiles:
        tile["frame_inds"] = frame_reindex[tile["frame_inds"]]
    pose_w2c = pose_w2c[selected_frame_inds]
    pose = pose[selected_frame_inds]
    rgb_imgfiles = np.array(rgb_imgfiles)[selected_frame_inds]

    factors = np.array([1 / 16, 1 / 8, 1 / 4])
    proj_mats = data.get_proj_mats(intr, pose_w2c, factors)
    proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}
    img_feats = get_img_feats(
        model,
        imheight,
        imwidth,
        proj_mats,
        rgb_imgfiles,
        cam_positions=pose[:, :3, 3],
    )
    for resname, res in model.vortx.resolutions.items():

        # populate feature volume independently for each tile
        for tile in tiles:
            voxel_coords = tile["voxel_coords"].cuda()
            voxel_batch_inds = torch.zeros(
                len(voxel_coords), dtype=torch.int64, device="cuda"
            )

            cur_img_feats = img_feats[resname][:, tile["frame_inds"]].cuda()
            cur_proj_mats = proj_mats[resname][:, tile["frame_inds"]]

            featheight, featwidth = img_feats[resname].shape[-2:]
            bp_uv, bp_depth, bp_mask = model.vortx.project_voxels(
                voxel_coords,
                voxel_batch_inds,
                cur_proj_mats.transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data = {
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }
            bp_feats, proj_occ_logits = model.vortx.back_project_features(
                bp_data,
                cur_img_feats.transpose(0, 1),
                model.vortx.mv_fusion[resname],
            )
            bp_feats = model.vortx.layer_norms[resname](bp_feats)

            tile["voxel_features"] = torch.cat(
                (tile["voxel_features"], bp_feats.cpu(), tile["voxel_logits"]),
                dim=-1,
            )

        # combine all tiles into one sparse tensor & run convolution
        voxel_inds = torch.cat([tile["voxel_inds"] for tile in tiles], dim=0)
        voxel_batch_inds = torch.zeros((len(voxel_inds), 1), dtype=torch.int32)
        voxel_features = torchsparse.SparseTensor(
            torch.cat([tile["voxel_features"] for tile in tiles], dim=0).cuda(),
            torch.cat([voxel_inds, voxel_batch_inds], dim=-1).cuda(),
        )

        voxel_features = model.vortx.cnns3d[resname](voxel_features)
        voxel_logits = model.vortx.output_layers[resname](voxel_features)

        if resname in ["coarse", "medium"]:
            # sparsify & upsample
            occupancy = voxel_logits.F.squeeze(1) > 0
            if not torch.any(occupancy):
                raise Exception("um")
            voxel_features = model.vortx.upsampler.upsample_feats(
                voxel_features.F[occupancy]
            )
            voxel_inds = model.vortx.upsampler.upsample_inds(voxel_logits.C[occupancy])
            voxel_logits = model.vortx.upsampler.upsample_feats(
                voxel_logits.F[occupancy]
            )
            voxel_features = voxel_features.cpu()
            voxel_inds = voxel_inds.cpu()
            voxel_logits = voxel_logits.cpu()

            # split back up into tiles
            for tile in tiles:
                tile["origin_ind"] *= 2
                tile["maxbound_ind"] *= 2

                tile_voxel_mask = (
                    (voxel_inds[:, 0] >= tile["origin_ind"][0])
                    & (voxel_inds[:, 1] >= tile["origin_ind"][1])
                    & (voxel_inds[:, 2] >= tile["origin_ind"][2])
                    & (voxel_inds[:, 0] < tile["maxbound_ind"][0])
                    & (voxel_inds[:, 1] < tile["maxbound_ind"][1])
                    & (voxel_inds[:, 2] < tile["maxbound_ind"][2])
                )

                tile["voxel_inds"] = voxel_inds[tile_voxel_mask, :3]
                tile["voxel_features"] = voxel_features[tile_voxel_mask]
                tile["voxel_logits"] = voxel_logits[tile_voxel_mask]
                tile["voxel_coords"] = tile["voxel_inds"] * (
                    res / 2
                ) + scene_minbound.astype(np.float32)

    tsdf_vol = utils.to_vol(
        voxel_logits.C[:, :3].cpu().numpy(),
        1.05 * torch.tanh(voxel_logits.F).squeeze(-1).cpu().numpy(),
    )
    mesh = utils.to_mesh(
        -tsdf_vol,
        voxel_size=0.04,
        origin=scene_minbound,
        level=0,
        mask=~np.isnan(tsdf_vol),
    )
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt", required=True)
    # parser.add_argument("--split", required=True)
    # parser.add_argument("--outputdir", required=True)
    # parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default='../ckpt/2dcnn.ckpt')
    # parser.add_argument("--ckpt", default='ckpt/checkpoint.ckpt')
    parser.add_argument("--split", default= 'test')
    # parser.add_argument("--outputdir", required='output')#output_dire
    parser.add_argument("--outputdir", default='../output_dire')
    # parser.add_argument("--config", required=True)
    parser.add_argument("--config", default='../config.yml')
    parser.add_argument("--use-proj-occ", default=True, type=bool)
    parser.add_argument("--n-imgs", default=60, type=int)
    parser.add_argument("--cropsize", default=96, type=int)
    args = parser.parse_args()

    pl.seed_everything(0)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cropsize = (args.cropsize, args.cropsize, 48)

    with torch.cuda.amp.autocast():

        info_files = utils.load_info_files(config["scannet_dir"], args.split)
        model = load_model(args.ckpt, args.use_proj_occ, config)
        for info_file in tqdm.tqdm(info_files):

            scene_name = os.path.basename(os.path.dirname(info_file))
            outdir = os.path.join(args.outputdir, scene_name)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, "prediction.ply")

            if os.path.exists(outfile):
                print(outfile, 'exists, skipping')
                continue

            try:
                mesh = inference(model, info_file, outfile, args.n_imgs, cropsize)
                o3d.io.write_triangle_mesh(outfile, mesh)
            except Exception as e:
                print(e)
