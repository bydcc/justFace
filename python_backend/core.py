import copy
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.datasets.dataset import NORMALIZE, TO_TENSOR
from src.models.networks import Net3
from src.pretrained.face_parsing.face_parsing_demo import (
    faceParsing_demo,
    init_faceParsing_pretrained_model,
    vis_parsing_maps,
)
from src.pretrained.face_vid2vid.driven_demo import (
    drive_source_demo,
    init_facevid2vid_pretrained_model,
)
from src.pretrained.gpen.gpen_demo import GPEN_demo, init_gpen_pretrained_model
from src.utils import torch_utils
from src.utils.alignment import Alignment
from src.utils.morphology import dilation, erosion
from src.utils.multi_band_blending import blending
from src.utils.swap_face_mask import swap_head_mask_revisit_considerGlass
from torch.nn import functional as F

base_path = os.getcwd()


def create_masks(mask, outer_dilation=0):
    radius = outer_dilation
    temp = copy.deepcopy(mask)

    full_mask = dilation(
        temp,
        torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
        engine='convolution',
    )
    erosion_mask = erosion(
        temp,
        torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
        engine='convolution',
    )
    border_mask = full_mask - erosion_mask

    border_mask = border_mask.clip(0, 1)
    content_mask = mask

    return content_mask, border_mask, full_mask


def swap_comp_style_vector(
    style_vectors1, style_vectors2, comp_indices=[], belowFace_interpolation=False
):
    """Replace the style_vectors1 with style_vectors2

    Args:
        style_vectors1 (Tensor): with shape [1,#comp,512], style vectors of target image
        style_vectors2 (Tensor): with shape [1,#comp,512], style vectors of source image
    """
    assert comp_indices is not None

    style_vectors = copy.deepcopy(style_vectors1)

    for comp_idx in comp_indices:
        style_vectors[:, comp_idx, :] = style_vectors2[:, comp_idx, :]

    # if no ear(7) region for source
    if torch.sum(style_vectors2[:, 7, :]) == 0:
        style_vectors[:, 7, :] = (style_vectors1[:, 7, :] + style_vectors2[:, 7, :]) / 2

    # if no teeth(9) region for source
    if torch.sum(style_vectors2[:, 9, :]) == 0:
        style_vectors[:, 9, :] = style_vectors1[:, 9, :]

    # # use the ear_rings(11) of target
    # style_vectors[:,11,:] = style_vectors1[:,11,:]

    # neck(8) interpolation
    if belowFace_interpolation:
        style_vectors[:, 8, :] = (style_vectors1[:, 8, :] + style_vectors2[:, 8, :]) / 2

    return style_vectors


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


class TT:
    def __init__(self):
        self.device = 'cpu'  # "cuda:0"
        self.net_params = {
            "num_seg_cls": 12,
            "train_G": False,
            "device": self.device,
            "lap_bld": True,
            # ================ Model":====================
            "out_size": 1024,
            "fsencoder_type": 'psp',
            "remaining_layer_idx": 13,
            "outer_dilation": 15,
            "erode_radius": 3,
            # ================= Pre-trained Models =================
            "learn_in_w": False,
            "start_from_latent_avg": True,
            "output_size": 1024,
            "n_styles": 18,
            "checkpoint_path": os.path.join(
                base_path, './src/pretrained_ckpts/e4s/iteration_300000.pt'
            ),
        }

        # face_vid2vid
        self.face_vid2vid_params = {
            "face_vid2vid_cfg": os.path.join(
                base_path, './src/pretrained_ckpts/facevid2vid/vox-256.yaml'
            ),
            "face_vid2vid_ckpt": os.path.join(
                base_path,
                './src/pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar',
            ),
        }

        # face parser segnext
        self.face_parser_ckpt = os.path.join(
            base_path,
            # './src/pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth',
            './src/pretrained_ckpts/face_parsing/model_changed.pth',
        )
        self.face_parser_config_path = os.path.join(
            base_path,
            './src/pretrained_ckpts/face_parsing/segnext.small.512x512.celebamaskhq.160k.py',
        )

        # gpen
        self.gpen_model_params = {
            "base_dir": os.path.join(
                base_path, './src/pretrained_ckpts/gpen/'
            ),  # a sub-folder named <weights> should exist
            "in_size": 512,
            "model": "GPEN-BFR-512",
            "use_sr": True,
            "sr_model": "realesrnet",
            "sr_scale": 4,
            "channel_multiplier": 2,
            "narrow": 1,
            'device': self.device,
        }
        self.init_model()
        self.load_source_image('./target.jpg')

    def init_model(self):
        self.alignment = Alignment()

        # face_vid2vid
        (
            self.generator,
            self.kp_detector,
            self.he_estimator,
            self.estimate_jacobian,
        ) = init_facevid2vid_pretrained_model(
            self.face_vid2vid_params['face_vid2vid_cfg'],
            self.face_vid2vid_params['face_vid2vid_ckpt'],
            cpu=self.device == 'cpu',
        )

        # GPEN
        self.GPEN_model = init_gpen_pretrained_model(self.gpen_model_params)

        # face parser
        self.face_parsing_model = init_faceParsing_pretrained_model(
            self.face_parser_ckpt, self.face_parser_config_path, device=self.device
        )
        print("Load pre-trained face parsing models success!")

        # E4S model
        self.net = Net3(self.net_params)
        self.net = self.net.to(self.net_params['device'])
        save_dict = torch.load(
            self.net_params['checkpoint_path'], map_location=torch.device(self.device)
        )
        self.net.load_state_dict(
            torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module.")
        )
        self.net.latent_avg = save_dict['latent_avg'].to(self.net_params['device'])
        print("Load E4S pre-trained model success!")

    def load_source_image(self, file_path):
        # 要替换成谁
        self.source_BGR = cv2.imread(file_path)

        # self.source_RGB = cv2.cvtColor(self.source_BGR, cv2.COLOR_BGR2RGB)
        self.source_RGB = self.source_BGR[:, :, ::-1]
        # self.S_256_RGB = resize(self.source_RGB / 255.0, (256, 256))

        crop_face_output_size = 1024
        crops, quads = self.alignment.crop_faces(
            crop_face_output_size,
            [self.source_RGB],
        )
        crops = [crop.convert("RGB") for crop in crops]

        S_RGB = np.array(crops[0])
        self.S_256_RGB = cv2.resize(
            S_RGB / 255.0, (256, 256), interpolation=cv2.INTER_CUBIC
        )

    @torch.no_grad()
    def faceSwapping_pipeline(
        self,
        target_BGR,
    ):
        """
        The overall pipeline of face swapping:

            Input: source image, target image

            (1) Crop the faces from the source and target and align them, obtaining S and T ;
            (2) Use faceVid2Vid & GPEN to re-enact S, resulting in driven face D, and then parsing the mask of D
            (3) Extract the texture vectors of D and T using RGI
            (4) Texture and shape swapping between face D and face T
            (5) Feed the swapped mask and texture vectors to the generator, obtaining swapped face I;
            (6) Stich I back to the target image

        Args:
            target : target
        """
        target_RGB = target_BGR[:, :, ::-1]
        # (1) Crop the faces from the source and target and align them, obtaining S and T
        crop_face_output_size = 1024
        crops, quads = self.alignment.crop_faces(
            crop_face_output_size,
            [target_RGB],
        )

        inv_transforms = [
            self.alignment.calc_alignment_coefficients(
                quad + 0.5,
                [
                    [0, 0],
                    [0, crop_face_output_size],
                    [crop_face_output_size, crop_face_output_size],
                    [crop_face_output_size, 0],
                ],
            )
            for quad in quads
        ]

        crops = [crop.convert("RGB") for crop in crops]

        T_RGB = np.array(crops[0])
        T_BGR = T_RGB[:, :, ::-1]
        cv2.imwrite('./D_mask.jpg', T_BGR)

        # T_256 = resize(T / 255.0, (256, 256))
        T_256_RGB = cv2.resize(T_RGB / 255.0, (256, 256), interpolation=cv2.INTER_CUBIC)

        # (2) faceVid2Vid  input & output [0,1] range with RGB
        predictions = drive_source_demo(
            self.S_256_RGB,
            [T_256_RGB],
            self.generator,
            self.kp_detector,
            self.he_estimator,
            self.estimate_jacobian,
            cpu=self.device == 'cpu',
        )
        predictions = [(pred * 255).astype(np.uint8) for pred in predictions]

        # (2) GPEN input & output [0,255] range with BGR
        drivens = [
            GPEN_demo(pred[:, :, ::-1], self.GPEN_model, aligned=False)
            for pred in predictions
        ]

        # (2) mask of D and T
        D_BGR = drivens[0]
        D_RBG = D_BGR[:, :, ::-1]
        cv2.imwrite('./D.jpg', D_BGR)
        D_mask = faceParsing_demo(
            self.face_parsing_model,
            D_BGR,
            convert_to_seg12=True,
        )[0]
        # print('cccccccc', D_mask.shape, D.size, type(D))
        # Image.fromarray(D_mask).save("./D_mask.png")
        cv2.imwrite('./D_mask.jpg', D_mask)
        D_mask_vis = vis_parsing_maps(D_RBG, D_mask)
        # Image.fromarray(D_mask_vis).save("./D_mask_vis.png")
        cv2.imwrite('./D_mask_vis.jpg', D_mask_vis)

        # [0,255] range with RGB
        T_mask = faceParsing_demo(
            self.face_parsing_model,
            T_BGR,
            convert_to_seg12=True,
        )[0]
        print('T_maskT_maskT_mask', T_mask.shape)
        # Image.fromarray(T_mask).save("./T_mask.png")
        cv2.imwrite('./T_mask.jpg', T_mask)
        T_mask_vis = vis_parsing_maps(T_RGB, T_mask)
        # Image.fromarray(T_mask_vis).save("./T_mask_vis.png")
        cv2.imwrite('./T_mask_vis.jpg', T_mask_vis)

        # wrap data
        D_RBG = np.ascontiguousarray(D_RBG)
        driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D_RBG)
        driven = driven.to(self.device).float().unsqueeze(0)
        driven_mask = transforms.Compose([TO_TENSOR])(D_mask)
        driven_mask = (driven_mask * 255).long().to(self.device).unsqueeze(0)
        driven_onehot = torch_utils.labelMap2OneHot(
            driven_mask, num_cls=self.net_params['num_seg_cls']
        )

        print('cccccccc1111', driven.size(), driven_mask.size(), driven_onehot.size())

        target = transforms.Compose([TO_TENSOR, NORMALIZE])(T_RGB)
        target = target.to(self.device).float().unsqueeze(0)
        target_mask = transforms.Compose([TO_TENSOR])(T_mask)
        target_mask = (target_mask * 255).long().to(self.device).unsqueeze(0)
        target_onehot = torch_utils.labelMap2OneHot(
            target_mask, num_cls=self.net_params['num_seg_cls']
        )

        print('cccccccc22222', target.size(), target_mask.size(), target_onehot.size())

        # (3) Extract the texture vectors of D and T using RGI
        driven_style_vector, _ = self.net.get_style_vectors(driven, driven_onehot)
        target_style_vector, _ = self.net.get_style_vectors(target, target_onehot)

        # (4) shape swapping between face D and face T
        swapped_msk, hole_map = swap_head_mask_revisit_considerGlass(D_mask, T_mask)

        print('cccccccc2222244444', swapped_msk.shape, hole_map)

        # Texture swapping between face D and face T. Retain the style_vectors of background(0), hair(4), era_rings(11), eye_glass(10) from target
        comp_indices = set(range(self.net_params['num_seg_cls'])) - {
            0,
            4,
            11,
            10,
        }  # 10 glass, 8 neck
        swapped_style_vectors = swap_comp_style_vector(
            target_style_vector,
            driven_style_vector,
            list(comp_indices),
            belowFace_interpolation=False,
        )

        # (5) Feed the swapped mask and texture vectors to the generator, obtaining swapped face I;
        # swapped_msk = Image.fromarray(swapped_msk[0])
        # print('swapped_mskswapped_mskswapped_msk', swapped_msk.size)
        swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
        print('swapped_mskswapped_mskswapped_msk', swapped_msk.size())
        swapped_msk = (swapped_msk * 255).long().to(self.device).unsqueeze(0)

        print('00000000000', swapped_msk.size())
        swapped_onehot = torch_utils.labelMap2OneHot(
            swapped_msk, num_cls=self.net_params['num_seg_cls']
        )
        print('swapped_onehotswapped_onehotswapped_onehot', swapped_onehot.size())
        #
        swapped_style_codes = self.net.cal_style_codes(swapped_style_vectors)

        print(
            'swapped_style_codesswapped_style_codesswapped_style_codes',
            swapped_style_codes.size(),
        )
        swapped_face, _, _ = self.net.gen_img(
            torch.zeros(1, 512, 32, 32).to(self.device),
            swapped_style_codes,
            swapped_onehot,
        )
        swapped_face_image = torch_utils.tensor2im(swapped_face[0])

        # (6) Stich I back to the target image
        #
        # Gaussian blending with mask
        outer_dilation = 5
        # For face swapping in video，consider 4,8,7 as part of background.
        # 11 earings; 4 hair; 8 neck; 7 ear
        mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0, 11, 4]])
        is_foreground = torch.logical_not(mask_bg)
        hole_index = hole_map[None][None] == 255
        is_foreground[hole_index[None]] = True
        foreground_mask = is_foreground.float()

        content_mask, border_mask, full_mask = create_masks(
            foreground_mask, outer_dilation=outer_dilation
        )

        content_mask = F.interpolate(
            content_mask, (1024, 1024), mode='bilinear', align_corners=False
        )
        full_mask = F.interpolate(
            full_mask, (1024, 1024), mode='bilinear', align_corners=False
        )
        # Paste swapped face onto the target's face
        content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = F.interpolate(
            border_mask, (1024, 1024), mode='bilinear', align_corners=False
        )
        border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = np.repeat(border_mask, 3, axis=-1)

        swapped_and_pasted = swapped_face_image * content_mask + T_RGB * (
            1 - content_mask
        )
        swapped_and_pasted = blending(
            T_RGB, np.uint8(swapped_and_pasted), mask=border_mask
        )

        inv_trans_coeffs = inv_transforms[0]

        projected = (
            Image.fromarray(swapped_and_pasted)
            .convert('RGBA')
            .transform(
                (target_BGR.shape[1], target_BGR.shape[0]),
                Image.PERSPECTIVE,
                inv_trans_coeffs,
                Image.BILINEAR,
            )
        )

        res = Image.fromarray(target_RGB).convert('RGBA')
        res.alpha_composite(projected)
        res.save("./res.png")
        return res


if __name__ == "__main__":
    target_BGR = cv2.imread('./image1.jpg')  # HWC
    # target_BGR = cv2.imread('./target.jpg')
    # print('target_BGR shape', target_BGR.shape)

    # target_RGB = target_BGR[:, :, ::-1]  # HWC

    # Image.fromarray(target_BGR).save("./1.png")

    # c = Image.open('./single.png')
    # arr = np.array(c)
    # print(arr.shape)
    t = TT()
    res = t.faceSwapping_pipeline(target_BGR)
    # res.save('./res.png')
# 思考
# 1. 能否提前计算所有角度的的 driven， 后期根据角度直接拿到最相似的driven，提升速度？
# 2.


# 单脸， camera实时
# 多脸， 加身份识别
