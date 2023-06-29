### Facial segmentation mask estimation
import cv2
import numpy as np
import torch
import torchvision
from mmseg.apis import inference_model, init_model
from PIL import Image
from src.datasets.dataset import (
    __celebAHQ_masks_to_faceParser_mask_detailed,
    __ffhq_masks_to_faceParser_mask_detailed,
)
from src.pretrained.face_parsing.model import BiSeNet, seg_mean, seg_std
from torch import nn
from torch.nn import functional as F


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.0:
            return (
                (a + 2.0) * torch.pow(abs_x, 3.0)
                - (a + 3.0) * torch.pow(abs_x, 2.0)
                + 1
            )
        elif 1.0 < abs_x < 2.0:
            return (
                a * torch.pow(abs_x, 3)
                - 5.0 * a * torch.pow(abs_x, 2.0)
                + 8.0 * a * abs_x
                - 4.0 * a
            )
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor(
            [
                self.bicubic_kernel(
                    (i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor
                )
                for i in range(size)
            ],
            dtype=torch.float32,
        )
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        # x = torch.from_numpy(x).type('torch.FloatTensor')
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
        filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(x, 2, 3), 1, 2)  # NHWC to NCHW

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.0)

        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.0)

        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x


def vis_parsing_maps(image, parsing_anno, stride=1):
    """Visualize the seg map, along with the original RGB image

    args:
        img (PIL.Image): [0, 255] PIL.Image
        parsing_anno (np.array): seg map, size [512, 512]
    return:
        vis_im (np.array): visualization image, cv2 format
    """
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255],
    ]
    # parsing_anno = np.transpose(parsing_anno, (1, 2, 0))
    # print(55555555, np.max(parsing_anno), parsing_anno.shape, image.size)

    # im = image.resize((parsing_anno.shape[0], parsing_anno.shape[1]), Image.BILINEAR)
    im = cv2.resize(
        image,
        (parsing_anno.shape[1], parsing_anno.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    # im = np.array(im)

    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    print(6666666666, np.max(vis_parsing_anno))
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST
    )
    print(77777777, np.max(vis_parsing_anno))
    vis_parsing_anno_color = (
        np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    )

    num_of_class = np.max(vis_parsing_anno)

    print('num_of_class=========', num_of_class)
    print(len(part_colors))

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    print('xxxxxxxxx', vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(
        cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0
    )

    return vis_im


class FaceParser(nn.Module):
    def __init__(self, seg_ckpt, size=1024, device="cuda"):
        super(FaceParser, self).__init__()
        self.seg_ckpt = seg_ckpt
        self.size = size
        self.device = device

        self.load_segmentation_network()
        self.load_downsampling()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.size // 256)

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=19)
        self.seg.to(self.device)

        self.seg.load_state_dict(torch.load(self.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def preprocess_img(self, img):
        img_orig = img
        if img_orig.size[0] >= 512:
            im = (
                torchvision.transforms.ToTensor()(img_orig)[:3]
                .unsqueeze(0)
                .to(self.device)
            )
            im = (self.downsample(im).clamp(0, 1) - seg_mean) / seg_std
        else:
            im = img_orig.resize((512, 512), Image.BILINEAR)
            im = torchvision.transforms.ToTensor()(im)[:3].unsqueeze(0).to(self.device)
            im = (im.clamp(0, 1) - seg_mean) / seg_std
        return im

    def forward(self, img):
        """To esitimate the facial mask for the given image
        Args:
            img (PIL.Image): [0, 255] PIL.Image
        """
        im = self.preprocess_img(img)
        down_seg, _, _ = self.seg(im)
        seg = torch.argmax(down_seg, dim=1)[0].long()

        # print(np.unique(seg))
        # cv2.imwrite(os.path.join("./tmp", "mask"+os.path.basename(img_path)), seg.astype(np.uint8))
        # vis_parsing_maps(img_path, seg, stride=1, save_im=True, save_path=os.path.join("./tmp", os.path.basename(img_path)))

        return seg


# ===============================================
def init_faceParsing_pretrained_model(ckpt_path, config_path="", device='cuda:0'):
    parser = init_model(config_path, ckpt_path, device)

    return parser


def faceParsing_demo(model, bgr_img, convert_to_seg12=True):
    """
    args:
        model (Object): Loaded pretrained model
        img (PIL.Image): [0, 255] PIL.Image
    """
    with torch.no_grad():
        # bgr_img = np.array(img)[:, :, ::-1]
        # bgr_img = torch.from_numpy(bgr_img)
        # xx = inference_model(model, [bgr_img])
        # print(xx)
        seg = inference_model(model, [bgr_img])[0].pred_sem_seg.data

        print('sssssssssss', seg.shape, seg)

        seg = np.uint8(seg.numpy())
        print('111111112222224444444', seg.shape, np.max(seg))

        if convert_to_seg12:
            seg = __celebAHQ_masks_to_faceParser_mask_detailed(seg)

        seg = np.uint8(seg)
        print('1111111122222', seg.shape, np.max(seg))

    return seg
