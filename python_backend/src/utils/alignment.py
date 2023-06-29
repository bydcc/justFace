import face_alignment
import numpy as np
import PIL
import PIL.Image
from scipy.ndimage import gaussian_filter1d


class Alignment:
    def __init__(self, device='cpu') -> None:
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=True, device=device
        )

    def crop_faces(self, output_size, images, scale=1, center_sigma=0.0, xy_sigma=0.0):
        cs, xs, ys = [], [], []
        for image in images:
            lm = self.get_landmark(image)
            if lm is not None:
                c, x, y = self.compute_transform(lm, scale=scale)
                cs.append(c)
                xs.append(x)
                ys.append(y)
        cs = np.stack(cs)
        xs = np.stack(xs)
        ys = np.stack(ys)
        if center_sigma != 0:
            cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

        if xy_sigma != 0:
            xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
            ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

        quads = np.stack(
            [cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1
        )
        quads = list(quads)

        crops, orig_images = self.crop_faces_by_quads(output_size, images, quads)

        return crops, quads

    # def crop_faces(self, output_size, image, scale=1, center_sigma=0.0, xy_sigma=0.0):
    #     cs, xs, ys = [], [], []

    #     landmarks = self.fa.get_landmarks(image, return_bboxes=False)
    #     for lm in landmarks:
    #         c, x, y = self.compute_transform(lm, scale=scale)
    #         cs.append(c)
    #         xs.append(x)
    #         ys.append(y)
    #     cs = np.stack(cs)
    #     xs = np.stack(xs)
    #     ys = np.stack(ys)
    #     if center_sigma != 0:
    #         cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    #     if xy_sigma != 0:
    #         xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
    #         ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    #     quads = np.stack(
    #         [cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1
    #     )
    #     quads = list(quads)

    #     crops, orig_images = self.crop_faces_by_quads(output_size, image, quads)

    #     return crops, quads

    def compute_transform(self, lm, scale=1.0):
        # lm_chin = lm[0:17]  # left-right
        # lm_eyebrow_left = lm[17:22]  # left-right
        # lm_eyebrow_right = lm[22:27]  # left-right
        # lm_nose = lm[27:31]  # top-down
        # lm_nostrils = lm[31:36]  # top-down
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise
        # lm_mouth_inner = lm[60:68]  # left-clockwise
        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

        x *= scale
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        return c, x, y

    def get_landmark(self, image):
        lms = self.fa.get_landmarks(image, return_bboxes=False)
        if len(lms) == 0:
            return None
        return lms[0]

    def crop_faces_by_quads(self, output_size, images, quads):
        orig_images = []
        crops = []
        for quad, image in zip(quads, images):
            if isinstance(image, np.ndarray):
                image = PIL.Image.fromarray(image)
            elif isinstance(image, PIL.Image.Image):
                image = image
            else:
                image = PIL.Image.open(image)
            crop = self.crop_image(image, output_size, quad.copy())
            orig_images.append(image)
            crops.append(crop)
        return crops, orig_images

    def crop_image(self, image, output_size, quad):
        x = (quad[3] - quad[1]) / 2
        qsize = np.hypot(*x) * 2
        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (
                int(np.rint(float(image.size[0]) / shrink)),
                int(np.rint(float(image.size[1]) / shrink)),
            )
            image = image.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, image.size[0]),
            min(crop[3] + border, image.size[1]),
        )
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            image = image.crop(crop)  # (left, upper, right, lower)
            quad -= crop[0:2]
        # Pad.
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - image.size[0] + border, 0),
            max(pad[3] - image.size[1] + border, 0),
        )
        # Transform.
        img = image.transform(
            (output_size, output_size),
            PIL.Image.QUAD,
            (quad + 0.5).flatten(),
            PIL.Image.BILINEAR,
        )
        return img

    @staticmethod
    def calc_alignment_coefficients(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        a = np.matrix(matrix, dtype=float)
        b = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
        return np.array(res).reshape(8)


if __name__ == "__main__":
    from pathlib import Path

    import cv2

    # 定义用于存储图像的列表
    images = []

    # 定义要迭代的文件夹路径
    folder_path = Path('../test_images')

    # 遍历文件夹中的所有图像文件
    for file_path in folder_path.glob('*.png'):
        # 使用cv2.imread()函数读取图像文件
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将读取的图像添加到列表中
        images.append(image)
    alignment = Alignment()
    crops, orig_images, quads = alignment.crop_faces(1023, images)
    for index, c in enumerate(crops):
        c.save(f'{index}.png', 'PNG')
