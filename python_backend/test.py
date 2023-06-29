# from face_extract import VideoFrameExtractor, FaceExtractor
import torchvision.transforms as transforms
import numpy as np

TO_TENSOR = transforms.ToTensor()


NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

D = np.random.rand(3, 3, 3)
driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D)

print(driven)
# if __name__ == "__main__":
# video_frame_extractor = VideoFrameExtractor()
# video_frame_extractor.extract_frames('.', './imgs')
# face_extractor = FaceExtractor()
# face_extractor.extract_and_save_images('./imgs', './faces')
