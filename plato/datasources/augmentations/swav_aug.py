"""
The implementation of the SWAV's [1] augmentation function.

[1]. https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py

#     def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops,
#                  normalize):
#         assert len(size_crops) == len(nmb_crops)
#         assert len(min_scale_crops) == len(nmb_crops)
#         assert len(max_scale_crops) == len(nmb_crops)

#         color_transform = [get_color_distortion(), PILRandomGaussianBlur()]

#         trans = []
#         for i, size_crop in enumerate(size_crops):
#             randomresizedcrop = T.RandomResizedCrop(
#                 size_crop,
#                 scale=(min_scale_crops[i], max_scale_crops[i]),
#             )
#             i_transform_funcs = [
#                 randomresizedcrop,
#                 T.RandomHorizontalFlip(p=0.5),
#                 T.Compose(color_transform),
#                 T.ToTensor()
#             ]

#             if normalize is not None:
#                 i_transform_funcs.append(T.Normalize(*normalize))

#             trans.extend([T.Compose(i_transform_funcs)] * nmb_crops[i])

# # class PILRandomGaussianBlur():

#     Apply Gaussian Blur to the PIL image. Take the radius and probability of
#     application as the parameter.
#     This transform was used in SimCLR - https://arxiv.org/abs/2002.05709


#     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max

#     def __call__(self, img):
#         Perform the Gaussian Blur
#         do_it = np.random.rand() <= self.prob
#         if not do_it:
#             return img

#         return img.filter(
#             ImageFilter.GaussianBlur(
#                 radius=random.uniform(self.radius_min, self.radius_max)))


# def get_color_distortion(s=1.0):
#     Get the color_distort
#     # s is the strength of color distortion.
#     color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
#     rnd_gray = T.RandomGrayscale(p=0.2)
#     color_distort = T.Compose([rnd_color_jitter, rnd_gray])
#     return color_distort
"""

from plato.datasources.augmentations.ssl_transform_base import get_ssl_base_transform


class SvAVTransform():
    """ This the contrastive data augmentation used by the SWAV method. """

    def __init__(self, image_size, normalize):
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        self.transform, transform_funcs = get_ssl_base_transform(
            image_size,
            normalize,
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
            color_jitter_prob=0.8,
            gray_scale_prob=0.2,
            horizontal_flip_prob=0.5,
            gaussian_prob=p_blur,
            solarization_prob=0.0,
            equalization_prob=0.0,
            min_scale=0.08,
            max_scale=1.0,
            crop_size=image_size)
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        return self.transform(x)
