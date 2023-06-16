import augment
from PIL import Image

im = Image.open('raw_data/sharks/blue/00000010.jpg')
ims_aug = augment.augment(im, batch=1)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))
ax1.imshow(im)
ax2.imshow(ims_aug[0][0][0].astype(np.uint8))
plt.show()
