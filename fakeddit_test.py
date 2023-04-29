import matplotlib.pyplot as plt
import os
import tarfile
from PIL import Image
import io
import numpy as np


tar = tarfile.open("../../public_images.tar.bz2")
print(len(list(tar)))
exit()

for id, member in enumerate(tar):
    # Folderu nie chcemy
    if id > 0:
        print(member.name)
        image=tar.extractfile(member)
        image = image.read()
        image = np.asarray(Image.open(io.BytesIO(image)))
        print(image.shape)
        plt.imshow(image)
        plt.savefig("fake.png")
        if id == 10:
            exit()