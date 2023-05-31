import os
import PIL
from PIL import Image

if __name__ == '__main__':
    image_filenames = os.listdir()
    ckpt = 0
    if os.path.exists('checkpoint.txt'):
        with open('checkpoint.txt', 'r') as ckpt_file:
            ckpt = int(ckpt_file.read())

    for i, file in enumerate([image_filenames[ckpt:]]):
        if file == 'error_log.txt' or 'checkpoint.txt':
            continue
        i += ckpt
        error_msg = 'No Errors'
        label = int(os.path.splitext(file)[0][-1])
        try:
            img = Image.open(file)
        except PIL.UnidentifiedImageError:
            error_msg = 'Removing Corrupt file: ' + file + 'Error: 1: Unidentified Image Error'
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f'{file} \n')
            continue

        try:
            mask_label = img.getpixel((img.width - (img.width // 4), img.height // 2))[-1] - 1
        except (OSError, PermissionError) as e:
            error_msg = 'Removing Corrupt file: ' + file + 'Error 2: Image file is Truncated'
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f'{file} \n')
            continue

        if mask_label not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            error_msg = 'Removing Corrupt file: ' + file + 'Error 3: Mask Label Not Valid'
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f'{file} \n')
            continue

        if mask_label != label:
            error_msg = 'Removing Corrupt file: ' + file + ' Error 4: Mask Label does not match the True Label'
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f'{file} \n')
            continue

        print(f"{i}: {file}: {mask_label} {label} {error_msg}")
        img.close()

        # Add checkpointer here
        if i % 100 == 0:
            with open('checkpoint.txt', 'w') as f:
                f.write(str(i))

    for file in open('error_log.txt', 'r').readlines():
        os.remove(file)
