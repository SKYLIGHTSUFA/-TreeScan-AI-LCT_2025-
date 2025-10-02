import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import exifread, os

from config import IMAGENET_MEAN, IMAGENET_STD


def build_transform(input_size=448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image_tensor(image: Image.Image, input_size=448):
    return build_transform(input_size)(image).unsqueeze(0)


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def get_gps_from_exif(image_file):
    """Извлекает GPS из EXIF и возвращает словарь или None"""
    try:
        tmp_name = "temp_image.jpg"
        with open(tmp_name, "wb") as f:
            f.write(image_file.getvalue())
        with open(tmp_name, "rb") as f:
            tags = exifread.process_file(f)
        os.remove(tmp_name)

        if "GPS GPSLatitude" not in tags or "GPS GPSLongitude" not in tags:
            return None

        def to_deg(values):
            d = float(values[0].num) / values[0].den
            m = float(values[1].num) / values[1].den
            s = float(values[2].num) / values[2].den
            return d + m / 60 + s / 3600

        lat = to_deg(tags["GPS GPSLatitude"].values)
        lon = to_deg(tags["GPS GPSLongitude"].values)

        if "GPS GPSLatitudeRef" in tags and str(tags["GPS GPSLatitudeRef"]) == "S":
            lat = -lat
        if "GPS GPSLongitudeRef" in tags and str(tags["GPS GPSLongitudeRef"]) == "W":
            lon = -lon

        return {"latitude": lat, "longitude": lon}
    except Exception:
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        return None


from PIL import Image


def dynamic_preprocess(
    image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """
    Делит изображение на блоки (тайлы) для подачи в модель.
    Можно добавлять уменьшенную мини-копию (thumbnail) для больших изображений.

    Args:
        image: PIL.Image, исходное изображение.
        min_num: минимальное количество тайлов.
        max_num: максимальное количество тайлов.
        image_size: размер одного тайла (обычно 448).
        use_thumbnail: добавлять ли уменьшенную версию всего изображения.

    Returns:
        List[PIL.Image]: список тайлов для модели.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Подбираем количество тайлов по близости к квадрату
    # Можно расширить алгоритм, сейчас просто ограничиваем max_num
    num_tiles = min(max_num, max(min_num, round(aspect_ratio * max_num)))

    # Размер блоков
    tile_w = orig_width // num_tiles
    tile_h = orig_height // num_tiles

    tiles = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            left = i * tile_w
            upper = j * tile_h
            right = (i + 1) * tile_w if i < num_tiles - 1 else orig_width
            lower = (j + 1) * tile_h if j < num_tiles - 1 else orig_height
            tile = image.crop((left, upper, right, lower))
            tile = tile.resize((image_size, image_size), Image.BICUBIC)
            tiles.append(tile)

    if use_thumbnail:
        thumbnail = image.resize((image_size, image_size), Image.BICUBIC)
        tiles.append(thumbnail)

    return tiles
