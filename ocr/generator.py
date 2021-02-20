from typing import Tuple, List
import pathlib
import io
import random

import numpy as np
import shapely.geometry as geometry
import pyclipper
from PIL import Image, ImageDraw, ImageFont
import tqdm
import tensorflow as tf


FONTS_DIR = pathlib.Path('assets/fonts')
VOCAB_DIR = pathlib.Path('assets/vocab')


FONT_SIZE = (10, 20)
WORDS_COUNT = (3, 300)
FONT_COLOR = (0, 200)
BACKGROUND_COLOR = (230, 255)
SHRINK_RATIO = 0.4

JPEG_FORMAT = 'JPEG'
PNG_FORMAT = 'PNG'


def write_tfrecords(fname: str, size: int, img_size: Tuple[int, int]):
    gen = DatasetGenerator()

    with tf.io.TFRecordWriter(fname) as file_writer:
        for _ in tqdm.tqdm(range(size)):
            try:
                example = create_example(gen, img_size)
                file_writer.write(example.SerializeToString())
            except IndexError:
                continue


# pylint: disable=too-few-public-methods
class DatasetGenerator():

    def __init__(self):
        self.fonts = [str(s) for s in FONTS_DIR.glob('*')]

        self.vocab = []
        for vocab_file in VOCAB_DIR.glob("*.txt"):
            lines = vocab_file.read_text().split()
            self.vocab.extend(lines)

    def create_sample(
        self,
        img_size: Tuple[int, int]) -> Tuple[Image.Image, List[geometry.Polygon]]:

        back_color = random.randint(*BACKGROUND_COLOR)
        img = Image.new(
            mode='RGB',
            size=img_size,
            color=(back_color, back_color, back_color))
        draw = ImageDraw.Draw(img)
        polygons: List[geometry.Polygon] = []

        for _ in range(random.randint(*WORDS_COUNT)):
            fnt = ImageFont.truetype(
                font=random.choice(self.fonts),
                size=random.randint(*FONT_SIZE))

            text = random.choice(self.vocab)
            bbox = fnt.getbbox(text)
            width = bbox[0] + bbox[2]
            height = bbox[1] + bbox[3]

            coord = (
                random.randint(0, img.width - width),
                random.randint(0, img.height - height))
            polygon = geometry.box(
                minx=coord[0],
                miny=coord[1],
                maxx=coord[0] + width,
                maxy=coord[1] + height,
            )
            if geometry.MultiPolygon(polygons).intersects(polygon):
                continue

            polygons.append(polygon)
            ink = random.randint(*FONT_COLOR)
            draw.text(coord, text, font=fnt, fill=(ink, ink, ink))

        return img, polygons


def create_threshold_map(
    polygons: List[geometry.Polygon],
    size: Tuple[int, int]) -> Image.Image:

    threshold_map_accum = Image.new('L', size, color=0)

    for polygon in polygons:
        threshold_mask = create_single_threshold_map(polygon, size)

        threshold_map_accum = Image.composite(
            Image.new('L', size, color=255),
            threshold_map_accum,
            threshold_mask)

    return threshold_map_accum


def create_prob_map(
    polygons: List[geometry.Polygon],
    size: Tuple[int, int]) -> Image.Image:

    prob_map = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(prob_map)

    for polygon in polygons:
        distance = get_distance(polygon)

        padding = pyclipper.PyclipperOffset()
        padding.AddPath(
            polygon.exterior.coords,
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON)

        shrinked = padding.Execute(-distance)[0]
        shrinked = [tuple(c) for c in shrinked]
        draw.polygon(shrinked, fill=255)

    return prob_map


# shrink / dilate distance
def get_distance(polygon: geometry.Polygon) -> float:
    distance = polygon.area * (1 - np.power(SHRINK_RATIO, 2)) / polygon.length
    return distance


def create_single_threshold_map(
    polygon:geometry.Polygon,
    size: Tuple[int, int]) -> Image.Image:

    distance = get_distance(polygon)
    threshold_map = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(threshold_map)

    # Calculate deluted and shrink polys
    padding = pyclipper.PyclipperOffset()

    padding.AddPath(
        polygon.exterior.coords,
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON)
    distance = int(distance)

    for cur_dist in reversed(range(distance)):
        color = 255 * (distance - cur_dist) / distance
        color = int(color)

        dilated = padding.Execute(cur_dist)[0]
        dilated = [tuple(c) for c in dilated]
        draw.polygon(dilated, fill=color)

    for cur_dist in range(int(distance)):
        color = 255 * (distance - cur_dist) / distance
        color = int(color)
        shrinked = padding.Execute(-cur_dist)
        shrinked = shrinked[0]  # we have only 1 polygon
        shrinked = [tuple(c) for c in shrinked]
        draw.polygon(shrinked, fill=color)

    shrinked = padding.Execute(-distance-1)[0]
    shrinked = [tuple(c) for c in shrinked]
    draw.polygon(shrinked, fill=0)

    return threshold_map


def img2bytes(img: Image.Image, img_format: str) -> bytes:
    with io.BytesIO() as memory_buffer:
        img.save(memory_buffer, img_format)
        return memory_buffer.getvalue()


def img_feature(img: Image.Image, img_format: str) -> tf.train.Feature:
    arr = img2bytes(img, img_format)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr]))


def create_example(
    gen: DatasetGenerator,
    img_size: Tuple[int, int]) -> tf.train.Example:

    img, polygons = gen.create_sample(img_size)
    prob_map = create_prob_map(polygons, img_size)
    threshold_map = create_threshold_map(polygons, img_size)

    return tf.train.Example(features=tf.train.Features(
        feature={
            'img': img_feature(img, JPEG_FORMAT),
            'prob_map': img_feature(prob_map, PNG_FORMAT),
            'threshold_map': img_feature(threshold_map, PNG_FORMAT),
        }))
