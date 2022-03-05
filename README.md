# Scene Text Detection with Differentiable Binarization

In 1st half of 2021 I was studying text recognition and tagging problem and was looking for an OCR pipeline which would take an image as an input and
produce tagged text as an output. The paper "[Differentiable Binarization](https://arxiv.org/abs/1911.08947)" looked like a good candidate for the first
step of this pipeline: it would give segments which then could be used to cut the image into smaller pieces.

This implementation is purely for getting better understanding of the paper and benchmarking. Benchmarking is on the list because the project on
my primary job was related to OCR task. The model showed nearly realtime performance on relatively small images.

Here is a [link to the paper](https://arxiv.org/abs/1911.08947).

## Model

Model is implemented in TensorFlow with Keras. The appoach is to split the model into separate blocks (`ImageEncoder` and `FeatureToImage`) which could be used separately.
That's necessary as the model which is used for inference phase is a bit different than the one which is used during trainging phase.
There is a ResNet in sources. I tried to use the ResNet from TF Hub, but at that time it didn't allowed to take only first N layers (and I needed that for DB model).

## Dataset

Dataset is generated using different fonts and image augmentation. See `generate-dataset.py`. Output of this script is a set tfrecord files which are fed into the model during training.
