import tensorflow as tf
import glob
from itertools import groupby
from collections import defaultdict


sess = tf.InteractiveSession()

image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")

image_filenames[0:2]

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

image_filename_with_breed = map(lambda filename: (filename.split("/")[2], filename), image_filenames)

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    """
    Append training/testing image datasets to respective dictionaries
    """
    # Enumerate each breed's image and send ~20% of the images to a testing set
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

    # Check that each breed includes at least 18% of the images for testing
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])

    assert round(breed_testing_count / (breed_training_count + breed_testing_count), 2) > 0.18, "Not enough testing images."

    #breed_lst = open('breeds.txt', 'r').read().split('\n')
    #old_dict = dict(enumerate(breed_lst))
    #breed_label_dict = dict(zip(old_dict.values(), old_dict.keys()))

def write_records_file(dataset, record_location):
    """
    Fill a TFRecords file with the images found in `dataset` and include their category.
    """
    writer = None

    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    current_index = 0

    old_dict = dict(enumerate(dataset.keys()))
    breed_label_dict = dict(zip(old_dict.values(), old_dict.keys()))

    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1

            image_file = tf.read_file(image_filename)

            # In ImageNet dogs, there are a few images which TensorFlow doesn't recognize as JPEGs. This
            # try/catch will ignore those images.
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # Converting to grayscale saves processing and memory but isn't required.
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])

            # tf.cast is used here because the resized images are floats but haven't been converted into
            # image floats where an RGB value is between [0,1).
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            image_int = breed_label_dict[breed]
            image_label = breed.encode("utf-8")

            #image_int = tf.cast(imgint2, tf.int32)

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                'int': _int64_feature(image_int)
            }))

            writer.write(example.SerializeToString())
    writer.close()

write_records_file(testing_dataset, "./output/testing-images/testing-image")
write_records_file(training_dataset, "./output/training-images/training-image")
