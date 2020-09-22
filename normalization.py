import json
import os
from collections import OrderedDict

import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import KMeans


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def get_annotations(data):
    return data['annotations'][0:2000]


def get_images(data):
    return data['images']


def get_keypoints(data, index):
    return data[index]['keypoints']


def get_bbox(data, index):
    return data[index]['bbox']


def crop_image(img_path, bbox, offset):
    im = Image.open(img_path)
    # im.show()
    # rectangle : x y w h -> crop: x y w+x h+y
    x = bbox[0] + bbox[2] / 2 - offset / 2
    y = bbox[1] + bbox[3] / 2 - offset / 2
    cropped_image = im.crop((int(x), int(y), int(x + offset), int(y + offset)))
    cropped_image.save('cropped.jpg')


def show_bounding_box(img_path, bbox):
    im = plt.imread(img_path)
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')


def show_points(img_path, keypoints):
    im = plt.imread(img_path)
    # plt.imshow(im)
    for i in range(0, len(keypoints), 3):
        # print(keypoints[i], keypoints[i + 1], keypoints[i + 2])
        plt.scatter(keypoints[i], keypoints[i + 1], 1, 'b')
    # plt.title(img_path)
    # plt.show()


def get_cropped_keypoints(img_path, bbox, keypoints, offset):
    im = plt.imread(img_path)
    plt.imshow(im)
    cropped_keypoints = []
    x = offset / 2
    y = offset / 2
    x_mid = bbox[0] + bbox[2] / 2
    y_mid = bbox[1] + bbox[3] / 2
    for i in range(0, len(keypoints), 3):
        # print(keypoints[i], keypoints[i + 1], keypoints[i + 2])
        cropped_keypoints.append(keypoints[i] - bbox[0])
        cropped_keypoints.append(keypoints[i + 1] - bbox[1])
        plt.scatter(keypoints[i] - (x_mid - x), keypoints[i + 1] - (y_mid - y), 1, 'r')

    # plt.axis('off')
    # plt.title(img_path)
    # plt.show()
    print(cropped_keypoints)
    return cropped_keypoints


def normalize_vector(x):
    w = np.sqrt(sum(power(x, 2)))
    x_norm = x / w
    print(x_norm)
    return x_norm


def power(my_list, p):
    return [x ** p for x in my_list]


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def compute_similarities():
    offset = 300
    data = read_json('annotations/person_keypoints_train2014.json')
    annotations = get_annotations(data)
    images = get_images(data)
    normalized_vectors = {}

    for annotation in annotations:
        img_id = annotation['image_id']
        img = [image for image in images if image['id'] == img_id][0]
        image_path = 'train2014/' + img['file_name']
        bbox = annotation['bbox']
        show_bounding_box(image_path, bbox)
        crop_image(image_path, bbox, offset)
        keypoints = annotation['keypoints']
        show_points(image_path, keypoints)
        cropped_keypoints = get_cropped_keypoints('cropped.jpg', bbox, keypoints, offset)
        normalized_keypoints = normalize_vector(cropped_keypoints)
        normalized_vectors.update({img['file_name']: normalized_keypoints})
    sorted_vectors = OrderedDict(normalized_vectors.items())
    return sorted_vectors
    # test_file_name = 'COCO_train2014_000000382669.jpg'
    # similarities = {}
    # for vector in normalized_vectors:
    #     similarities.update({vector: cosine_similarity(normalized_vectors[test_file_name], normalized_vectors[vector])})
    # similarity = max(similarities, key=similarities.get)
    # del similarities[similarity]
    # similarity = max(similarities, key=similarities.get)
    # print(similarity)


def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def cluster_images(normalized_vectors):
    print(normalized_vectors)
    vectors = list(normalized_vectors.values())
    X = np.array([vector for vector in vectors])
    kmeans = KMeans(n_clusters=15, random_state=0).fit(X)
    labels = list(kmeans.labels_)
    print(labels)
    # keys = list(normalized_vectors.keys())
    # values = list(normalized_vectors.values())
    for label in unique(labels):
        os.mkdir(str(label))
    for i, x in enumerate(labels):
        print(labels[i])
        print(list(normalized_vectors.items())[i][0])
        file_path = str(labels[i]) + '/' + str(list(normalized_vectors.items())[i][0])
        im = Image.open('train2014/' + str(list(normalized_vectors.items())[i][0]))
        im.save(file_path)
    return labels


if __name__ == '__main__':
    print(torch.cuda.is_available())
    normalized_vectors = compute_similarities()
    cluster_images(normalized_vectors)
