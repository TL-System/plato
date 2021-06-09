#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import xml.etree.ElementTree as ET


def filter_bad_boxes(boxes_coor):
    filted_boxes = list()
    for box_coor in boxes_coor:
        [xmin, ymin, xmax, ymax] = box_coor
        if xmin < xmax and ymin < ymax:
            filted_boxes.append(box_coor)

    return filted_boxes


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to
    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id,
                                               phrase_type):
            sentence_data['phrases'].append({
                'first_word_index': index,
                'phrase': phrase,
                'phrase_id': p_id,
                'phrase_type': p_type
            })

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def align_anno_sent(image_sents, image_annos):
    """[align the items in annotations and sentences]

    Args:
        image_sents ([list]): [each itme is a dict that contains 'sentence', 'phrases']
        image_annos ([dict]): [contain 'boxes' - a dict presents the phrase_id: box]

    Return:
            aligned_items ([list]): [each itme is a dict that contains the sentence with the corresponding phrases information,
                                    there should have several items because for one image, there are 5 sentences.
                                    Sometimes, some sentences are useless, making the number of items less than 5]
    """
    aligned_items = list()  # each item is a dict
    for sent_info in image_sents:

        img_sent = sent_info["sentence"]
        img_sent_phrases = list()
        img_sent_phrases_type = list()
        img_sent_phrases_id = list()
        img_sent_phrases_boxes = list()
        for phrase_info_idx in range(len(sent_info["phrases"])):
            phrase_info = sent_info["phrases"][phrase_info_idx]

            phrase = phrase_info["phrase"]
            phrase_type = phrase_info["phrase_type"]
            phrase_id = phrase_info["phrase_id"]
            if phrase_id not in image_annos["boxes"].keys():
                continue

            phrase_boxes = image_annos["boxes"][phrase_id]  # a nested list
            filted_boxes = filter_bad_boxes(phrase_boxes)
            if not filted_boxes:
                continue

            img_sent_phrases.append(phrase)
            img_sent_phrases_type.append(phrase_type)
            img_sent_phrases_id.append(phrase_id)
            img_sent_phrases_boxes.append(filted_boxes)

        if not img_sent_phrases:
            continue

        items = dict()
        items["sentence"] = img_sent  # a string shows the sentence
        items[
            "sentence_phrases"] = img_sent_phrases  # a list that contains the phrases
        items[
            "sentence_phrases_type"] = img_sent_phrases_type  # a nested list that contains phrases type
        items[
            "sentence_phrases_id"] = img_sent_phrases_id  # a list that contains the phrases  id
        items[
            "sentence_phrases_boxes"] = img_sent_phrases_boxes  # a nested list that contains boxes for each phrase

        aligned_items.append(items)

    return aligned_items


def operate_integration(images_name, images_annotations_path,
                        images_sentences_path):
    integrated_data = dict()
    for image_name_idx in range(len(images_name)):
        image_name = images_name[image_name_idx]
        image_sent_path = images_sentences_path[image_name_idx]
        image_anno_path = images_annotations_path[image_name_idx]

        image_sents = get_sentence_data(image_sent_path)
        try:
            image_annos = get_annotations(image_anno_path)
        except:
            print("image_anno_path: ", image_anno_path)
            image_annos = get_annotations(image_anno_path)

        aligned_items = align_anno_sent(image_sents, image_annos)
        if not aligned_items:
            continue
        for item_idx in range(len(aligned_items)):
            integrated_data[image_name +
                            str(item_idx)] = aligned_items[item_idx]

    return integrated_data
