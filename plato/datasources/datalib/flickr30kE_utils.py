"""
Necessary functions for the Flickr30K Entities dataset

"""

import os
import json
import xml.etree.ElementTree as ET
import logging

from plato.datasources.datalib import data_utils


def phrase_boxes_alignment(flatten_boxes, ori_phrases_boxes):
    """ align the bounding boxes with corresponding phrases. """
    phrases_boxes = []

    ori_pb_boxes_count = []
    for ph_boxes in ori_phrases_boxes:
        ori_pb_boxes_count.append(len(ph_boxes))

    strat_point = 0
    for pb_boxes_num in ori_pb_boxes_count:
        sub_boxes = []
        for i in range(strat_point, strat_point + pb_boxes_num):
            sub_boxes.append(flatten_boxes[i])

        strat_point += pb_boxes_num
        phrases_boxes.append(sub_boxes)

    pb_boxes_count = []
    for ph_boxes in phrases_boxes:
        pb_boxes_count.append(len(ph_boxes))

    assert pb_boxes_count == ori_pb_boxes_count

    return phrases_boxes


def filter_bad_boxes(boxes_coor):
    """ Filter the boxes with wrong coordinates """
    filted_boxes = []
    for box_coor in boxes_coor:
        [xmin, ymin, xmax, ymax] = box_coor
        if xmin < xmax and ymin < ymax:
            filted_boxes.append(box_coor)

    return filted_boxes


def get_sentence_data(parse_file_path):
    """ Parses a sentence file from the Flickr30K Entities dataset

    Args:
        parse_file_path - full file path to the sentence file to parse
    Return:
        a list of dictionaries for each sentence with the following fields:
            sentence - the original sentence
            phrases - a list of dictionaries for each phrase with the
                    following fields:
                        phrase - the text of the annotated phrase
                        first_word_index - the position of the first word of
                                            the phrase in the sentence
                        phrase_id - an identifier for this phrase
                        phrase_type - a list of the coarse categories this phrase belongs to
    """
    with open(parse_file_path, 'r') as opened_file:
        sentences = opened_file.read().split('\n')

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


def get_annotations(parse_file_path):
    """ Parses the xml files in the Flickr30K Entities dataset.
    Args:
        parse_file_path - full file path to the annotations file to parse
    Return:
        dictionary with the following fields:
            scene - list of identifiers which were annotated as
                    pertaining to the whole scene
            nobox - list of identifiers which were annotated as
                    not being visible in the image
            boxes - a dictionary where the fields are identifiers
                    and the values are its list of boxes in the [xmin ymin xmax ymax] format
    """
    tree = ET.parse(parse_file_path)
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
    """Align the items in annotations and sentences.

    Args:
        image_sents ([list]): [each itme is a dict that contains 'sentence', 'phrases']
        image_annos ([dict]): [contain 'boxes' - a dict presents the phrase_id: box]

    Return:
            aligned_items ([list]): [each itme is a dict that contains the sentence with
                corresponding phrases information, there should have several
                items because for one image, there are 5 sentences. Sometimes,
                some sentences are useless, making the number of items less than 5]
    """
    aligned_items = []  # each item is a dict
    for sent_info in image_sents:

        img_sent = sent_info["sentence"]
        img_sent_phrases = []
        img_sent_phrases_type = []
        img_sent_phrases_id = []
        img_sent_phrases_boxes = []
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
        # a string shows the sentence
        items["sentence"] = img_sent
        # a list that contains the phrases
        items["sentence_phrases"] = img_sent_phrases
        # a nested list that contains phrases type
        items["sentence_phrases_type"] = img_sent_phrases_type
        # a list that contains the phrases  id
        items["sentence_phrases_id"] = img_sent_phrases_id
        # a nested list that contains boxes for each phrase
        items["sentence_phrases_boxes"] = img_sent_phrases_boxes

        aligned_items.append(items)

    return aligned_items


def integrate_data_to_json(splits_info,
                           mm_data_info,
                           data_types,
                           split_wise=True,
                           globally=True):
    """ Integrate the data into one json file that contains aligned
        annotation-sentence for each image.
        
        The integrated data info is presented as a dict type.
        
        Each item in dict contains image and one of its annotation.

        For example, one randomly item:
            {
            ...,
                "./data/Flickr30KEntities/test/test_Images/1011572216.jpg0"
                 {"sentence": "bride and groom",
                 "sentence_phrases": ["bride", "groom"],
                "sentence_phrases_type": [["people"], ["people"]],
                "sentence_phrases_id": ["370", "372"],
                "sentence_phrases_boxes": [[[161, 21, 330, 357]], 
                                            [[195, 82, 327, 241]]],
                },
            ....
            }
        """

    def operate_integration(images_name, images_annotations_path,
                            images_sentences_path):
        """ Obtain the integrated for images. """
        integrated_data = dict()
        for image_name_idx, image_name in enumerate(images_name):
            image_sent_path = images_sentences_path[image_name_idx]
            image_anno_path = images_annotations_path[image_name_idx]

            image_sents = get_sentence_data(image_sent_path)

            image_annos = get_annotations(image_anno_path)

            aligned_items = align_anno_sent(image_sents, image_annos)
            if not aligned_items:
                continue
            for item_idx, item in enumerate(aligned_items):
                integrated_data[image_name + str(item_idx)] = item

        return integrated_data

    if split_wise:
        for split_type in list(splits_info.keys()):
            path = splits_info[split_type]["path"]
            save_path = os.path.join(path,
                                     split_type + "_integrated_data.json")
            if os.path.exists(save_path):
                logging.info("Integrating %s: the file already exists.",
                             split_type)
                continue

            split_data_types_samples_path = []
            for _, data_type in enumerate(data_types):
                data_type_format = splits_info[split_type][data_type]["format"]
                split_data_type_path = splits_info[split_type][data_type][
                    "path"]

                split_data_type_samples = data_utils.list_inorder(
                    os.listdir(split_data_type_path),
                    flag_str=data_type_format)

                split_data_type_samples_path = [
                    os.path.join(split_data_type_path, sample)
                    for sample in split_data_type_samples
                ]

                split_data_types_samples_path.append(
                    split_data_type_samples_path)

            split_integrated_data = operate_integration(
                images_name=split_data_types_samples_path[0],
                images_annotations_path=split_data_types_samples_path[1],
                images_sentences_path=split_data_types_samples_path[2])
            with open(save_path, 'w', encoding='utf-8') as outfile:
                json.dump(split_integrated_data, outfile)

            logging.info("The integration process for %s is done.", split_type)

    if globally:
        save_path = os.path.join(mm_data_info["data_path"],
                                 "total_integrated_data.json")
        if os.path.exists(save_path):
            logging.info("Gloablly integrated file already exists.")
            return

        raw_data_types_samples_path = []
        for _, data_type in enumerate(data_types):
            data_type_format = mm_data_info[data_type]["format"]
            raw_data_type_path = mm_data_info[data_type]["path"]

            global_raw_type_samples = data_utils.list_inorder(
                os.listdir(raw_data_type_path), flag_str=data_type_format)

            global_raw_type_samples_path = [
                os.path.join(raw_data_type_path, sample)
                for sample in global_raw_type_samples
            ]
            raw_data_types_samples_path.append(global_raw_type_samples_path)

        global_integrated_data = operate_integration(
            images_name=raw_data_types_samples_path[0],
            images_annotations_path=raw_data_types_samples_path[1],
            images_sentences_path=raw_data_types_samples_path[2])
        with open(save_path, 'w', encoding='utf-8') as outfile:
            json.dump(global_integrated_data, outfile)

        logging.info("Integration for the whole dataset, Done.")
