# copied from https://github.com/shoumikchow/bbox-visualizer

# MIT License
#
# Copyright (c) 2020, Shoumik Sharar Chowdhury
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2


def draw_rectangle(img, bbox, bbox_color=(255, 255, 255), thickness=3, is_opaque=False, alpha=0.5):
    output = img.copy()
    if not is_opaque:
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, thickness)
    else:
        overlay = img.copy()

        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def add_label(img,
              label,
              bbox,
              draw_bg=True,
              text_bg_color=(255, 255, 255),
              text_color=(0, 0, 0),
              font_scale=1.0,
              font_thickness=2,
              top=True):
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    if top:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] - text_height - int(15 * font_scale)]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]), (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] - int(15 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, font_thickness)

    else:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] + text_height + int(15 * font_scale)]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]), (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + 5, bbox[1] + (16 * font_scale) + (4 * font_thickness)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return img


def add_T_label(img,
                label,
                bbox,
                draw_bg=True,
                text_bg_color=(255, 255, 255),
                text_color=(0, 0, 0),
                font_scale=1.0,
                font_thickness=2):
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # draw vertical line
    x_center = (bbox[0] + bbox[2]) // 2
    y_top = bbox[1] - 50
    cv2.line(img, (x_center, bbox[1]), (x_center, y_top), text_bg_color, 3)

    # draw rectangle with label
    y_bottom = y_top
    y_top = y_bottom - text_height - 5
    x_left = x_center - (text_width // 2) - 5
    x_right = x_center + (text_width // 2) + 5
    if draw_bg:
        cv2.rectangle(img, (x_left, y_top - 30), (x_right, y_bottom), text_bg_color, -1)
    cv2.putText(img, label, (x_left + 5, y_bottom - int(8 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                text_color, font_thickness)

    return img


def draw_flag_with_label(img,
                         label,
                         bbox,
                         write_label=True,
                         line_color=(255, 255, 255),
                         text_bg_color=(255, 255, 255),
                         text_color=(0, 0, 0),
                         font_scale=1.0,
                         font_thickness=2):
    (label_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # draw vertical line
    x_center = (bbox[0] + bbox[2]) // 2
    y_bottom = int((bbox[1] * .75 + bbox[3] * .25))
    y_top = bbox[1] - (y_bottom - bbox[1])

    start_point = (x_center, y_top)
    end_point = (x_center, y_bottom)

    cv2.line(img, start_point, end_point, line_color, 3)

    # write label
    if write_label:
        label_bg = [
            start_point[0], start_point[1], start_point[0] + label_width,
            start_point[1] - text_height - int(10 * font_scale)
        ]
        cv2.rectangle(img, (label_bg[0], label_bg[1]), (label_bg[2] + 5, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (start_point[0] + 7, start_point[1] - int(13 * font_scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return img


# THE FOLLOWING ARE OPTIONAL FUNCTIONS THAT CAN BE USED FOR DRAWING OR LABELLING MULTIPLE OBJECTS IN THE SAME
# IMAGE. IN ORDER TO HAVE FULL CONTROL OF YOUR VISUALIZATIONS IT IS ADVISABLE TO USE THE ABOVE FUNCTIONS IN FOR LOOPS
# INSTEAD OF THE FUNCTIONS BELOW


def draw_multiple_rectangles(img, bboxes, bbox_color=(255, 255, 255), thickness=3, is_opaque=False, alpha=0.5):

    for bbox in bboxes:
        img = draw_rectangle(img, bbox, bbox_color, thickness, is_opaque, alpha)
    return img


def add_multiple_labels(img,
                        labels,
                        bboxes,
                        draw_bg=True,
                        text_bg_color=(255, 255, 255),
                        text_color=(0, 0, 0),
                        font_scale=1.0,
                        font_thickness=2,
                        top=True):
    for label, bbox in zip(labels, bboxes):
        img = add_label(img, label, bbox, draw_bg, text_bg_color, text_color, font_scale, font_thickness, top)

    return img


def add_multiple_T_labels(
        img,
        labels,
        bboxes,
        draw_bg=True,
        text_bg_color=(255, 255, 255),
        text_color=(0, 0, 0),
        font_scale=1.0,
        font_thickness=2,
):
    for label, bbox in zip(labels, bboxes):
        add_T_label(img, label, bbox, draw_bg, text_bg_color, text_color, font_scale, font_thickness)

    return img


def draw_multiple_flags_with_labels(
        img,
        labels,
        bboxes,
        write_label=True,
        line_color=(255, 255, 255),
        text_bg_color=(255, 255, 255),
        text_color=(0, 0, 0),
        font_scale=1.0,
        font_thickness=2,
):
    for label, bbox in zip(labels, bboxes):
        img = draw_flag_with_label(img, label, bbox, write_label, line_color, text_bg_color, text_color, font_scale,
                                   font_thickness)
    return img
