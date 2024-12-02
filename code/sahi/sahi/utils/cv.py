# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.
import math
import copy
import os
import random
import time
from typing import List, Optional, Union

import cv2
import numpy as np
import requests
from PIL import Image

from sahi.utils.file import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".flv", ".avi", ".ts", ".mpg", ".mov", "wmv"]


class Colors:
    # color palette
    def __init__(self):
        hex = (
            "FF3838",
            "2C99A8",
            "FF701F",
            "6473FF",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "FF9D97",
            "00C2FF",
            "344593",
            "FFB21D",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
def compute_bounding_box(mask):
    # 找到非零元素的索引
    indices = np.nonzero(mask)

    # 計算邊界框的座標
    ymin = np.min(indices[0])
    ymax = np.max(indices[0])
    xmin = np.min(indices[1])
    xmax = np.max(indices[1])

    return ymin, xmin, ymax, xmax


def crop_object_predictions(
    image: np.ndarray,
    object_prediction_list,
    output_dir: str = "",
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Crops bounding boxes over the source image and exports it to output folder.
    Arguments:
        object_predictions: a list of prediction.ObjectPrediction
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    # create output folder if not present
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # add bbox and mask to image if present
    for ind, object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        mask = object_prediction.mask.bool_mask
        bbox = object_prediction.bbox.to_xyxy()
        category_id = object_prediction.category.id
        

                
        rgb_mask = apply_color_mask(mask, (255,255,255))
        # crop detections
        # deepcopy crops so that original is not altered
        cropped_img = copy.deepcopy(
            image[
                int(bbox[1]) : int(bbox[3]),
                int(bbox[0]) : int(bbox[2]),
                :,
            ]
        )

        ymin, xmin, ymax, xmax = compute_bounding_box(rgb_mask)
        cropped_mask = rgb_mask[ymin:ymax, xmin:xmax]
        mask_image = cv2.bitwise_and(cropped_img,cropped_mask)
        
        save_path = os.path.join(
            output_dir,
            file_name + "_box" + str(ind) + "_class" + str(category_id) + "." + export_format,
        )
        cv2.imwrite(save_path, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
    """
    for ind, object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        bbox = object_prediction.bbox.to_xyxy()
        category_id = object_prediction.category.id
        # crop detections
        # deepcopy crops so that original is not altered
        cropped_img = copy.deepcopy(
            image[
                int(bbox[1]) : int(bbox[3]),
                int(bbox[0]) : int(bbox[2]),
                :,
            ]
        )
        save_path = os.path.join(
            output_dir,
            file_name + "_box" + str(ind) + "_class" + str(category_id) + "." + export_format,
        )
        cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        """

def crop_object_predictions_fish(
    image: np.ndarray,
    object_prediction_list,
    output_dir: str = "",
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Crops bounding boxes over the source image and exports it to output folder.
    Arguments:
        object_predictions: a list of prediction.ObjectPrediction
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    # create output folder if not present
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # add bbox and mask to image if present
    for ind, object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        mask = object_prediction.mask.bool_mask
        bbox = object_prediction.bbox.to_xyxy()
        category_id = object_prediction.category.id
        

                
        rgb_mask = apply_color_mask(mask, (255,255,255))
        # crop detections
        # deepcopy crops so that original is not altered
        cropped_img = copy.deepcopy(
            image[
                int(bbox[1]) : int(bbox[3]),
                int(bbox[0]) : int(bbox[2]),
                :,
            ]
        )

        ymin, xmin, ymax, xmax = compute_bounding_box(rgb_mask)
        cropped_mask = rgb_mask[ymin:ymax, xmin:xmax]
        mask_image = cv2.bitwise_and(cropped_img,cropped_mask)
        
        save_path = os.path.join(
            output_dir,
            file_name + "_box" + str(ind) + "_class" + str(category_id) + "." + export_format,
        )
        if not os.path.isdir(output_dir+'_spine_image'):
            os.mkdir(output_dir+'_spine_image')
        if int(str(category_id)) == 3:
            save_path = os.path.join(
            output_dir+'_spine_image',
            #file_name + "_box" + str(ind) + "_class" + str(category_id) + "." + export_format,
            file_name + "_spine"+ "." + export_format,
        )
        cv2.imwrite(save_path, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
    """
    for ind, object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        bbox = object_prediction.bbox.to_xyxy()
        category_id = object_prediction.category.id
        # crop detections
        # deepcopy crops so that original is not altered
        cropped_img = copy.deepcopy(
            image[
                int(bbox[1]) : int(bbox[3]),
                int(bbox[0]) : int(bbox[2]),
                :,
            ]
        )
        save_path = os.path.join(
            output_dir,
            file_name + "_box" + str(ind) + "_class" + str(category_id) + "." + export_format,
        )
        cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        """


def convert_image_to(read_path, extension: str = "jpg", grayscale: bool = False):
    """
    Reads image from path and saves as given extension.
    """
    image = cv2.imread(read_path)
    pre, ext = os.path.splitext(read_path)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pre = pre + "_gray"
    save_path = pre + "." + extension
    cv2.imwrite(save_path, image)


def read_large_image(image_path: str):
    use_cv2 = True
    # read image, cv2 fails on large files
    try:
        # convert to rgb (cv2 reads in bgr)
        img_cv2 = cv2.imread(image_path, 1)
        image0 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    except:
        try:
            import skimage.io
        except ImportError:
            raise ImportError(
                'Please run "pip install -U scikit-image" ' "to install scikit-image first for large image handling."
            )
        image0 = skimage.io.imread(image_path, as_grey=False).astype(np.uint8)  # [::-1]
        use_cv2 = False
    return image0, use_cv2


def read_image(image_path: str):
    """
    Loads image as numpy array from given path.
    """
    # read image
    image = cv2.imread(image_path)
    print(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image
    return image


def read_image_as_pil(image: Union[Image.Image, str, np.ndarray], exif_fix: bool = False):
    """
    Loads an image as PIL.Image.Image.

    Args:
        image : Can be image path or url (str), numpy image (np.ndarray) or PIL.Image
    """
    # https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
    Image.MAX_IMAGE_PIXELS = None

    if isinstance(image, Image.Image):
        image_pil = image
    elif isinstance(image, str):
        # read image if str image path is provided
        try:
            image_pil = Image.open(
                requests.get(image, stream=True).raw if str(image).startswith("http") else image
            ).convert("RGB")
            if exif_fix:
                image_pil = exif_transpose(image_pil)
        except:  # handle large/tiff image reading
            try:
                import skimage.io
            except ImportError:
                raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")
            image_sk = skimage.io.imread(image).astype(np.uint8)
            if len(image_sk.shape) == 2:  # b&w
                image_pil = Image.fromarray(image_sk, mode="1")
            elif image_sk.shape[2] == 4:  # rgba
                image_pil = Image.fromarray(image_sk, mode="RGBA")
            elif image_sk.shape[2] == 3:  # rgb
                image_pil = Image.fromarray(image_sk, mode="RGB")
            else:
                raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")
    elif isinstance(image, np.ndarray):
        if image.shape[0] < 5:  # image in CHW
            image = image[:, :, ::-1]
        image_pil = Image.fromarray(image)
    else:
        raise TypeError("read image with 'pillow' using 'Image.open()'")
    return image_pil


def select_random_color():
    """
    Selects random color.
    """
    colors = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    return colors[random.randrange(0, 10)]


def apply_color_mask(image: np.ndarray, color: tuple):
    """
    Applies color mask to given input image.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    (r[image == 1], g[image == 1], b[image == 1]) = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def get_video_reader(
    source: str,
    save_dir: str,
    frame_skip_interval: int,
    export_visual: bool = False,
    view_visual: bool = False,
):
    """
    Creates OpenCV video capture object from given video file path.

    Args:
        source: Video file path
        save_dir: Video export directory
        frame_skip_interval: Frame skip interval
        export_visual: Set True if you want to export visuals
        view_visual: Set True if you want to render visual

    Returns:
        iterator: Pillow Image
        video_writer: cv2.VideoWriter
        video_file_name: video name with extension
    """
    # get video name with extension
    video_file_name = os.path.basename(source)
    # get video from video path
    video_capture = cv2.VideoCapture(source)

    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if view_visual:
        num_frames /= frame_skip_interval + 1
        num_frames = int(num_frames)

    def read_video_frame(video_capture, frame_skip_interval):
        if view_visual:
            cv2.imshow("Prediction of {}".format(str(video_file_name)), cv2.WINDOW_AUTOSIZE)

            while video_capture.isOpened:

                frame_num = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num + frame_skip_interval)

                k = cv2.waitKey(20)
                frame_num = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

                if k == 27:
                    print(
                        "\n===========================Closing==========================="
                    )  # Exit the prediction, Key = Esc
                    exit()
                if k == 100:
                    frame_num += 100  # Skip 100 frames, Key = d
                if k == 97:
                    frame_num -= 100  # Prev 100 frames, Key = a
                if k == 103:
                    frame_num += 20  # Skip 20 frames, Key = g
                if k == 102:
                    frame_num -= 20  # Prev 20 frames, Key = f
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

                ret, frame = video_capture.read()
                if not ret:
                    print("\n=========================== Video Ended ===========================")
                    break
                yield Image.fromarray(frame)

        else:
            while video_capture.isOpened:
                frame_num = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num + frame_skip_interval)

                ret, frame = video_capture.read()
                if not ret:
                    print("\n=========================== Video Ended ===========================")
                    break
                yield Image.fromarray(frame)

    if export_visual:
        # get video properties and create VideoWriter object
        if frame_skip_interval != 0:
            fps = video_capture.get(cv2.CAP_PROP_FPS)  # original fps of video
            # The fps of export video is increasing during view_image because frame is skipped
            fps = (
                fps / frame_skip_interval
            )  # How many time_interval equals to original fps. One time_interval skip x frames.
        else:
            fps = video_capture.get(cv2.CAP_PROP_FPS)

        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(os.path.join(save_dir, video_file_name), fourcc, fps, size)
    else:
        video_writer = None

    return read_video_frame(video_capture, frame_skip_interval), video_writer, video_file_name, num_frames


def visualize_prediction(
    image: np.ndarray,
    boxes: List[List],
    classes: List[str],
    masks: Optional[List[np.ndarray]] = None,
    rect_th: float = None,
    text_size: float = None,
    text_th: float = None,
    color: tuple = None,
    output_dir: Optional[str] = None,
    file_name: Optional[str] = "prediction_visual",
):
    """
    Visualizes prediction classes, bounding boxes over the source image
    and exports it to output folder.
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3
    # add bbox and mask to image if present
    for i in range(len(boxes)):
        # deepcopy boxso that original is not altered
        box = copy.deepcopy(boxes[i])
        class_ = classes[i]

        # set color
        if colors is not None:
            color = colors(class_)
        # visualize masks if present
        if masks is not None:
            # deepcopy mask so that original is not altered
            mask = copy.deepcopy(masks[i])
            # draw mask
            rgb_mask = apply_color_mask(np.squeeze(mask), color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.7, 0)
        # set bbox points
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            p1,
            p2,
            color=color,
            thickness=rect_th,
        )
        # arange bounding box text location
        label = f"{category_name}"
        w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # add bounding box text
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0]-400, p1[1] - 400),#2 if outside else p1[1] + h + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
        )#感覺很像印出label文字的code
    if output_dir:
        # create output folder if not present
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = os.path.join(output_dir, file_name + ".png")
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def visualize_object_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
    color: tuple = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    output_dir: Optional[str] = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.
    Arguments:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        hide_labels: hide labels
        hide_conf: hide confidence
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        # visualize masks if present
        if object_prediction.mask is not None:
            # deepcopy mask so that original is not altered
            mask = object_prediction.mask.bool_mask
            # set color
            if colors is not None:
                color = colors(object_prediction.category.id)
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

    # add bboxes to image if present
    for object_prediction in object_prediction_list:
        #print('==============object_prediction===============')
        #print(object_prediction)
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()

        bbox = object_prediction.bbox.to_xyxy()
        category_name = object_prediction.category.name
        score = object_prediction.score.value

        # set color
        if colors is not None:
            color = colors(object_prediction.category.id)
        # set bbox points
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            p1,
            p2,
            color=color,
            thickness=rect_th,
        )

        if not hide_labels:
            # arange bounding box text location
            label = f"{category_name}"

            if not hide_conf:
                label = f"{category_name}"
                #label += f" {score:.2f}"
            '''
            w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p1_r = p1[0], p1[1] if outside else p2[1] 
            p2_r = p1[0] + w, p1[1] - h - 3 if outside else p2[1] + h + 3
            #print(p1)
            #print(p2)
            # add bounding box text
            cv2.rectangle(image, p1_r, p2_r, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (p1_r, p2_r),#p1[1] + h + 2),
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )'''
            w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height

            # Calculate the position for the text and the rectangle
            p1_r = p1[0], p1[1] - h - 3
            p2_r = p1[0] + w, p1[1]
            p1_o = p1[0], p2[1] + h + 3
            p2_o = p1[0] + w, p2[1]

            # Check if the label fits outside the box
            outside = p1[1] - h - 3 >= 0

            if outside:
                # Draw the filled rectangle
                cv2.rectangle(image, p1_r, p2_r, color, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(image, p1_o, p2_o, color, -1, cv2.LINE_AA)
            # Adjust the y-coordinate for drawing the text
            text_position = (p1[0], p1[1] - 2) if outside else (p1[0], p2[1] + h + 3)

            # Add bounding box text
            cv2.putText(
                image,
                label,
                text_position,
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )
            

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / (file_name + "." + export_format))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    print(image)
    return {"image": image, "elapsed_time": elapsed_time}


def visualize_moment_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
    color: tuple = None,
    output_dir_moment: Optional[str] = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.
    Arguments:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    #print(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.001), 1)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3
    # add bbox and mask to image if present
    
    allfishmoment=[]
    fishscorelist=[]
    fishscore = 0
    a = 0
    b = 0
    r = 0
    try:
        for object_prediction in object_prediction_list:
            # deepcopy object_prediction_list so that original is not altered
            object_prediction = object_prediction.deepcopy()

            bbox = object_prediction.bbox.to_xyxy()
            category_name = object_prediction.category.name
            score = object_prediction.score.value
            
            # set color
            if colors is not None:
                color = colors(object_prediction.category.id)
            # visualize masks if present
            if object_prediction.mask is not None:
                # deepcopy mask so that original is not altered
                mask = object_prediction.mask.bool_mask
                # draw mask
                
                rgb_mask = apply_color_mask(mask, color)
                #draw mask moments
                
                maskgray = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
                mu = cv2.moments(maskgray, False)
                mux,muy= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
                allfishmoment.append([mux,muy])       
                
                image = cv2.addWeighted(image, 1, rgb_mask, 0.4, 0)
                image = cv2.circle(image, (mux,muy) , 5 ,(255, 0, 0),-1)
                
            # set bbox points
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            # visualize boxes
            cv2.rectangle(
                image,
                p1,
                p2,
                color=color,
                thickness=rect_th,
            )
            # arange bounding box text location
            score= round(score,2)
            label = f"{category_name} {score:.2f}"#label的字?
            w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # add bounding box text
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )
            
            #fish score
            moment=[(file_name + "." + export_format),category_name,score,mux,muy]
            fishscorelist.append(moment)
            path='./outfile'
            if not os.path.isdir(path):
                os.mkdir(path)
            import csv
            pos = 0
            with open('./outfile/fishscore.csv','a+') as f:
                pos += 1
                #writer = csv.writer(f)
                #writer.writerow(moment)
                if category_name == "a":
                    a+=1
                    fishscore+=5
                if category_name == "b":
                    b+=1
                    fishscore+= 10
                if category_name == "r":
                    r+=1
                    fishscore+= 15
        fishscorelist.sort(key = lambda s: s[3])             
        #對x排序
        allfishmoment.sort(key = lambda s: s[0])
        
        #對每個重心算距離
        distrance=[]
        for momentdistance in range(0,len(allfishmoment)-2):
            dis=round(math.sqrt(((allfishmoment[momentdistance][0]-allfishmoment[momentdistance+1][0])**2)+(allfishmoment[momentdistance][1]-allfishmoment[momentdistance+1][1])**2),2)
            distrance.append(dis)
            if momentdistance == len(allfishmoment)-2:
                break
        #這條魚中位數距離 

        fishdisrancemed = float(np.median(distrance[:int(len(distrance)/3)]))
        spine2x=fishscorelist[0][3] + fishdisrancemed
        print(spine2x)
        spine2y=fishscorelist[0][4] 
        spine3x=fishscorelist[0][3] + fishdisrancemed + fishdisrancemed/1.5
        spine3y=fishscorelist[0][4]
        print("first spine = ",fishscorelist[0][3])
        print("med = ",fishdisrancemed)
        print("all list = ",fishscorelist)
        for findspine2 in range(0,len(fishscorelist)-1):
            #找第2節
            if fishscorelist[findspine2][3]> (spine2x-fishdisrancemed/2) and fishscorelist[findspine2][3]< (spine2x+(fishdisrancemed/1.5/2)):
                if fishscorelist[findspine2][1] == "a":
                    fishscore+=15

                if fishscorelist[findspine2][1] == "b":
                    fishscore+=30

                if fishscorelist[findspine2][1] == "r":
                    fishscore+=45

            #找第3節
            if fishscorelist[findspine2][3]> (spine3x-(fishdisrancemed/1.5/2)) and fishscorelist[findspine2][3]< (spine3x+(fishdisrancemed/1.5/2)):
                if fishscorelist[findspine2][1] == "a":
                    fishscore+=15

                if fishscorelist[findspine2][1] == "b":
                    fishscore+=30
                
                if fishscorelist[findspine2][1] == "r":
                    fishscore+=45 

        #扣第1節           
        fishscore-=15                   
        fisharray = ["name",file_name,"fishscore",fishscore, "a:"+str(a), "b:"+str(b), "r:"+str(r)]            
        with open('./outfile/fishscore.csv','a+') as f:
            if fishscorelist[0][1] == 'r':
                writer = csv.writer(f)
                writer.writerow(fisharray)
            else:
                f.write('name,'+file_name+', error\n')
        allfishmoment.clear()
        fishscorelist.clear()
        distrance.clear()
    except:
        pass
    
    # export if output_dir is present
    if output_dir_moment is not None:
        # export image with predictions
        Path(output_dir_moment).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir_moment) / (file_name + "." + export_format))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    
    return {"image": image, "elapsed_time": elapsed_time}


def get_coco_segmentation_from_bool_mask(bool_mask):
    """
    Convert boolean mask to coco segmentation format
    [
        [x1, y1, x2, y2, x3, y3, ...],
        [x1, y1, x2, y2, x3, y3, ...],
        ...
    ]
    """
    # Generate polygons from mask
    mask = np.squeeze(bool_mask)
    mask = mask.astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    # Convert polygon to coco segmentation
    coco_segmentation = []
    for polygon in polygons:
        segmentation = polygon.flatten().tolist()
        # at least 3 points needed for a polygon
        if len(segmentation) >= 6:
            coco_segmentation.append(segmentation)
    return coco_segmentation


def get_bool_mask_from_coco_segmentation(coco_segmentation, width, height):
    """
    Convert coco segmentation to 2D boolean mask of given height and width
    """
    size = [height, width]
    points = [np.array(point).reshape(-1, 2).round().astype(int) for point in coco_segmentation]
    bool_mask = np.zeros(size)
    bool_mask = cv2.fillPoly(bool_mask, points, 1)
    bool_mask.astype(bool)
    return bool_mask


def get_bbox_from_bool_mask(bool_mask):
    """
    Generate voc bbox ([xmin, ymin, xmax, ymax]) from given bool_mask (2D np.ndarray)
    """
    rows = np.any(bool_mask, axis=1)
    cols = np.any(bool_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    width = xmax - xmin
    height = ymax - ymin

    if width == 0 or height == 0:
        return None

    return [xmin, ymin, xmax, ymax]


def normalize_numpy_image(image: np.ndarray):
    """
    Normalizes numpy image
    """
    return image / np.max(image)


def ipython_display(image: np.ndarray):
    """
    Displays numpy image in notebook.

    If input image is in range 0..1, please first multiply img by 255
    Assumes image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
    """
    import IPython

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, ret = cv2.imencode(".png", image)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def exif_transpose(image: Image.Image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()
    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image
