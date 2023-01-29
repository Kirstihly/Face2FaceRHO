import cv2
import os
import shutil
import subprocess

import numpy as np

from argparse import ArgumentParser

TRIALS = 5
PADDING = 5


def generate_frames(video_file):
    assert os.path.isfile(video_file), video_file + " not exist!"

    video_dir = video_file.split(".")[0]
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)

    print("Generating frames:", video_dir)
    os.makedirs(video_dir)
    command = "ffmpeg -i " + video_file + " -vf fps=30 " + video_dir + "/%05d.png"
    print(command)
    subprocess.check_call(command, shell=True, timeout=60)

    frame_num = len(os.listdir(video_dir))
    print("Frames extracted:", frame_num)
    return frame_num


def stitch_video(args, source_num, driving_num):
    images = []
    for i in range(min(source_num, driving_num)):
        frame_name = str(i + 1).zfill(PADDING) + ".png"
        source_frame = os.path.join("data", args.source, frame_name)
        driving_frame = os.path.join("data", args.driving, frame_name)
        result_frame = os.path.join(
            "data", "result", args.source + "_" + args.driving + "_" + frame_name
        )

        source_img = cv2.imread(source_frame, cv2.IMREAD_COLOR)
        driving_img = cv2.imread(driving_frame, cv2.IMREAD_COLOR)
        result_img = cv2.imread(result_frame, cv2.IMREAD_COLOR)

        # Result images defined to be 512x512
        source_img = cv2.resize(source_img, (512, 512))
        driving_img = cv2.resize(driving_img, (512, 512))
        assert result_img.shape == driving_img.shape
        assert result_img.shape == source_img.shape

        source_img = cv2.putText(
            source_img,
            "SOURCE",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        driving_img = cv2.putText(
            driving_img,
            "DRIVING",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        result_img = cv2.putText(
            result_img,
            "RESULT",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        canvas = np.concatenate((source_img, driving_img), axis=1)
        canvas = np.concatenate((canvas, result_img), axis=1)
        images.append(canvas)

    out_path = os.path.join(
        "data", "result", args.source + "_" + args.driving + "." + args.format
    )
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (512 * 3, 512),
    )
    for image in images:
        out.write(image)
    out.release()
    print("Stitched video saved:", out_path)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="A Python script to concat source, driving, and generated result videos. \
                     Assume source and driving videos are saved in data/",
        epilog="python video_demo.py -s 00001 -d 00051",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="The source video filename.",
    )
    parser.add_argument(
        "-d",
        "--driving",
        type=str,
        help="The driving video filename.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        help="The video format. Default: mp4.",
    )
    args = parser.parse_args()

    source_file = os.path.join("data", args.source + "." + args.format)
    source_num = generate_frames(source_file)

    driving_file = os.path.join("data", args.driving + "." + args.format)
    driving_num = generate_frames(driving_file)

    # Process frames
    for i in range(min(source_num, driving_num)):
        num_padding = str(i + 1).zfill(PADDING)
        frame_name = num_padding + ".png"
        headpose_name = num_padding + "_headpose.txt"
        landmark_name = num_padding + "_landmark.txt"

        src_headpose = os.path.join("data", "source", "FLAME", headpose_name)
        src_landmark = os.path.join("data", "source", "FLAME", landmark_name)
        drv_headpose = os.path.join("data", "driving", "FLAME", headpose_name)
        drv_landmark = os.path.join("data", "driving", "FLAME", landmark_name)
        result_name = os.path.join(
            "data", "result", args.source + "_" + args.driving + "_" + frame_name
        )

        # Generate facial landmarks and head pose coefficients
        command = (
            "python src/fitting.py"
            + " --device cuda"
            + " --src_img ./data/"
            + args.source
            + "/"
            + frame_name
            + " --drv_img ./data/"
            + args.driving
            + "/"
            + frame_name
            + " --output_src_headpose "
            + src_headpose
            + " --output_src_landmark "
            + src_landmark
            + " --output_drv_headpose "
            + drv_headpose
            + " --output_drv_landmark "
            + drv_landmark
        )
        print(command)
        trial = 0
        while trial < TRIALS:
            trial += 1
            if not (
                os.path.exists(src_headpose)
                and os.path.exists(src_landmark)
                and os.path.exists(drv_headpose)
                and os.path.exists(drv_landmark)
            ):
                subprocess.check_call(command, shell=True, timeout=60)

        # Generate final reenacted results
        command = (
            "python src/reenact.py"
            + " --config src/config/test_face2facerho.ini"
            + " --src_img ./data/"
            + args.source
            + "/"
            + frame_name
            + " --src_headpose "
            + src_headpose
            + " --src_landmark "
            + src_landmark
            + " --drv_headpose "
            + drv_headpose
            + " --drv_landmark "
            + drv_landmark
            + " --output_dir ./data/result"
        )
        print(command)
        trial = 0
        while trial < TRIALS:
            trial += 1
            if os.path.exists("./data/result/result.png"):
                command = "mv ./data/result/result.png " + result_name
                subprocess.check_call(command, shell=True, timeout=60)
            elif not os.path.exists(result_name):
                subprocess.check_call(command, shell=True, timeout=60)

    # Concat source, driving, and generated result videos
    stitch_video(args, source_num, driving_num)
