import json
import math
import os

import cv2
import pandas as pd
import torch
import transformers

import prompts


def inf(processor, model, img):
    inputs = processor(prompts.prompt, img, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=1000)
    r = processor.decode(output[0], skip_special_tokens=True)

    return r


def inf_frame(processor, model, cap, ds, sampling_time, frame_index):
    frame_index = round(frame_index)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        return False, ""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outp = inf(processor, model, rgb_frame)

    try:
        r = json.loads(
            outp.split("ASSISTANT: ```json")[1].rstrip("```"))["Direction"]
    except Exception:
        return False, ""

    print(f"{sampling_time} frame {frame_index}: {r}")
    ds.loc[len(ds.index)] = (sampling_time, r)

    return True, r


def proc_vid(processor, model, ds, sampling_time, fname):
    print("Video: " + fname)
    cap = cv2.VideoCapture(fname, cv2.CAP_ANY)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cur_sec = 1

    while cap.isOpened():
        ret, f_d = inf_frame(processor, model, cap, ds, sampling_time,
                             (cur_sec - 1) * fps)

        if not ret:
            break

        ret, m_d = inf_frame(processor, model, cap, ds, sampling_time,
                             math.floor(((cur_sec - 1) * fps) + (fps / 2)))

        if not ret:
            break

        ret, l_d = inf_frame(processor, model, cap, ds, sampling_time,
                             (cur_sec * fps) - 1)

        if not ret:
            break

        sampling_time += pd.Timedelta(seconds=1)
        cur_sec += 1


def main():
    pretrained_model = "llava-hf/llava-v1.6-vicuna-7b-hf"

    transformers.BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)

    processor = transformers.LlavaNextProcessor.from_pretrained(
        pretrained_model)
    model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
        # attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        device_map="auto")

    ds_dir = "ds/video"

    for dname in sorted(os.listdir(ds_dir)):
        v_dir = os.path.join(ds_dir, dname)

        ds = pd.DataFrame(columns=("Sampling Time", "Direction"))

        for fname in sorted(os.listdir(v_dir)):
            v_fname = os.path.join(v_dir, fname)
            sampling_time = pd.to_datetime(fname.rstrip("B.mp4"),
                                           format="%Y%m%d_%H%M%S")
            proc_vid(processor, model, ds, sampling_time, v_fname)

        ds.to_csv(os.path.join(v_dir, "d_labels.csv"))


if __name__ == "__main__":
    main()
