import json

import cv2
import pandas as pd
import torch
import transformers

import prompts


def inf(processor, model, img):
    inputs = processor(prompts.prompt, img, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=1000)
    r = processor.decode(output[0], skip_special_tokens=True)

    return r


def inf_frame(processor, model, cap, ds, frame_index):
    frame_index = round(frame_index)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        return False, ""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outp = inf(processor, model, rgb_frame)
    r = json.loads(
        outp.split("ASSISTANT: ```json")[1].rstrip("```"))["Direction"]
    print(f"frame {frame_index}: {r}")
    ds.loc[frame_index] = (frame_index, r)

    return True, r


def main():
    pretrained_model = "llava-hf/llava-v1.6-vicuna-7b-hf"

    processor = transformers.LlavaNextProcessor.from_pretrained(
        pretrained_model)
    model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto")

    ds = pd.DataFrame(columns=("Sampling time", "Direction"))
    ds['Sampling time'] = pd.to_datetime(ds['Sampling time'])

    cap = cv2.VideoCapture("ds/vid.mp4", cv2.CAP_ANY)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cur_sec = 1

    while cap.isOpened():
        ret, f_d = inf_frame(processor, model, cap, ds, (cur_sec - 1) * fps)

        if not ret:
            break

        ret, m_d = inf_frame(processor, model, cap, ds, fps / 2)

        if not ret:
            break

        ret, l_d = inf_frame(processor, model, cap, ds, (cur_sec * fps) - 1)

        if not ret:
            break
        cur_sec += 1

    ds.to_csv("d_labels.csv")


if __name__ == "__main__":
    main()
