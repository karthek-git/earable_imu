import json
import os

import av
import numpy as np
import pandas as pd
import torch
import transformers

import prompts


def inf(processor, model, vid):
    prompt = processor.apply_chat_template(prompts.mlm_m_conv,
                                           add_generation_prompt=True)
    inputs = processor(text=prompt,
                       videos=vid,
                       padding=True,
                       return_tensors="pt").to("cuda")
    processor.tokenizer.padding_side = "left"
    outp = model.generate(**inputs, max_new_tokens=100)
    r = processor.batch_decode(outp,
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)

    return r[0]


def inf_frames(processor, model, ds, sampling_time, frame_index, frames):
    outp = inf(processor, model, frames)

    try:
        r = json.loads(outp.split("ASSISTANT: ")[1])["Mouth"]
    except Exception:
        return False, ""

    print(f"{sampling_time} frame {frame_index}: {r}")
    ds.loc[frame_index] = (sampling_time, r)


def proc_vid(processor, model, ds, sampling_time, fpath):
    print("Video: " + fpath)
    container = av.open(fpath)
    fps = 25
    stream = container.streams.video[0]
    n_frames = round(stream.duration * stream.time_base * fps)
    print("Total frames:", n_frames)
    container.seek(0)
    cur_sec = 0
    frames = []
    step_size = (fps * 3) / 12
    indices = np.arange(cur_sec * fps, (cur_sec + 3) * fps,
                        step_size).astype(int)

    for i, frame in enumerate(container.decode(video=0)):
        if (len(frames) >= 12) or ((i == (n_frames - 1)) and
                                   (len(frames) >= 4)):
            inf_frames(processor, model, ds, sampling_time, indices[0],
                       np.array(frames))
            sampling_time += pd.Timedelta(seconds=3)
            cur_sec += 3
            indices = np.arange(cur_sec * fps, (cur_sec + 3) * fps,
                                step_size).astype(int)
            frames = []

        if i > indices[-1]:
            continue

        if i >= indices[0] and i in indices:
            frames.append(frame.to_ndarray(format='rgb24'))


def main():
    pretrained_model = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)

    processor = transformers.LlavaNextVideoProcessor.from_pretrained(
        pretrained_model)
    model = transformers.LlavaNextVideoForConditionalGeneration.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        # attn_implementation="flash_attention_2",
        device_map="auto")
    ds_dir = "ds/video"

    for dname in sorted(os.listdir(ds_dir)):
        v_dir = os.path.join(ds_dir, dname)

        ds = pd.DataFrame(columns=("Sampling time", "Mouth"))
        ds['Sampling time'] = pd.to_datetime(ds['Sampling time'])

        for fname in sorted(os.listdir(v_dir)):
            v_fname = os.path.join(v_dir, fname)
            sampling_time = pd.to_datetime(fname.rstrip("B.mp4"),
                                           format="%Y%m%d_%H%M%S")
            proc_vid(processor, model, ds, sampling_time, v_fname)

        ds.to_csv(os.path.join(v_dir, "d_m_labels.csv"))


if __name__ == "__main__":
    main()
