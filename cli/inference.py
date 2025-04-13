# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform
import signal

from cli.SparkTTS import SparkTTS

TIMEOUT_SECONDS = 30  # Global timeout for inference

def parse_args():
    parser = argparse.ArgumentParser(description="Run TTS inference.")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Spark-TTS-0.5B", help="Path to the model directory")
    parser.add_argument("--save_dir", type=str, default="example/results", help="Directory to save generated audio files")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument("--text", type=str, required=True, help="Text for TTS generation")
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument("--prompt_speech_path", type=str, help="Path to the prompt audio file")
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument("--pitch", choices=["very_low", "low", "moderate", "high", "very_high"])
    parser.add_argument("--speed", choices=["very_low", "low", "moderate", "high", "very_high"])
    return parser.parse_args()

# Signal-based timeout handler (UNIX only)
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("TTS inference timed out.")

def run_inference_with_timeout(model, text, prompt_speech_path, prompt_text, gender, pitch, speed, temperature, timeout=TIMEOUT_SECONDS):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        wav = model.inference(
            text,
            prompt_speech_path,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch,
            speed=speed,
            temperature=temperature
        )
        signal.alarm(0)  # Cancel alarm on success
        return wav
    except TimeoutException as e:
        raise TimeoutError("TTS generation timed out.") from e
    finally:
        signal.alarm(0)  # Ensure alarm is always cleared

def run_tts(args):
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.info(f"Cloning text: {args.text}")
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device(f"mps:{args.device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    model = SparkTTS(args.model_dir, device)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, "cloned_audio.wav")

    logging.info("Starting inference...")

    wav = None

    with torch.no_grad():
        for temp in [round(t, 1) for t in [0.8 - 0.1 * i for i in range(4)]]:  # 0.8 → 0.7 → 0.6 → 0.5
            try:
                logging.info(f"Trying TTS generation with temperature: {temp}")
                wav = run_inference_with_timeout(
                    model, args.text, args.prompt_speech_path,
                    args.prompt_text, args.gender, args.pitch, args.speed,
                    temperature=temp
                )
                break  # Success
            except TimeoutError:
                logging.error(f"TTS generation timed out at temperature {temp}")
            except Exception as e:
                logging.error(f"Error during inference at temperature {temp}: {e}")
                break

    if wav is None:
        raise TimeoutError("TTS generation timed out for all temperatures from 0.8 to 0.5")

    sf.write(save_path, wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_tts(args)
