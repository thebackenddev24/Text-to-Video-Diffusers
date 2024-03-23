from flask import Flask, request, jsonify
import io
import base64
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

step = 4  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

@app.route('/', methods=['GET'])
def index():
    return "Hello World!"

@app.route('/prompt=<prompt>', methods=['GET'])
def generate_animation(prompt):
    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
    gif_bytes = io.BytesIO()
    export_to_gif(output.frames[0], gif_bytes)
    gif_bytes.seek(0)
    encoded_gif = base64.b64encode(gif_bytes.getvalue()).decode('utf-8')
    return jsonify({"video": encoded_gif})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
  