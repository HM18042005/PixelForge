# PixelForge
PixelForge is a fully offline, locally executed AI image-generation system built using Stable Diffusion 1.5 as the base model. Unlike cloud-based AI generators that rely on external APIs, PixelForge performs all model inference directly on the user’s machine using a Python backend and a Next.js/React frontend. This ensures fast generation, privacy preservation, and complete independence from cloud services. The system offers full control over generation parameters such as resolution, CFG scale, inference steps, seed value, negative prompts, and more. PixelForge is suitable for creative users, students, researchers, and developers who need a customizable and fully local generative AI platform. PixelForge also supports LoRA-based fine-tuning using images sourced from the Open Images V7 dataset, incrementally trained across semantic categories such as Person, Indoor, Outdoor, Product, Food, Animal, Vehicle, Architecture, and Landscape. Captioning for dataset images is automated using BLIP, ensuring high-quality text–image pairs for fine-tuning.

## Stable Diffusion 3 model download

1. Install the Python dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Create a Hugging Face token with the required model access scopes and expose it before running the download script:

	```powershell
	setx HF_TOKEN "hf_your_token_here"
	```

	Restart the terminal so the new environment variable is loaded.

3. Run the downloader to fetch Stability AI’s Stable Diffusion 3 Medium weights into the local `models` directory:

	```bash
	python scripts/download_stable_diffusion.py
	```

The snapshot download replicates the exact repository layout published on Hugging Face at `stabilityai/stable-diffusion-3-medium-diffusers`, which is the latest generally available Stable Diffusion release compatible with diffusers-based pipelines.
