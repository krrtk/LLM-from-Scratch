# Apex LLM Engine — First-Principles GPT-2

![Apex LLM Engine](https://img.shields.io/badge/Architecture-GPT--2-blue?style=for-the-badge&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)
![Tech](https://img.shields.io/badge/Tech-PyTorch_%7C_FastAPI-black?style=for-the-badge)

Apex LLM Engine is a professional, first-principles GPT-2 visualiser and fine-tuning suite built entirely from scratch in PyTorch with **zero high-level abstractions**. It provides a "Noir-Tech" dashboard to inspect the internal mechanics of a Transformer model in real-time.

---

## 🌌 Key Features

- **🧠 Raw GPT-2 Architecture**: A 124M parameter model (12 layers, 12 attention heads) implemented from the ground up.
- **📡 Neural Attention Landscape**: Interactive 3D visualization of attention weights across all 12 layers using Plotly.
- **⚡ Real-time X-Ray Inspection**:
  - **Token Candidates**: View top-5 next-token probabilities for every generation step.
  - **Token Breakdown**: Visual "pills" for every generated token.
  - **Typewriter Streaming**: Smooth, simulated token streaming for a premium UX.
- **🧪 Dynamic Fine-Tuning**: Inject a "personality" into the model by uploading `.txt` or `.pdf` files. The engine performs a quick training session to adapt the model weights on-the-fly.
- **🎛️ Generation Control**: Precision tuning with Temperature, Top-K, and Max Token sliders.

## 🛠️ Tech Stack

- **Backend**: Python, PyTorch (Deep Learning), FastAPI (Web Server)
- **Frontend**: Vanilla JS, CSS3 (Modern "Noir" Design), Plotly.js (3D Viz)
- **Deployment**: Docker, Hugging Face Spaces

## 📂 Project Structure

```text
├── hf_space/           # Core application for Hugging Face deployment
│   ├── static/         # Frontend (HTML/CSS/JS)
│   ├── main.py         # FastAPI Entry point
│   ├── engine.py       # Generation & Training logic
│   ├── model.py        # GPT-2 Architecture (PyTorch)
│   ├── tokenizer.py    # Tiktoken wrapper
│   └── data.py         # DataLoader for fine-tuning
├── final/              # Alternative production build
├── final/Dockerfile    # Deployment configuration
└── weights/            # Model weights (pth files)
```

## 🚀 Getting Started

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/apex-llm-engine.git
   cd apex-llm-engine
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r hf_space/requirements.txt
   ```

3. **Run the Engine**:
   ```bash
   cd hf_space
   python main.py
   ```
   The dashboard will be available at `http://localhost:7860`.

### Docker (Recommended)

```bash
docker build -t apex-llm-engine ./hf_space
docker run -p 7860:7860 apex-llm-engine
```

## 👤 Author

**Kartik Yadav**
*Aspiring ML Engineer*
"Building intelligence from zero abstractions."

---

*Note: This project was developed as part of a "LLM from scratch" deep-dive, focusing on understanding the mathematical foundations of modern Transformers.*
