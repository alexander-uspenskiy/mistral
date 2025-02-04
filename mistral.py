import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import threading

# Check if MPS (Apple Silicon GPU) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the Mistral-Small-24B-Base-2501 model
model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    torch_dtype=torch.float16,  # Optimized for M1 GPU
    token=token
)

def show_progress():
    while not stop_event.is_set():
        for char in "|/-\\":
            print(f"\rGenerating response {char}", end="", flush=True)
            time.sleep(0.1)

# Interactive terminal loop
print("mistralai/Mistral-Small-24B-Instruct-2501 Chat Interface (type 'exit' to quit)")
interaction_count = 0  # Initialize counter
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=show_progress)
    progress_thread.start()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id  # Added to avoid warning
        )

    stop_event.set()
    progress_thread.join()
    print("\r", end="")  # Clear the progress line

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Mistral: {response}\n")
