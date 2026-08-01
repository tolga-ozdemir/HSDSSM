import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available")

if __name__ == "__main__":
    main()
