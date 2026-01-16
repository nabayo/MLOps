from src.utils import print_hello_world
from PIL import Image

def main():
    print_hello_world()
    
    print("hello!")
    
    image = Image.open("dataset/image.jpg")
    print(f"Image size: {image.size}")
    
    
if __name__ == "__main__":
    main()