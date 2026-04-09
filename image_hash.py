from PIL import Image
import imagehash

def perceptually_same(p1, p2, threshold=5):
    h1 = imagehash.phash(Image.open(p1))
    h2 = imagehash.phash(Image.open(p2))
    return (h1 - h2) <= threshold  # 汉明距离

if __name__ == "__main__":
    img1 = "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0401-130226/base/images/episode_1/episode_1_step_3.png"
    img2 = "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0401-130226/base/images/episode_1/episode_1_step_2.png"

    if perceptually_same(img1, img2):
        print("The images are perceptually the same.")
    else:
        print("The images are different.")