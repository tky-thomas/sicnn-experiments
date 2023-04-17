from experiments.train.train_vgg16 import train_vgg16


def main():
    vgg16_results = train_vgg16("imagenet_classify", save_model_path="/saved_models")

    print(vgg16_results)


if __name__ == "__main__":
    main()
