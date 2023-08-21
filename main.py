from experiments.train.train_vgg16 import train_vgg16
from experiments.train.train_sesn import train_sesn
from experiments.train.train_ss import train_ss
import pickle


def main():
    #     vgg16_aar_results = train_vgg16("aar",
    #                                     save_model_path="saved_models/vgg16_aar.pt",
    #                                     epochs=30)

    #     # Saves the results to a file for later analysis
    #     with open('experiment_results/vgg16_aar.pkl', 'wb') as fp:
    #        pickle.dump(vgg16_aar_results, fp)
    #        print('Results of VGG16_AAR saved to experiment_results/vgg16_aar.pkl')

    # vgg16_imagenet_classify_results = train_vgg16("imagenet_classify",
    #                                 save_model_path="saved_models/vgg16_imagenet_classify.pt",
    #                                 epochs=30,
    #                                 batch_size=32)
    # with open('experiment_results/vgg16_imagenet_classify.pkl', 'wb') as fp:
    #     pickle.dump(vgg16_imagenet_classify_results, fp)
    #     print('Results of VGG16_IMAGENET_CLASSIFY saved to experiment_results/vgg16_imagenet_classify.pkl')

    #     ss_aar_results = train_ss("aar",
    #                                     save_model_path="saved_models/ss_aar.pt",
    #                                     epochs=30)

    #     # Saves the results to a file for later analysis
    #     with open('experiment_results/ss_aar.pkl', 'wb') as fp:
    #        pickle.dump(ss_aar_results, fp)
    #        print('Results of SS_AAR saved to experiment_results/ss_aar.pkl')

    sesn_aar_results = train_sesn("aar",
                                  save_model_path="saved_models/sens_aar.pt",
                                  epochs=30)

    # Saves the results to a file for later analysis
    with open('experiment_results/sesn_aar.pkl', 'wb') as fp:
        pickle.dump(sesn_aar_results, fp)
        print('Results of SENS_AAR saved to experiment_results/sesn_aar.pkl')

    # ss_imagenet_classify_results = train_ss("imagenet_classify",
    #                               save_model_path="saved_models/ss_imagenet_classify.pt",
    #                               epochs=30,
    #                               batch_size=32)
    # with open('experiment_results/ss_imagenet_classify.pkl', 'wb') as fp:
    #     pickle.dump(ss_imagenet_classify_results, fp)
    #     print('Results of SS_IMAGENET_CLASSIFY saved to experiment_results/ss_imagenet_classify.pkl')

    # sesn_imagenet_classify_results = train_sesn("imagenet_classify",
    #                               save_model_path="saved_models/sesn_imagenet_classify.pt",
    #                               epochs=30,
    #                               batch_size=32)
    # with open('experiment_results/sesn_imagenet_classify.pkl', 'wb') as fp:
    #     pickle.dump(sesn_imagenet_classify_results, fp)
    #     print('Results of SESN_IMAGENET_CLASSIFY saved to experiment_results/sesn_imagenet_classify.pkl')

    # sesn_voc_segmentation_results = train_sesn("voc_segmentation",
    #                                 save_model_path="saved_models/sesn_voc_segmentation.pt",
    #                                 epochs=20,
    #                                 batch_size=8)
    # with open('experiment_results/sesn_voc_segmentation.pkl', 'wb') as fp:
    #     pickle.dump(sesn_voc_segmentation_results, fp)
    #     print('Results of SESN_VOC_SEGMENTATION saved to experiment_results/sesn_voc_segmentation.pkl')


if __name__ == "__main__":
    main()
