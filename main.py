from images_preprocessing import ImagePreprocessor


if __name__ == "__main__":
    # Inizializza il preprocessore delle immagini
    preprocessor = ImagePreprocessor()

    # Esegui il preprocessing delle immagini
    preprocessor.run_preprocessing_pipeline()

    print("Processo di preprocessing e data augmentation completato.")