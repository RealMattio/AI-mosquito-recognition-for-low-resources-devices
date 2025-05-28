from images_preprocessing import ImagePreprocessor
from transfer_learning import TransferLearning


def main():
    # Inizializza il preprocessore delle immagini
    preprocessor = ImagePreprocessor(target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Esegui il preprocessing delle immagini
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, label_map = preprocessor.run_preprocessing_pipeline()

    print("Processo di preprocessing e data augmentation completato.")

    # Inizializza il modello di transfer learning
    transfer_learning = TransferLearning(X_train, y_train, X_val, y_val, need_normalize=False,need_resize=False, num_epochs=60, early_stop_patience=10)
    # Esegui il training del modello
    transfer_learning.run_transfer_learning()
    print("Processo di transfer learning completato.")
    transfer_learning.show_training_results()


if __name__ == "__main__":
    main()