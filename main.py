from images_preprocessing import ImagePreprocessor
from transfer_learning_tf import TransferLearning
import os


def main():
    """ # Inizializza il preprocessore delle immagini
    preprocessor = ImagePreprocessor(target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Esegui il preprocessing delle immagini
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, label_map = preprocessor.run_preprocessing_pipeline()
    """
    # Devi solo fornire i percorsi alle cartelle del tuo dataset
    TRAIN_PATH = 'augmented_dataset_splitted/train'
    VAL_PATH = 'augmented_dataset_splitted/validation'
    TEST_PATH = 'augmented_dataset_splitted/test'
    
    # Assicurati che le cartelle esistano prima di eseguire
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH) or not os.path.exists(TEST_PATH):
        print(f"ERRORE: Assicurati che le cartelle train, val e test esistano.")
        print("Esegui prima lo script per creare la struttura dati.")
    else:
        trainer = TransferLearning(
            train_dir=TRAIN_PATH,
            val_dir=VAL_PATH,
            test_dir=TEST_PATH,
            num_classes=2,
            num_epochs=100,
            early_stop_patience=10,
            learning_rate=0.001,
            k_folds=5,
            lr_patience=3
        )
        
        trainer.run_transfer_learning()
        trainer.save_training_results()
    
if __name__ == "__main__":
    main()