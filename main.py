#from images_preprocessing import ImagePreprocessor
from transfer_learning_tf import TransferLearning, plot_saved_histories
import os
import datetime
import argparse

# clacola la data corrente per il salvataggio dei risultati - data in formato DDMM
DATE = datetime.datetime.now().strftime("%d%m%y-%H%M")

def main():
    parser = argparse.ArgumentParser(description="Training mosquito recognition model")
    parser.add_argument(
        '--unfreeze',
        action='store_true',
        default=False,
        help="Scongela i pesi della base pre-addestrata per abilitare il fine-tuning completo (default: pesi congelati)"
    )
    parser.add_argument(
        '--aug-percentage',
        type=int,
        choices=[0, 25, 50, 75, 100],
        default=0,
        help=(
            "Percentuale di data augmentation da includere nel training. "
            "0=solo dataset originale, 100=tutto il dataset aumentato. "
            "I valori intermedi aggiungono la frazione corrispondente delle immagini augmentate "
            "al dataset originale (es. 50 = originali + 50%% delle augmented). (default: 0)"
        )
    )
    args = parser.parse_args()
    """ # Inizializza il preprocessore delle immagini
    preprocessor = ImagePreprocessor(target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Esegui il preprocessing delle immagini
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, label_map = preprocessor.run_preprocessing_pipeline()
    """
    # 0% usa il dataset originale, altrimenti si parte sempre da augmented_dataset_splitted
    # e si filtra la percentuale di immagini augmentate desiderata
    BASE_PATH = 'augmented_dataset_splitted' if args.aug_percentage > 0 else 'dataset_splitted'
    TRAIN_PATH = f'{BASE_PATH}/train'
    VAL_PATH = f'{BASE_PATH}/validation'
    TEST_PATH = f'{BASE_PATH}/test'

    print(f"DATASET: aug_percentage={args.aug_percentage}% (source: {BASE_PATH})")

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
            early_stop_patience=20,
            learning_rate=0.002,
            k_folds=5,
            lr_patience=10,
            aug_percentage=args.aug_percentage,
            models_dir= f"keras_training/keras_models_{DATE}_{args.unfreeze}_unfreeze_aug{args.aug_percentage}pct_models",
            results_dir=f"keras_training/keras_models_{DATE}_{args.unfreeze}_unfreeze_aug{args.aug_percentage}pct_performances_and_history",
            image_size=(96, 96),
            freeze_base=not args.unfreeze,
            # models_names=['CustomCNN'],
            # name_to_save_models=['CustomCNN_noDense_conv2D'],
            models_names=['ResNet50', 
                          'MobileNetV2', 
                          'NASNetMobile', 
                          'MobileNet'], #, 'CustomCNN'],
            name_to_save_models=['ResNet50_ClassicDenseClassifier',
                                 'MobileNetV2_ClassicDenseClassifier',
                                 'NASNetMobile_ClassicDenseClassifier',
                                 'MobileNet_ClassicDenseClassifier'], #, 'CustomCNN_ClassicDenseClassifier'],
            final_dense_classifier=True,
            # mobilenet_alpha=0.75  # Ridotto per risparmiare memoria
        )
        
        trainer.run_transfer_learning()
        trainer.save_training_results(output_filename=f'final_training_results_{DATE}.png', show_plots=False)
    
if __name__ == "__main__":
    main()
    
    # plot_saved_histories(
    #     results_dir=f"keras_training/keras_models_{DATE}_performances_and_history",
    #     output_filename=f'final_training_results_{DATE}.png',
    #     show_plots=False)
    # plot_saved_histories(
    #     results_dir=f"keras_training/keras_models_3007_performances_and_history",
    #     output_filename=f'final_training_results_3007.png',
    #     show_plots=False)
    
    print("☑️  ☑️  ☑️ Esecuzione completata con successo! ☑️  ☑️  ☑️")