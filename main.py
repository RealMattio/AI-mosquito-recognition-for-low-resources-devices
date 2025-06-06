from images_preprocessing import ImagePreprocessor
from transfer_learning import TransferLearning, evaluate_and_save_results, generate_learning_curves
from transfer_learning import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
def my_get_transfer_model(name, num_classes):
    # Esempio: costruisce un modello di transfer learning con classificatore binario
    return TransferLearning(None, None, None, None, num_classes).get_model(name, num_classes)

def main():
    # 4.1) Path dataset e modelli
    DATA_DIR = "./augmented_dataset"  
    SKLEARN_MODELS_DIR = "models"            # dove hai i .joblib (allenati flattenando immagini)
    PYTORCH_MODELS_DIR = "saved_pytorch"     # dove hai i .pth (transfer pretrained)

    # 4.2) Carica dataset
    #    Otteniamo:
    #      X_flat: array (N, 128*128*3),
    #      paths: lista di file path,
    #      y: array di etichette numeriche
    X_flat, paths, y, encoder = load_dataset(DATA_DIR, image_size=(128,128))
    print("\n --- Dataset per ML models caricato con successo. ---")
    # 4.3) Split train/validation (fisso) – stratificato per classe
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=42
    )

    X_flat_train = X_flat[train_idx]
    y_flat_train = y[train_idx]
    X_flat_val = X_flat[val_idx]
    y_flat_val = y[val_idx]

    print("\n --- Dataset per ML models diviso in train e validation. ---")
    print("\n --- Inizializzazione del preprocessore delle immagini e dei modelli di transfer learning. ---")

    # Inizializza il preprocessore delle immagini
    preprocessor = ImagePreprocessor(target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Esegui il preprocessing delle immagini
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, label_map = preprocessor.run_preprocessing_pipeline()

    print("Processo di preprocessing e data augmentation completato.")

    # Inizializza il modello di transfer learning
    #transfer_learning = TransferLearning(X_train, y_train, X_val, y_val, need_normalize=False,need_resize=False, num_epochs=60, early_stop_patience=10)
    # Esegui il training del modello
    #transfer_learning.run_transfer_learning()
    #print("Processo di transfer learning completato.")
    #transfer_learning.show_training_results()
    metrics = generate_learning_curves(
        X_flat_train, y_flat_train,
        X_flat_val, y_flat_val,
        X_train, y_train,
        X_val, y_val,
        sklearn_models_dir='models',
        pytorch_models_dir='saved_models',
        get_transfer_model=my_get_transfer_model,
        output_plot_path='results/learning_curves.png',
        output_metrics_path='results/learning_metrics.json',
        output_models_dir='results/retrained_models'
    )
""" 
    results = evaluate_and_save_results(X_test, y_test, need_normalize=False, need_resize=False, output_json="test_results_2_fiscore4all.json", label_map=label_map)
    print("Processo di valutazione e salvataggio dei risultati completato.")
    print("Risultati:", results)
 """
if __name__ == "__main__":
    main()