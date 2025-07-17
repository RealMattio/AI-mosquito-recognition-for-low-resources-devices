from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub

# Interfaccia base per i prodotti
class Model(ABC):
    @abstractmethod
    def get_model(self):
        pass

# Implementazione MobileNetV2
class MobileNetV2(Model):
    def get_model(self, input_shape=(96, 96, 3), num_classes=2):
        """
        Costruisce un modello basato su MobileNetV2 pre-addestrata,
        sostituendo il classificatore finale con uno basato su Conv2D 1x1.

        Args:
            input_shape (tuple): La forma delle immagini di input.
            num_classes (int): Il numero di classi finali.

        Returns:
            Un modello Keras pronto per il fine-tuning.
        """
        print("Caricamento della base MobileNetV2 pre-addestrata su ImageNet...")

        # 1. Carichiamo il modello base SENZA il suo classificatore originale
        #    - include_top=False: Rimuove l'ultimo layer Dense.
        #    - weights='imagenet': Carica i pesi pre-addestrati.
        base_model = models.Sequential([
            layers.Input(shape=input_shape),
            tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        ], name="MobileNetV2_Base")


        # 2. Congeliamo i pesi del modello base.
        #    Non vogliamo riaddestrare la parte che ha già imparato da ImageNet.
        base_model.trainable = False
        print("I pesi della base MobileNetV2 sono stati congelati.")

        # 3. Otteniamo l'output del modello base
        #    L'output ha una forma del tipo (None, 3, 3, 1280)
        last_layer_output_shape = base_model.output_shape
        print(f"L'output della base convoluzionale ha forma: {last_layer_output_shape}")
        
        # Estraiamo il numero di filtri dell'ultimo layer (es. 1280 per MobileNetV2)
        last_layer_filters = last_layer_output_shape[-1]

        # 4. Costruiamo il nostro nuovo classificatore efficiente
        #    Questo prende l'output della base e lo processa.
        new_classifier = models.Sequential([
            # Usiamo un InputLayer per definire la forma che il nostro classificatore si aspetta
            layers.InputLayer(input_shape=last_layer_output_shape[1:]),

            # Usiamo GlobalAveragePooling2D per ridurre le dimensioni spaziali
            layers.GlobalAveragePooling2D(name="global_pool"),

            # Reshape per preparare l'input per la Conv2D 1x1
            # da (batch, 1280) a (batch, 1, 1, 1280)
            layers.Reshape((1, 1, last_layer_filters), name="reshape_for_conv"),

            layers.Dropout(0.2, name="dropout_layer"),  # Aggiunta di un layer di dropout per regolarizzazione
            # La nostra Conv2D 1x1 che emula il layer Dense
            layers.Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax', name="classifier_conv"),

            # Flatten finale per avere l'output nel formato corretto (batch, num_classes)
            layers.Flatten(name="final_flatten")
        ], name="Fully_Convolutional_Classifier")

        # 5. Uniamo la base e il nuovo classificatore in un unico modello finale
        model = models.Sequential([
            base_model,
            new_classifier
        ], name="Modified_MobileNetV2")

        return model

# Implementazione MobileNet
class MobileNet(Model):
    def get_model(self, input_shape=(96, 96, 3), num_classes=2):
        """
        Costruisce un modello basato su MobileNet pre-addestrata,
        sostituendo il classificatore finale con uno basato su Conv2D 1x1.

        Args:
            input_shape (tuple): La forma delle immagini di input.
            num_classes (int): Il numero di classi finali.

        Returns:
            Un modello Keras pronto per il fine-tuning.
        """
        print("Caricamento della base MobileNetV2 pre-addestrata su ImageNet...")

        # 1. Carichiamo il modello base SENZA il suo classificatore originale
        #    - include_top=False: Rimuove l'ultimo layer Dense.
        #    - weights='imagenet': Carica i pesi pre-addestrati.
        base_model = models.Sequential([
            layers.Input(shape=input_shape),
            tf.keras.applications.MobileNet(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        ], name="MobileNetV2_Base")


        # 2. Congeliamo i pesi del modello base.
        #    Non vogliamo riaddestrare la parte che ha già imparato da ImageNet.
        base_model.trainable = False
        print("I pesi della base MobileNet sono stati congelati.")

        # 3. Otteniamo l'output del modello base
        #    L'output ha una forma del tipo (None, 3, 3, 1280)
        last_layer_output_shape = base_model.output_shape
        print(f"L'output della base convoluzionale ha forma: {last_layer_output_shape}")
        
        # Estraiamo il numero di filtri dell'ultimo layer (es. 1280 per MobileNetV2)
        last_layer_filters = last_layer_output_shape[-1]

        # 4. Costruiamo il nostro nuovo classificatore efficiente
        #    Questo prende l'output della base e lo processa.
        new_classifier = models.Sequential([
            # Usiamo un InputLayer per definire la forma che il nostro classificatore si aspetta
            layers.InputLayer(input_shape=last_layer_output_shape[1:]),

            # Usiamo GlobalAveragePooling2D per ridurre le dimensioni spaziali
            layers.GlobalAveragePooling2D(name="global_pool"),

            # Reshape per preparare l'input per la Conv2D 1x1
            # da (batch, 1280) a (batch, 1, 1, 1280)
            layers.Reshape((1, 1, last_layer_filters), name="reshape_for_conv"),

            # Aggiunta di un layer di dropout per regolarizzazione
            layers.Dropout(0.2, name="dropout_layer"),
            # La nostra Conv2D 1x1 che emula il layer Dense
            layers.Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax', name="classifier_conv"),

            # Flatten finale per avere l'output nel formato corretto (batch, num_classes)
            layers.Flatten(name="final_flatten")
        ], name="Fully_Convolutional_Classifier")

        # 5. Uniamo la base e il nuovo classificatore in un unico modello finale
        model = models.Sequential([
            base_model,
            new_classifier
        ], name="Modified_MobileNet")

        return model

# Implementazione NASNetMobile
class NASNetMobile(Model):
    def get_model(self, input_shape=(96, 96, 3), num_classes=2):
        """
        Costruisce un modello basato su NASNetMobile pre-addestrata,
        sostituendo il classificatore finale con uno basato su Conv2D 1x1.

        Args:
            input_shape (tuple): La forma delle immagini di input.
            num_classes (int): Il numero di classi finali.

        Returns:
            Un modello Keras pronto per il fine-tuning.
        """
        print("Caricamento della base NASNetMobile pre-addestrata su ImageNet...")

        # 1. Carichiamo il modello base SENZA il suo classificatore originale
        base_model = models.Sequential([
            layers.Input(shape=input_shape),
            tf.keras.applications.NASNetMobile(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        ], name="NASNetMobile_Base")

        # 2. Congeliamo i pesi del modello base
        base_model.trainable = False
        print("I pesi della base NASNetMobile sono stati congelati.")

        # 3. Otteniamo dinamicamente le informazioni sull'output della base
        last_layer_output_shape = base_model.output_shape
        print(f"L'output della base convoluzionale ha forma: {last_layer_output_shape}")
        last_layer_filters = last_layer_output_shape[-1] # Es. 1056 per NASNetMobile

        # 4. Costruiamo il nostro nuovo classificatore efficiente
        new_classifier = models.Sequential([
            layers.InputLayer(input_shape=last_layer_output_shape[1:]),
            layers.GlobalAveragePooling2D(name="global_pool"),
            layers.Reshape((1, 1, last_layer_filters), name="reshape_for_conv"),
            layers.Dropout(0.2, name="dropout_layer"),  # Aggiunta di un layer di dropout per regolarizzazione
            layers.Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax', name="classifier_conv"),
            layers.Flatten(name="final_flatten")
        ], name="Fully_Convolutional_Classifier")

        # 5. Uniamo la base e il nuovo classificatore
        model = models.Sequential([
            base_model,
            new_classifier
        ], name="Modified_NASNetMobile")

        return model

# Implementazione ResNet50
class ResNet50(Model):
    def get_model(self, input_shape=(96, 96, 3), num_classes=2):
        """
        Costruisce un modello basato su ResNet50 pre-addestrata,
        sostituendo il classificatore finale con uno basato su Conv2D 1x1.

        Args:
            input_shape (tuple): La forma delle immagini di input.
            num_classes (int): Il numero di classi finali.

        Returns:
            Un modello Keras pronto per il fine-tuning.
        """
        print("Caricamento della base ResNet50 pre-addestrata su ImageNet...")

        # 1. Carichiamo il modello base SENZA il suo classificatore originale
        base_model = models.Sequential([
            layers.Input(shape=input_shape),
            tf.keras.applications.ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        ], name="ResNet50_Base")


        # 2. Congeliamo i pesi del modello base
        base_model.trainable = False
        print("I pesi della base ResNet50 sono stati congelati.")

        # 3. Otteniamo dinamicamente le informazioni sull'output della base
        last_layer_output_shape = base_model.output_shape
        print(f"L'output della base convoluzionale ha forma: {last_layer_output_shape}")
        last_layer_filters = last_layer_output_shape[-1] # Es. 2048 per ResNet50

        # 4. Costruiamo il nostro nuovo classificatore efficiente
        new_classifier = models.Sequential([
            layers.InputLayer(input_shape=last_layer_output_shape[1:]),
            layers.GlobalAveragePooling2D(name="global_pool"),
            layers.Reshape((1, 1, last_layer_filters), name="reshape_for_conv"),
            layers.Dropout(0.2, name="dropout_layer"),  # Aggiunta di un layer di dropout per regolarizzazione
            layers.Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax', name="classifier_conv"),
            layers.Flatten(name="final_flatten")
        ], name="Fully_Convolutional_Classifier")

        # 5. Uniamo la base e il nuovo classificatore
        model = models.Sequential([
            base_model,
            new_classifier
        ], name="Modified_ResNet50")

        return model

# Implementazione Custom CNN con Conv1D come classificatore
class CustomCNN_Conv1D_Classifier(Model):
    def get_model(self, input_shape=(96, 96, 3), num_classes=2):
        """
        Costruisce una CNN modulare che si adatta a diverse dimensioni di input.
        """
        # 1. Definisci l'input del modello
        inputs = layers.Input(shape=input_shape)

        # 2. Costruisci la base convoluzionale
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        # L'output di questa base avrà una forma (H, W, C) che dipende dall'input
        conv_base_output = layers.Conv2D(32, (3, 3), activation='relu')(x)

        # 3. Calcola dinamicamente la dimensione per il flatten
        # Otteniamo la forma dell'output della base (es. (None, 10, 10, 64))
        shape = conv_base_output.shape
        # Calcoliamo la dimensione del vettore appiattito (es. 10 * 10 * 64 = 6400)
        flattened_size = shape[1] * shape[2] * shape[3]
        
        # 4. Costruisci il classificatore dinamico
        # Il Reshape ora usa la dimensione calcolata al volo
        x = layers.Reshape((flattened_size, 1))(conv_base_output)

        # Il kernel della Conv1D ora usa la dimensione calcolata al volo
        x = layers.Conv1D(filters=32, kernel_size=flattened_size, activation='relu', padding='valid')(x)

        # Aggiungiamo un layer di dropout per regolarizzazione
        x = layers.Dropout(0.2)(x)

        # Il resto del classificatore
        x = layers.Reshape((32, 1))(x) # Questo 32 è fisso perché è il n° di filtri del layer precedente       
        x = layers.Conv1D(num_classes, 32, activation='softmax', padding='valid')(x)

        # Flatten finale per ottenere l'output nella forma corretta (batch, num_classes)
        outputs = layers.Flatten()(x)

        # 5. Crea e restituisci il modello finale
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

# Implementazione Custom CNN con Conv2D 1x1 come classificatore
# Questa versione usa GlobalAveragePooling2D per un classificatore più efficiente
class CustomCNN_Conv2D_Classifier(Model):
    def get_model(self, input_shape=(96, 96, 3), num_classes=2):
        """
        Costruisce una CNN modulare con un classificatore efficiente basato su
        GlobalAveragePooling2D e Conv2D 1x1.
        """
        # 1. Definisci l'input del modello
        inputs = layers.Input(shape=input_shape)

        # 2. Costruisci la base convoluzionale (INVARIATA)
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        conv_base_output = layers.Conv2D(32, (3, 3), activation='relu')(x)

        # --- SEZIONE CLASSIFICATORE SOSTITUITA ---
        # Il vecchio codice con Conv1D e calcoli manuali viene rimosso.
        
        # 3. Nuovo classificatore (naturalmente modulare)
        
        # GlobalAveragePooling2D collassa le dimensioni spaziali (altezza e larghezza)
        # in un singolo pixel, mantenendo i canali (32).
        # Il suo output si adatta automaticamente a qualsiasi dimensione di input.
        x = layers.GlobalAveragePooling2D()(conv_base_output)

        x = layers.Dropout(0.2)(x)
        # L'output di GlobalAveragePooling2D è un vettore (batch, 32).
        # Per usare una Conv2D, dobbiamo dargli di nuovo una dimensione spaziale di 1x1.
        # Otteniamo il numero di filtri (32) dinamicamente.
        num_filters = x.shape[1]
        x = layers.Reshape((1, 1, num_filters))(x)

        # La Conv2D 1x1 agisce come un layer Dense efficiente.
        x = layers.Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(x)

        # Appiattiamo l'output finale per avere la forma corretta (batch, num_classes).
        outputs = layers.Flatten()(x)
        # --- FINE SEZIONE SOSTITUITA ---

        # 4. Crea e restituisci il modello finale
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

# Factory per creare i modelli
class ModelFactory:
    @staticmethod
    def create_model(model, input_shape=(96, 96, 3), num_classes=2):
        if model == "MobileNetV2":
            return MobileNetV2().get_model(input_shape, num_classes)
        elif model == "NASNetMobile":
            return NASNetMobile().get_model(input_shape, num_classes)
        elif model == "ResNet50":
            return ResNet50().get_model(input_shape, num_classes)
        elif model == "MobileNet":
            return MobileNet().get_model(input_shape, num_classes)
        elif model == "CustomCNN_Conv1D_Classifier":
            return CustomCNN_Conv1D_Classifier().get_model(input_shape, num_classes)
        elif model == "CustomCNN_Conv2D_Classifier":
            return CustomCNN_Conv2D_Classifier().get_model(input_shape, num_classes)
        else:
            raise ValueError(f"Tipo di modello sconosciuto: {model}")

# Esempio d'uso:
# model = ModelFactory.create_model("simple")
# print(model.get_object())