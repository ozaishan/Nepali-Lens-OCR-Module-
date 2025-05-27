import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import pickle

class NepaliOCRModel:
    def __init__(self, img_height=64, img_width=256, max_length=20):
        self.img_height = img_height
        self.img_width = img_width
        self.max_length = max_length
        
        # Create character mappings for Nepali text
        self.characters = self._create_character_set()
        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.characters)}
        
    def _create_character_set(self):
        """Create character set including Nepali Devanagari characters"""
        # Basic Nepali Devanagari characters
        nepali_chars = [
            'क', 'ख', 'ग', 'घ', 'ङ',
            'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण',
            'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म',
            'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह',
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ै', 'ओ', 'ौ',
            'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ',
            '्', 'ं', 'ः', 'ँ', 'ऽ',
            '०', '१', '२', '३', '४', '५', '६', '७', '८', '९'
        ]
        
        # Add space and special tokens
        chars = ['[PAD]', '[UNK]', ' '] + nepali_chars
        return sorted(list(set(chars)))
    
    def load_data(self, csv_path, images_dir, sample_limit=None):
        """Load and preprocess data from CSV and images with optional sampling"""
        print(f"Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path, header=None, encoding="utf-16")
        df.columns = ['filename', 'text']
        
        # Optional: Sample data for testing
        if sample_limit and len(df) > sample_limit:
            df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_limit} records for training")
        
        print(f"Processing {len(df)} records...")
        
        images = []
        labels = []
        failed_count = 0
        
        for idx, row in df.iterrows():
            if idx % 500 == 0 and idx > 0:
                print(f"Processed {idx}/{len(df)} images...")
            
            img_path = os.path.join(images_dir, row['filename'])
            if os.path.exists(img_path):
                # Load and preprocess image
                img = self._preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(str(row['text']))  # Ensure text is string
                else:
                    failed_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # Show first 10 missing files
                    print(f"Warning: Image not found: {img_path}")
        
        print(f"Successfully loaded {len(images)} images")
        print(f"Failed to load {failed_count} images")
        
        return np.array(images), labels
    
    def _preprocess_image(self, img_path):
        """Preprocess image for OCR"""
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target dimensions
            resized = cv2.resize(gray, (self.img_width, self.img_height))
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            # Add channel dimension
            return np.expand_dims(normalized, axis=-1)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None
    
    def encode_labels(self, labels):
        """Encode text labels to numerical sequences"""
        encoded_labels = []
        
        for label in labels:
            encoded = []
            for char in label:
                if char in self.char_to_num:
                    encoded.append(self.char_to_num[char])
                else:
                    encoded.append(self.char_to_num['[UNK]'])
            
            # Pad or truncate to max_length
            if len(encoded) < self.max_length:
                encoded.extend([self.char_to_num['[PAD]']] * (self.max_length - len(encoded)))
            else:
                encoded = encoded[:self.max_length]
            
            encoded_labels.append(encoded)
        
        return np.array(encoded_labels)
    
    def create_model(self):
        """Create enhanced CRNN model for OCR with better architecture for larger dataset"""
        # Input layer
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1), name='image')
        
        # Enhanced CNN layers for feature extraction
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(512, (2, 2), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Reshape for RNN
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 512)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Enhanced RNN layers
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
        
        # Output layer
        x = layers.Dense(len(self.characters), activation='softmax', name='dense_output')(x)
        
        model = keras.Model(inputs=input_img, outputs=x, name='nepali_ocr_enhanced')
        return model
    
    def ctc_loss(self, y_true, y_pred):
        """CTC loss function"""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss
    
    def create_data_generator(self, images, labels, batch_size=32, shuffle=True):
        """Create data generator for memory-efficient training"""
        def generator():
            indices = np.arange(len(images))
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(images), batch_size):
                end_idx = min(start_idx + batch_size, len(images))
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]
                
                yield batch_images, batch_labels
        
        return generator
    
    def train_model(self, csv_path, images_dir, epochs=50, batch_size=32, validation_split=0.2):
        """Train the OCR model with memory-efficient data loading"""
        print("Loading data...")
        images, labels = self.load_data(csv_path, images_dir)
        
        if len(images) == 0:
            print("No images found! Please check your paths.")
            return None
        
        print(f"Loaded {len(images)} images")
        
        # Encode labels
        print("Encoding labels...")
        encoded_labels = self.encode_labels(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, encoded_labels, test_size=validation_split, random_state=42, stratify=None
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create model
        print("Creating model...")
        model = self.create_model()
        
        # Compile model with custom optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=self.ctc_loss
        )
        
        print("Model architecture:")
        model.summary()
        
        # Enhanced callbacks for larger dataset
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15, 
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_nepali_ocr_model.h5', 
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=8, 
                factor=0.5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.CSVLogger('training_log.csv'),
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Starting training for {epochs} epochs...")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model and character mappings
        model.save('nepali_ocr_final_model.h5')
        
        with open('char_mappings.pkl', 'wb') as f:
            pickle.dump({
                'char_to_num': self.char_to_num,
                'num_to_char': self.num_to_char,
                'characters': self.characters,
                'max_length': self.max_length,
                'img_height': self.img_height,
                'img_width': self.img_width
            }, f)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate model performance
        self.evaluate_model(model, X_val, y_val)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def predict_text(self, model, image_path):
        """Predict text from image"""
        img = self._preprocess_image(image_path)
        if img is None:
            return "Error processing image"
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img_batch)
        
        # Decode prediction
        decoded_text = self.decode_prediction(predictions[0])
        return decoded_text
    
    def decode_prediction(self, prediction):
        """Decode numerical prediction to text"""
        # Get most likely characters
        pred_indices = np.argmax(prediction, axis=1)
        
        # Remove consecutive duplicates and padding
        decoded_chars = []
        prev_char = None
        
        for idx in pred_indices:
            char = self.num_to_char[idx]
            if char != '[PAD]' and char != prev_char:
                decoded_chars.append(char)
            prev_char = char
        
    def evaluate_model(self, model, X_val, y_val, num_samples=20):
        """Evaluate model performance on validation set"""
        print("\nEvaluating model performance...")
        
        # Sample random validation examples
        indices = np.random.choice(len(X_val), min(num_samples, len(X_val)), replace=False)
        
        correct_predictions = 0
        total_predictions = 0
        
        for idx in indices:
            # Get prediction
            img_batch = np.expand_dims(X_val[idx], axis=0)
            prediction = model.predict(img_batch, verbose=0)
            predicted_text = self.decode_prediction(prediction[0])
            
            # Get ground truth
            true_text = self.decode_label(y_val[idx])
            
            # Compare
            if predicted_text.strip() == true_text.strip():
                correct_predictions += 1
            total_predictions += 1
            
            if idx < 5:  # Show first 5 examples
                print(f"True: '{true_text}' | Predicted: '{predicted_text}'")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\nValidation Accuracy on {total_predictions} samples: {accuracy:.2%}")
        
        return accuracy
    
    def decode_label(self, encoded_label):
        """Decode numerical label back to text"""
        decoded_chars = []
        for idx in encoded_label:
            if idx < len(self.num_to_char):
                char = self.num_to_char[idx]
                if char != '[PAD]':
                    decoded_chars.append(char)
        return ''.join(decoded_chars)

# Usage example for large dataset
def main():
    # Initialize OCR model
    ocr = NepaliOCRModel(img_height=64, img_width=256, max_length=30)
    
    # For testing with smaller subset, set sample_limit
    # For full training, remove sample_limit parameter
    print("Starting training with large dataset...")
    model, history = ocr.train_model(
        csv_path='./dataset/labels.csv',
        images_dir='./dataset/train',  # Update this to your images directory
        epochs=50,  # Reduced epochs for large dataset
        batch_size=8,  # Larger batch size for efficiency
        validation_split=0.15  # 15% for validation
    )
    
    if model is None:
        print("Training failed!")
        return
    
    print("Training completed successfully!")
    
    # Test prediction on some sample images
    try:
        test_images = ['1.jpg', '2.jpg', '3.jpg']
        for img_name in test_images:
            if os.path.exists(img_name):
                predicted_text = ocr.predict_text(model, img_name)
                print(f"Predicted text for {img_name}: {predicted_text}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Function to resume training from checkpoint
def resume_training():
    """Resume training from saved model"""
    ocr = NepaliOCRModel()
    
    # Load saved model
    model = keras.models.load_model('best_nepali_ocr_model.h5', 
                                   custom_objects={'ctc_loss': ocr.ctc_loss})
    
    # Load character mappings
    with open('char_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        ocr.char_to_num = mappings['char_to_num']
        ocr.num_to_char = mappings['num_to_char']
        ocr.characters = mappings['characters']
    
    # Continue training
    images, labels = ocr.load_data('./dataset/labels.csv', './dataset/train')
    encoded_labels = ocr.encode_labels(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, test_size=0.15, random_state=42
    )
    
    # Resume training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Additional epochs
        batch_size=32,
        initial_epoch=0  # Set this to the epoch you want to resume from
    )
    
    return model, history

# Performance monitoring function
def monitor_training_progress():
    """Monitor training progress from logs"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    try:
        # Read training log
        log_df = pd.read_csv('training_log.csv')
        
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(log_df['epoch'], log_df['loss'], label='Training Loss')
        plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        if 'lr' in log_df.columns:
            plt.subplot(1, 3, 2)
            plt.plot(log_df['epoch'], log_df['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
        
        # Show recent performance
        plt.subplot(1, 3, 3)
        recent_epochs = log_df.tail(20)
        plt.plot(recent_epochs['epoch'], recent_epochs['loss'], 'o-', label='Training Loss')
        plt.plot(recent_epochs['epoch'], recent_epochs['val_loss'], 'o-', label='Validation Loss')
        plt.title('Recent Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nTraining Summary:")
        print(f"Total epochs: {len(log_df)}")
        print(f"Best validation loss: {log_df['val_loss'].min():.4f}")
        print(f"Final training loss: {log_df['loss'].iloc[-1]:.4f}")
        print(f"Final validation loss: {log_df['val_loss'].iloc[-1]:.4f}")
        
    except FileNotFoundError:
        print("Training log not found. Make sure training has started.")
    except Exception as e:
        print(f"Error reading training log: {e}")

if __name__ == "__main__":
    main()

# Data augmentation function (optional)
def augment_data(csv_path, images_dir, output_dir):
    """Create augmented versions of training data"""
    import albumentations as A
    
    transform = A.Compose([
        A.Rotate(limit=5, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
    ])
    
    df = pd.read_csv(csv_path, header=None, encoding="utf-16")
    df.columns = ['filename', 'text']
    
    os.makedirs(output_dir, exist_ok=True)
    
    augmented_data = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(images_dir, row['filename'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            
            # Create 3 augmented versions of each image
            for i in range(3):
                augmented = transform(image=img)['image']
                
                # Save augmented image
                base_name = os.path.splitext(row['filename'])[0]
                aug_filename = f"{base_name}_aug_{i}.jpg"
                aug_path = os.path.join(output_dir, aug_filename)
                cv2.imwrite(aug_path, augmented)
                
                # Add to augmented data list
                augmented_data.append([aug_filename, row['text']])
    
    # Save augmented labels
    aug_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, 'augmented_labels.csv'), 
                      header=False, index=False)
    
    print(f"Created {len(augmented_data)} augmented samples")
