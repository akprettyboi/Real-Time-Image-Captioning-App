import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import threading
import queue
import time
from transformers import TFBlipForConditionalGeneration, BlipProcessor

class PIDController:
    def __init__(self, kp=0.6, ki=0.2, kd=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_update = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_update
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        self.prev_error = error
        self.last_update = current_time
        return output

class ImageCaptioner:
    def __init__(self, 
                 model_name="Salesforce/blip-image-captioning-base",
                 caption_queue_size=3,
                 process_interval=1.0):
        
        self.generate_caption_fn = self._generate_caption

        
        # Enable GPU memory growth to prevent OOM errors
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU acceleration enabled")
        
        # Load models
        print("Loading models...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = 'left'
        self.model = TFBlipForConditionalGeneration.from_pretrained(model_name)
        print("Models loaded successfully")
        
        # Enable mixed precision for faster inference
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        self.caption_queue = queue.Queue(maxsize=caption_queue_size)
        self.accuracy_queue = queue.Queue(maxsize=caption_queue_size)
        
        self.process_interval = process_interval
        self.is_processing = False
        self.processing_thread = None
        
        self.frame_queue = None
        self.pid = PIDController()
        self.target_accuracy = 0.85
        
        # Set generation parameters
        self.max_length = 30
        self.num_beams = 4
        self.gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        
        
        }

    def _preprocess_image(self, image):
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # # Convert BGR to RGB if needed
        # if len(image.shape) == 3 and image.shape[2] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # # Process image using the processor
        # inputs = self.processor(images=image, return_tensors="tf")
        # return inputs
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image using the processor
        inputs = self.processor(images=image, return_tensors="tf", padding=True)
        return inputs['pixel_values']

    @tf.function(experimental_relax_shapes=True)
    def _generate_caption(self, pixel_values):
        # outputs = self.model.generate(
        #     pixel_values,
        #     **self.gen_kwargs,
        #     return_dict_in_generate=True,
        #     output_scores=True
         outputs = self.model.generate(
        pixel_values=pixel_values,
        max_length=self.max_length,
        num_beams=self.num_beams,
        temperature=self.gen_kwargs["temperature"],
        top_p=self.gen_kwargs["top_p"],
        repetition_penalty=self.gen_kwargs["repetition_penalty"],
        return_dict_in_generate=True,
        output_scores=True
        )
         return outputs

    def _process_frames(self):
        batch_size = 1
        frames_buffer = []
        last_process_time = time.time()
        
        
        while self.is_processing:
            try:
                current_time = time.time()
                if current_time - last_process_time >= self.process_interval:
                    while len(frames_buffer) < batch_size and not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        frames_buffer.append(frame)
                    
                    if frames_buffer:
                        try:
                            # Process batch
                            processed_images = [self._preprocess_image(frame) for frame in frames_buffer]
                            
                            # Combine pixel values from processed images
                            batch_pixel_values = tf.concat([img['pixel_values'] for img in processed_images], axis=0)
                            
                            # Generate captions
                            outputs = self.generate_caption_fn(batch_pixel_values)
                            
                            # Decode captions
                            captions = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                            
                            # Calculate confidence scores
                            scores = tf.nn.softmax(outputs.scores[0], axis=-1)
                            accuracies = tf.reduce_max(scores, axis=-1).numpy()
                            
                            # Update queues
                            for caption, accuracy in zip(captions, accuracies):
                                self._update_caption_queue(caption, float(np.mean(accuracy)))
                            
                            # PID control
                            avg_accuracy = np.mean(accuracies)
                            error = self.target_accuracy - avg_accuracy
                            adjustment = self.pid.update(error)
                            self.process_interval = max(0.3, min(2.0, self.process_interval + adjustment))
                        
                        except Exception as e:
                            print(f"Error in caption generation: {e}")
                        
                        frames_buffer.clear()
                        last_process_time = current_time
                
                time.sleep(0.01)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {e}")
                continue

    def start_captioning(self, frame_queue):
        self.frame_queue = frame_queue
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _update_caption_queue(self, caption, accuracy):
        try:
            if self.caption_queue.full():
                self.caption_queue.get_nowait()
                self.accuracy_queue.get_nowait()
            
            self.caption_queue.put(caption, block=False)
            self.accuracy_queue.put(accuracy, block=False)
        except queue.Full:
            pass
    
    def get_latest_caption(self):
        try:
            caption = self.caption_queue.get_nowait()
            accuracy = self.accuracy_queue.get_nowait()
            return f"{caption} (Confidence: {accuracy:.2%})"
        except queue.Empty:
            return None
    
    def stop_captioning(self):
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join()
        
        while not self.caption_queue.empty():
            try:
                self.caption_queue.get_nowait()
                self.accuracy_queue.get_nowait()
            except queue.Empty:
                break