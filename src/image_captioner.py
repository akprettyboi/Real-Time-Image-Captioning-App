import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import threading
import queue
import time

class PIDController:
    def __init__(self, kp=0.5, ki=0.1, kd=0.1):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain 
        self.kd = kd  # Derivative gain
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # PID calculation
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class ImageCaptioner:
    def __init__(self, 
                 model_name="microsoft/git-base-coco", 
                 caption_queue_size=3, 
                 process_interval=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        self.caption_queue = queue.Queue(maxsize=caption_queue_size)
        self.accuracy_queue = queue.Queue(maxsize=caption_queue_size)
        
        self.process_interval = process_interval
        self.is_processing = False
        self.processing_thread = None
        
        self.frame_queue = queue.Queue(maxsize=5)
        self.pid = PIDController()
        self.target_accuracy = 0.8  # Target accuracy threshold
    
    def start_captioning(self, frame_queue):
        self.frame_queue = frame_queue
        
        self.is_processing = True
        
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_frames(self):
        while self.is_processing:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                pil_image = self._convert_frame_to_pil(frame)
                
                caption, accuracy = self._generate_caption(pil_image)
                
                # Use PID to adjust processing based on accuracy
                error = self.target_accuracy - accuracy
                adjustment = self.pid.update(error)
                
                # Adjust process interval based on PID output
                adjusted_interval = max(0.5, self.process_interval + adjustment)
                
                self._update_caption_queue(caption, accuracy)
                
                time.sleep(adjusted_interval)
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {e}")
    
    def _convert_frame_to_pil(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def _generate_caption(self, image):
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            caption = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Calculate accuracy/confidence score
            scores = outputs.scores[0]
            probabilities = torch.nn.functional.softmax(scores, dim=-1)
            accuracy = torch.max(probabilities).item()
            
            return caption, accuracy
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Unable to generate caption", 0.0
    
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