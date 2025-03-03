import cv2
import threading
import queue
import time
import numpy as np

class CameraHandler:
    def __init__(self, camera_index=0, buffer_size=10, frame_rate=30):
        self.camera_index = camera_index
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate
        
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.is_capturing = False
        self.capture_thread = None
        self.camera = None
        
        # Video enhancement parameters
        self.brightness = 1.2
        self.contrast = 1.3
        self.sharpness_kernel = np.array([[-1,-1,-1],
                                        [-1, 9,-1],
                                        [-1,-1,-1]])
    
    def start_capture(self):
        if not self.is_capturing:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set higher resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Enable camera hardware optimizations
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.is_capturing = True
            
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
    
    def _enhance_frame(self, frame):
        # Apply brightness and contrast adjustment
        enhanced = cv2.convertScaleAbs(frame, alpha=self.brightness, beta=5)
        
        # Apply sharpening
        enhanced = cv2.filter2D(enhanced, -1, self.sharpness_kernel)
        
        # Apply subtle noise reduction
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        return enhanced
    
    def _capture_frames(self):
        last_frame_time = time.time()
        frame_interval = 1.0 / self.frame_rate
        
        while self.is_capturing:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Enhance the captured frame
                enhanced_frame = self._enhance_frame(frame)
                
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(enhanced_frame, block=False)
                except queue.Full:
                    pass
                
                last_frame_time = current_time
            else:
                # Small sleep to prevent CPU overuse
                time.sleep(0.001)
    
    def get_latest_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
    
    def save_frame(self, frame, filename='latest_capture.jpg'):
        if frame is not None:
            # Save with high quality
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def stop_capture(self):
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join()
        
        if self.camera:
            self.camera.release()
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break