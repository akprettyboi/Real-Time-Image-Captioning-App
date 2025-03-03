import cv2
import threading
import queue
import time
import numpy as np

class CameraHandler:
    def __init__(self, camera_index=0, buffer_size=5, frame_rate=30):
        self.camera_index = camera_index
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate
        
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.is_capturing = False
        self.capture_thread = None
        self.camera = None
        
        # Enhanced video parameters
        self.brightness = 1.1
        self.contrast = 1.2
        self.saturation = 1.1
        
        # Advanced sharpening kernel
        self.sharpness_kernel = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ]) / 8.0
    
    def _init_camera(self):
        """Initialize camera with multiple attempts and fallback options"""
        camera_backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,   # Media Foundation (Windows)
            cv2.CAP_ANY     # Any available
        ]
        
        for backend in camera_backends:
            try:
                print(f"Trying camera backend: {backend}")
                self.camera = cv2.VideoCapture(self.camera_index + backend)
                
                if not self.camera.isOpened():
                    print(f"Failed to open camera with backend {backend}")
                    continue
                
                # Try to read a test frame
                ret, frame = self.camera.read()
                if not ret:
                    print(f"Failed to read test frame with backend {backend}")
                    self.camera.release()
                    continue
                
                # If we got here, camera is working
                print(f"Successfully initialized camera with backend {backend}")
                
                # Configure camera settings
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced from 1920
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Reduced from 1080
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
                # Try different formats if MJPG fails
                for fourcc in ['MJPG', 'YUY2', 'I420']:
                    try:
                        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                        ret, frame = self.camera.read()
                        if ret:
                            print(f"Using format: {fourcc}")
                            break
                    except Exception as e:
                        print(f"Failed to set format {fourcc}: {e}")
                
                return True
                
            except Exception as e:
                print(f"Error initializing camera with backend {backend}: {e}")
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
        
        return False
    
    def start_capture(self):
        if not self.is_capturing:
            if not self._init_camera():
                raise RuntimeError("Failed to initialize camera with any backend")
            
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
    
    def _enhance_frame(self, frame):
        # Convert to float32 for better precision
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply advanced color enhancement
        frame_enhanced = cv2.convertScaleAbs(
            frame_float, 
            alpha=self.contrast, 
            beta=self.brightness - 1
        )
        
        # Apply sharpening
        frame_sharp = cv2.filter2D(frame_enhanced, -1, self.sharpness_kernel)
        
        # Apply selective noise reduction
        frame_denoised = cv2.fastNlMeansDenoisingColored(
            (frame_sharp * 255).astype(np.uint8),
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Enhance saturation
        hsv = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        frame_final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return frame_final
    
    def _capture_frames(self):
        consecutive_failures = 0
        max_failures = 5
        
        while self.is_capturing:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    consecutive_failures += 1
                    print(f"Failed to capture frame ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive failures, attempting to reinitialize camera")
                        self.camera.release()
                        if not self._init_camera():
                            print("Failed to reinitialize camera")
                            break
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                enhanced_frame = self._enhance_frame(frame)
                
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(enhanced_frame, block=False)
                except queue.Full:
                    pass
                
                time.sleep(1.0 / self.frame_rate)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
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