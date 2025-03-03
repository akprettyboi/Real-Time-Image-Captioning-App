import sys
import signal
import time
import tkinter as tk
from tkinter import scrolledtext
import cv2
from PIL import Image, ImageTk
import os

from camera_handler import CameraHandler
from image_captioner import ImageCaptioner

os.environ['DISPLAY'] = ':0'

class RealTimeCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Image Captioning")
        
        # Initialize camera handler
        self.camera_handler = CameraHandler(
            camera_index=0,  # Primary camera
            buffer_size=10,   # Frame buffer size
            frame_rate=30     # Frames per second
        )
        
        # Initialize image captioner
        self.image_captioner = ImageCaptioner(
            model_name="microsoft/git-base-coco",
            caption_queue_size=3,
            process_interval=2
        )
        
        # Create UI components
        self.create_ui()
        
        # Start camera and captioning
        self.start_capture()
    
    def create_ui(self):
        # Video display frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(padx=10, pady=10)
        
        # Caption display
        self.caption_label = tk.Label(self.root, text="Waiting for caption...", 
                                      font=("Arial", 12), wraplength=400)
        self.caption_label.pack(padx=10, pady=10)
        
        # Update method
        self.root.after(100, self.update_frame)
        self.root.after(1000, self.update_caption)
    
    def start_capture(self):
        # Start camera capture
        self.camera_handler.start_capture()
        
        # Start captioning process
        self.image_captioner.start_captioning(self.camera_handler.frame_queue)
    
    def update_frame(self):
        # Get latest frame
        frame = self.camera_handler.get_latest_frame()
        
        if frame is not None:
            # Convert frame for Tkinter
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Resize if needed
            pil_image = pil_image.resize((640, 480), Image.LANCZOS)
            
            # Convert to Tkinter-compatible image
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update video frame
            self.video_frame.config(image=tk_image)
            self.video_frame.image = tk_image
        
        # Schedule next update
        self.root.after(100, self.update_frame)
    
    def update_caption(self):
        # Get latest caption
        caption = self.image_captioner.get_latest_caption()
        
        if caption:
            # Update caption label
            self.caption_label.config(text=caption)
        
        # Schedule next caption check
        self.root.after(1000, self.update_caption)
    
    def on_closing(self):
        # Stop camera and captioning
        self.camera_handler.stop_capture()
        self.image_captioner.stop_captioning()
        
        # Close application
        self.root.destroy()

def main():
    root = tk.Tk()
    app = RealTimeCaptioningApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()