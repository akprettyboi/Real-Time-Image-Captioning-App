# Real-Time Image Captioning App

A sophisticated real-time image captioning application that uses deep learning to generate natural language descriptions of camera feed in real-time. The application combines computer vision and natural language processing to provide instant, accurate descriptions of what your camera sees.

## Features

- Real-time video capture and processing
- Advanced deep learning-based image captioning using state-of-the-art models
- Adaptive processing with PID controller for optimal performance
- Modern Tkinter-based GUI interface
- GPU acceleration support for faster inference
- Confidence score display for generated captions
- Keyboard shortcuts for easy control

## Technology Stack

- **Python 3.x**
- **Deep Learning Framework**: PyTorch/TensorFlow
- **Image Captioning Models**: 
  - Microsoft GIT (git-base-coco)
  - Salesforce BLIP
- **Computer Vision**: OpenCV, PIL
- **GUI**: Tkinter
- **Additional Libraries**: transformers, numpy

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- Webcam or camera device

### Setup

1. Clone the repository:
```bash
git clone https://github.com/akproprettyboi/image_captioning.git
cd image_captioning
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. The application window will open with your camera feed and real-time captions.

### Keyboard Shortcuts

- `Space`: Toggle capture
- `Esc`: Quit application

## Project Structure

```
image_captioning/
├── src/
│   ├── main.py              # Main application entry point
│   ├── image_captioner.py   # Image captioning model implementation
│   └── camera_handler.py    # Camera capture and processing
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## How It Works

1. **Camera Handling**: The `CameraHandler` class manages video capture and maintains a frame buffer for smooth processing.

2. **Image Captioning**: The `ImageCaptioner` class:
   - Uses pre-trained models (GIT/BLIP) for generating captions
   - Implements PID control for adaptive processing
   - Maintains caption and accuracy queues for smooth output

3. **GUI Interface**: The `RealTimeCaptioningApp` class:
   - Displays the camera feed
   - Shows real-time captions with confidence scores
   - Handles user input and application control

## Performance Optimization

- GPU acceleration when available
- PID controller for adaptive processing intervals
- Frame buffering for smooth video display
- Mixed precision inference for faster processing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the GIT model
- Salesforce for the BLIP model
- The open-source community for various tools and libraries

## Contact

Your Name - [@akproprettyboi](https://github.com/akproprettyboi)

Project Link: [https://github.com/akproprettyboi/image_captioning](https://github.com/akproprettyboi/image_captioning)