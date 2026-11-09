# AI-Assisted Video Post-Production

Bachelor's thesis in Computer Engineering, University of Murcia (grade: 9.7/10).

A set of AI tools that take repetitive tasks off video editors. Each tool can be run on its own over a folder of videos and was built after evaluating alternative models and approaches.

## Tools

| Folder | Task | Main technologies |
| --- | --- | --- |
| `transcription/` | Audio transcription into an editor-ready format: timestamps, speakers, take markers ("Action!"/"Cut!"), notes and silences | FFmpeg, WhisperX |
| `detection/` | Detects whether anyone in the footage looks at the camera, and saves the frames that support the answer for review | FFmpeg, YOLOv8 (face detection), Supervision, MediaPipe FaceMesh, OpenCV |
| `questions/` | Answers questions about a video, mainly to compare different takes of the same scene | Video-LLaVA (Hugging Face Transformers), PyAV |
| `scenes/` | Detects scene changes in a master file and either lists their start/end times or splits the video into clips | PySceneDetect |

## How it works

All scripts are run from inside their own folder and read the input videos from a `videos/` folder at the root of the repository.

### Transcription
FFmpeg extracts the audio, WhisperX transcribes it, and a Python script turns the output into the format agreed with the editor.

- `transcribir_videos.sh` + `leer_json.py`: transcript with timestamps, take markers, notes and silences.
- `transcribir_videos_speaker.sh` + `leer_json_speaker.py`: same output, with the speaker label added to each line.

Transcripts are saved in `transcription/transcripciones/`.

### Gaze detection
Videos are split into one frame per second, faces are detected and cropped with a YOLOv8 face model (`face_detector.py`), and each face is analysed with MediaPipe FaceMesh: pupil position, eye openness (Eye Aspect Ratio) and head pose estimated with OpenCV. A person counts as looking at the camera when both eyes are open, the head faces the camera and the pupils are centred.

Two approaches were evaluated: asking a multimodal model (LLaVA, via LangChain and Ollama) and the landmark-based approach above. LLaVA gave inconsistent answers, so the landmark-based approach was kept.

- `run_eye_head_detection.sh`: final approach, runs `face_detector.py` and then `mp_eye_head_pose.py` (pupils, eye openness and head pose).
- `mp_eye_pose.py`: earlier version of the landmark-based approach, using pupils and eye openness only, without head pose.
- `run_llava_detection.sh`: LLaVA approach (`look_at_camera.py`), kept for comparison.

`yolov8n-face.pt` contains pretrained third-party weights for face detection; they were not trained as part of this project.

### Questions about a video
`ask_videos.py` runs Video-LLaVA-7B over each video with the questions listed in a `questions.txt` file inside `questions/` (one question per line, created by the user), using deterministic decoding so that answers for different takes can be compared. Run `run_ask_videos.sh`; answers are saved in `responses/`.

### Scene detection and splitting
After benchmarking PySceneDetect's five detectors, `AdaptiveDetector` was chosen. `run_scene_detect.sh` saves the start and end time of each scene in `scenes-times/`, and `run_scene_split.sh` saves one clip per scene in a `scenes-clips-<video name>/` folder for each video.

## Setup notes
- Python packages are listed in `requirements.txt`. Package versions were not pinned during development.
- The models have conflicting dependencies, so two separate Python virtual environments were used.
- Requires FFmpeg and, for the LLaVA approach, a local Ollama installation.

## Author
Alba González Gómez

