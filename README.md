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

### Transcription
FFmpeg extracts the audio, WhisperX transcribes it, and `leer_json.py` turns the output into the format agreed with the editor. Run `transcribir_videos.sh` with the videos in a `videos/` folder; transcripts are saved in `transcripciones/`.

### Gaze detection
Videos are split into one frame per second, faces are detected and cropped with a YOLOv8 face model, and each face is analysed with MediaPipe FaceMesh: pupil position, eye openness (Eye Aspect Ratio) and head pose estimated with OpenCV. A person counts as looking at the camera when both eyes are open, the head faces the camera and the pupils are centred.

Two approaches were evaluated: asking a multimodal model (LLaVA, via LangChain and Ollama) and the landmark-based approach above. LLaVA gave inconsistent answers, so the landmark-based approach was kept.

- `run_eye_head_detection.sh`: final approach (MediaPipe + OpenCV)
- `run_llava_detection.sh`: LLaVA approach, kept for comparison

### Questions about a video
`ask_videos.py` runs Video-LLaVA-7B over each video with the questions in `questions.txt` (one per line), using deterministic decoding so that answers for different takes can be compared. Run `run_ask_videos`; answers are saved in `responses/`.

### Scene detection and splitting
After benchmarking PySceneDetect's five detectors, `AdaptiveDetector` was chosen. `run_scene_detect.sh` saves the start and end time of each scene in `scenes-times/`, and `run_scene_split.sh` saves one clip per scene in `scenes-clips/`.

## Setup notes
- Input videos go in a `videos/` folder.
- The models have conflicting dependencies, so two separate Python virtual environments were used.
- Requires FFmpeg and, for LLaVA, a local Ollama installation.

## Author
Alba González Gómez
