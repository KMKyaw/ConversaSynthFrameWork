# A Framework for Synthetic Audio Conversations Generation using Large Language Models

This is the repository for [A Framework for Synthetic Audio Conversations Generation using Large Language Models](https://kmkyaw.github.io/conversaSynth/index.html) by [Kaung Myat Kyaw](https://kmkyaw.github.io/) and Jonathan Hoyin Chan.

This repository contains a tool to generate synthetic conversations using user defined scenario and personas. The entire pipeline covers the process from generating text conversations using LLM to creating audio conversations using TTS models and saving annotations of each conversations.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Parameters](#parameters)
- [Pipeline Overview](#pipeline-overview)
- [Personas](#personas)
- [Annotation and Outputs](#annotation-and-outputs)
- [Background Noise](#background-noise)

## Requirements

For optimal performance, this program uses CUDA for generating voices and running TTS models. Ensure you have CUDA installed and properly configured. If your system doesn't support CUDA, the program will automatically fall back to the CPU, but with slower performance.

To check your CUDA version and availability in PyTorch, you can run:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

To install the required libraries, run the following:

```bash
pip install -r requirements.txt
```

## Usage

To generate conversations, convert them into audio, and save the results:

```bash
python3 generate.py --n <num_conversations> --o <output_folder> --min <min_personas> --max <max_personas>
```

### Example

```bash
python3 generate.py --n 5 --o conversation_outputs --min 2 --max 4
```

This command will generate 5 synthetic conversations with a random number of personas between 2 and 4, saving the output to the folder `conversation_outputs`.

### Parameters

- `--n`: Number of conversations to generate. Default: `1`. Must be a positive integer.
- `--o`: Output folder where the final conversations and audios will be stored. Default: `./conversations`.
- `--min`: Minimum number of personas to participate in the conversation. Default: `2`.
- `--max`: Maximum number of personas to participate in the conversation. Must be greater than `min`.

### Additional Notes

1.  **Invalid Persona Count**: If the `min` value is less than 2, or the `max` value is smaller than `min`, the script will raise an error.
2.  **Performance**: The audio generation and TTS conversion process can be time-consuming depending on the hardware, especially when working with multiple conversations and personas.

## Pipeline Overview

1.  **Conversation Generation**: Randomly selected personas engage in a conversation, which is saved as a `.json` file.
2.  **Unique Voice Generation**: Unique voices are generated for each persona
3.  **Text-to-Speech Conversion**: Each dialogue is converted into an audio file using unique persona voices.
4.  **Audio Concatenation**: The audio files for each persona are concatenated to form a complete conversation.
5.  **Annotations**: Time-based annotations are generated for the speakers and saved in a CSV file.
6.  [Optional] **Background Noise**: Adding background noises

## Personas

The conversations in this project are driven by a set of predefined personas, each with unique characteristics, personality traits, and speaking styles. These personas are defined in `personas.py` and used to add depth and variation to the generated conversations. Here is an overview of the main personas used:

### Class: Persona

Each persona is an instance of the `Persona` class, which is defined as follows:

```python
class Persona:
    def __init__(self, name, characteristics, personality, style):
        self.name = name
        self.characteristics = characteristics
        self.personality = personality
        self.style = style
```

#### Sample Personas

- **Alice**: Enthusiastic, Brave, Curious, and Optimistic. She is adventurous, always seeking new experiences, and inspiring others with her fearlessness.
- **Ben**: Intellectual, Introverted, Thoughtful, and Analytical. Ben is a deep thinker and a quiet observer, who enjoys meaningful discussions.
- **Cathy**: Humorous, Outgoing, Witty, and Charismatic. She brings laughter and energy to every conversation, making her a vibrant and magnetic personality.
- **David**: Imaginative, Creative, Idealistic, and Passionate. A dreamer and artist, David is constantly envisioning new ideas and passionately pursuing his creative endeavors.
- **Eva**: Compassionate, Sensitive, Supportive, and Nurturing. She is always in tune with the emotions of others and provides comfort and care to those around her.
- **Frank**: Energetic, Disciplined, Health-conscious, and Motivational. Frank is dedicated to health and fitness, motivating others to pursue a healthy lifestyle.
- **Grace**: Patient, Serene, Nature-loving, and Nurturing. Grace finds peace in nature and brings a calming presence to her interactions.
- **Henry**: Knowledgeable, Detail-oriented, Curious, and Meticulous. Henry loves history and spends his time researching and uncovering the mysteries of the past.
- **Isabella**: Inventive, Forward-thinking, Resourceful, and Determined. Isabella is always coming up with creative solutions and ideas, inspiring others to innovate.

These personas contribute to generating conversations with varied emotional depth, simulating real-world dialogue scenarios.

### List of Personas

All the personas defined in the project are stored in a list:

```python
personas = [alice, ben, cathy, david, eva, frank, grace, henry, isabella]
```

This list is used to randomly select personas for each conversation, ensuring a diverse set of dialogues.

## Annotation and Outputs

The program generates the following outputs:

- **Audio Files**: Each conversation is saved as a WAV file in the specified output folder.
- **Annotations**: A `annotations.csv` file is generated with the structure:
  - `filename`: The name of the conversation file.
  - `start`: The start time of the speaker's dialogue.
  - `end`: The end time of the speaker's dialogue.
  - `speaker`: The name of the persona speaking.

You can find the `annotations.csv` in the root folder after the process is complete.

## Background Noise

The `add_noise.py` script adds background noise to audio files to enhance realism.

### Usage

To run the script, use the following command:

```bash
python3 add_noise.py --i input_folder --n noise_file.wav --o output_folder
```

### Parameters

- `--i`: The folder containing the audio files to which noise will be added.
- `--n`: The path to the .wav file used as background noise.
- `--o`: The folder where the resulting audio files with added noise will be saved.

### Script Details

1.  **Load Noise File**: The script loads the specified noise file.
2.  **Process Each Audio File**: For each `.wav` file in the input directory:
    - The noise file is repeated if shorter than the audio file.
    - The noise is trimmed to match the length of the audio.
    - The noise is normalized and its volume is reduced randomly.
    - The noise is overlaid onto the original audio.
3.  **Save Output**: The combined audio (original audio with added noise) is saved in the specified output directory.

## Citation

If you find this useful for your research, please consider citing

```bibtex
@article{kyaw2024framework,
title={A Framework for Synthetic Audio Conversations Generation using Large Language Models},
author={Kyaw, Kaung Myat and Chan, Jonathan Hoyin},
journal={arXiv preprint arXiv:2409.00946},
year={2024}
}
```
