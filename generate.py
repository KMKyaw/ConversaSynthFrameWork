import argparse
from langchain_community.llms import Ollama
from personas import personas
import random
import json
import re
import argparse
import os
import time
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from datasets import load_dataset  # Importing the datasets module from Hugging Face
from tqdm import tqdm
from TTS.api import TTS
from pydub import AudioSegment
import pandas as pd
import soundfile as sf
from tqdm import tqdm

def extract_conversation(text):
    start_token = "[CONV_BEGIN]"
    end_token = "[CONV_END]"
    
    start_index = text.find(start_token) + len(start_token)
    end_index = text.find(end_token)
    
    if start_index == -1 or end_index == -1:
        return ""
    
    return text[start_index:end_index].strip()

def extract_conversation_to_jsonl(conversation):
    dialogue_list = []
    pattern = re.compile(r"\[(\w+)\]\s(.*?)\s*(?=\[\w+\]|$)")
    matches = pattern.findall(conversation)
    
    for name, dialogue in matches:
        dialogue_obj = {
            "name": name,
            "dialogue": dialogue.strip()
        }
        dialogue_list.append(json.dumps(dialogue_obj))
    
    return dialogue_list

# Function to generate and save conversation
def generate_and_save_conversation(file_number, folder, selected_personas):
    llm = Ollama(model='llama3')
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the prompt
    prompt = ""
    for persona in selected_personas:
        prompt += f"\n{persona.name}:\nCharacteristics: {', '.join(persona.characteristics)}\nPersonality: {persona.personality}\n"

    prompt += f""" They are all sitting around a table, having a lively and engaging conversation. Always place the whole story inside [CONV_BEGIN] and [CONV_END]. The order of the personas doesn't have to be in sequential order it could be random. When referring to each character, please put their name in square brackets. Do not use any shortened form or contraction. Follow the format of the following example: 
Example 1:
[CONV_BEGIN]

[{selected_personas[0].name}] I believe there is a lot to be discussed. 
[{selected_personas[1].name}] I agree!

[CONV_END]

Example 2:
[CONV_BEGIN]

[{selected_personas[0].name}] Sometimes, I think about my life being good.
[{selected_personas[1].name}] That is great! I envy you.

[CONV_END]

IMPORTANT : MAKE SURE THAT EVERYONE PARTICIPATES IN THE CONVERSATION
"""

    # Get response from LLM
    response = llm.invoke(prompt)
    #print(response)
    # Extract conversation
    conversation = extract_conversation(response)
    #print(conversation)
    # Convert to JSONL
    jsonl_conversation = extract_conversation_to_jsonl(conversation)
    #print(jsonl_conversation)
    # Save to file in the specified folder
    filename = os.path.join(folder, f"{file_number}.json")
    with open(filename, 'w') as file:
        for line in jsonl_conversation:
            file.write(line + "\n")

    # print(f"Conversation saved to {filename}")

def remove_empty_json_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    for file in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, file)
        
        # Check if the file is a JSON file
        if file.endswith('.json'):
            with open(file_path, 'r') as f:
                content = f.read().strip()
                # Check if the file is empty
                if not content:
                    os.remove(file_path)
                    print(f"Removed empty JSON file: {file_path}")

# Function to clean JSON files and remove unwanted files
def clean_and_filter_json_files(folder_path, personas_list):
    personas_list = [name.__name__ if isinstance(name, type) else str(name) for name in personas_list]
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Initialize a flag to check if the file contains any valid personas
            valid_persona_found = False
            
            # Filter out lines where dialogue is empty and check for valid personas
            cleaned_lines = []
            print('personas list:')
            print(personas_list)
            for line in lines:
                print(line)
                try:
                    json_obj = json.loads(line)
                    if json_obj.get("name") in personas_list:
                        valid_persona_found = True
                        print(f"found")
                        if json_obj.get("dialogue"):
                            cleaned_lines.append(line)
                except json.JSONDecodeError:
                    continue
            
            # If no valid persona is found, delete the file
            if not valid_persona_found:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                # Otherwise, write the cleaned lines back to the file
                with open(file_path, 'w') as file:
                    file.writelines(cleaned_lines)
                print(f"Cleaned file: {file_path}")

def remove_bounded_words(text):
    # Pattern to match text within *, (), [], {}
    pattern = r'\\u[0-9a-fA-F]{4}'
    return re.sub(pattern, '', text).strip()

def generate_unique_voices(selected_personas):
    file_counter = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
 
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    # Convert personas list to a dictionary for easy access
    persona_dict = {persona.name: persona for persona in selected_personas}

    output_directory = './unique_voices'
    os.makedirs(output_directory, exist_ok=True)
    # print("Starting to generate audio...")

    full_prompt = "Exploring the wonders of the natural world can be an incredibly enriching experience. From the vast, serene landscapes of national parks to the intricate details of a single leaf, nature offers an endless array of beauty and complexity. Observing wildlife in their natural habitats reveals fascinating behaviors and interactions that are often hidden from view. Each ecosystem, whether it’s a bustling coral reef or a quiet forest, plays a crucial role in maintaining the delicate balance of our planet. Understanding and appreciating these natural processes can inspire us to protect and preserve the environment for future generations. By fostering a deeper connection with nature, we not only enhance our own well-being but also contribute to the health of the Earth as a whole."
    # Iterate through each persona
    for persona_name, persona in persona_dict.items():
        file_counter += 1
        #print(f"#### Currently processing {persona_name} ####")
        
        # Retrieve the style from the persona
        description = persona.style if persona_name in persona_dict else "Default style if name not found"
        
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        
        # Prepare the prompt
        prompt_input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
        
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, max_new_tokens=4000)
        audio_arr = generation.cpu().numpy().squeeze()
        
        output_file_path = os.path.join(output_directory, f"{persona_name}.wav")
        sf.write(output_file_path, audio_arr, model.config.sampling_rate)

def generate_dialogue_audio(dialogue_folder, raw_folder, output_folder):
    # Get device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Init TTS model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Iterate over all JSON files in the dialogue folder
    for json_filename in os.listdir(dialogue_folder):
        if json_filename.endswith('.json'):
            dialogue_path = os.path.join(dialogue_folder, json_filename)
            
            # Extract dialogue index from the filename
            dialogue_index = os.path.splitext(json_filename)[0]
            
            # Read dialogues from JSON file
            with open(dialogue_path, 'r') as f:
                dialogues = [json.loads(line) for line in f]
            
            # Counter for the overall order of dialogues for the current JSON file
            overall_counter = 0

            # Iterate over each dialogue
            for dialogue in tqdm(dialogues, desc=f"Processing {json_filename}"):
                name = dialogue['name']
                text = dialogue['dialogue']
                text = remove_bounded_words(text)
                # Construct the path to the speaker's audio file
                speaker_wav = os.path.join(raw_folder, f"{name}.wav")
                
                # Ensure the speaker's audio file exists
                if not os.path.exists(speaker_wav):
                    print(f"Warning: {speaker_wav} does not exist. Skipping...")
                    continue
                
                # Construct the path for the output file
                output_wav_folder = os.path.join(output_folder, dialogue_index)
                output_wav = os.path.join(output_wav_folder, f"{name}_{overall_counter}.wav")
                
                # Ensure the output directory exists
                os.makedirs(output_wav_folder, exist_ok=True)
                print(text)
                print(speaker_wav)
                # Generate TTS audio
                tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_wav)
                
                # Increment the overall counter
                overall_counter += 1

def concatenate_audios(main_directory, output_directory):
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            # Create a list to store tuples of (filename, number)
            files_with_numbers = []

            # Iterate over all files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith('.wav'):
                    # Extract the number part from the filename
                    number_part = int(filename.split('_')[1].split('.')[0])
                    files_with_numbers.append((filename, number_part))

            # Sort the list of tuples by the number part
            files_with_numbers.sort(key=lambda x: x[1])

            # Initialize an empty AudioSegment
            combined_audio = AudioSegment.empty()

            # Concatenate the audio files in the sorted order
            for filename, _ in files_with_numbers:
                audio_path = os.path.join(subdir_path, filename)
                audio = AudioSegment.from_wav(audio_path)
                combined_audio += audio

            # Export the combined audio to a new file with the name of the subdirectory
            output_path = os.path.join(output_directory, f'{subdir}.wav')
            combined_audio.export(output_path, format='wav')

def annotate_data(dialogues_path):
    # List to store annotation data
    annotations = []

    # Iterate over each dialogue folder
    for dialogue_folder in os.listdir(dialogues_path):
        dialogue_path = os.path.join(dialogues_path, dialogue_folder)
        
        if os.path.isdir(dialogue_path):
            # Sort files to ensure correct order processing
            files = sorted([f for f in os.listdir(dialogue_path) if f.endswith('.wav')], key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            # Variable to keep track of the end time of the previous segment
            prev_end_time = 0
            
            for file_name in files:
                # Extract speaker and order from the filename
                speaker = file_name.split('_')[0]
                order = int(file_name.split('_')[1].split('.')[0])
                
                # Get the duration of the audio file
                file_path = os.path.join(dialogue_path, file_name)
                try:
                    audio, sr = sf.read(file_path)
                    duration = len(audio) / sr
                    
                    # Calculate start and end times
                    start_time = prev_end_time
                    end_time = start_time + duration
                    
                    # Append the annotation
                    annotations.append([f'conversation{dialogue_folder}.wav', start_time, end_time, speaker])
                    
                    # Update prev_end_time
                    prev_end_time = end_time
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Convert the annotations to a DataFrame and save to CSV
    annotations_df = pd.DataFrame(annotations, columns=['filename', 'start', 'end', 'speaker'])
    annotations_df.to_csv('annotations.csv', index=False)

def clean_filename(filename):
    # Split the filename to extract the number part
    parts = filename.split('conversation')
    if len(parts) > 1:
        return parts[1]
    return filename

# Main function to parse arguments and generate conversations
def main():
    parser = argparse.ArgumentParser(description="Generate conversations.")
    parser.add_argument("--n", type=int, default=1, help="Number of conversations to generate")
    parser.add_argument("--o", type=str, default="conversations", help="Folder to save the conversations")
    parser.add_argument("--min", type=int, default=2, help="Minimum number of personas participating in the conversation")
    parser.add_argument("--max", type=int, help="Maximum number of personas participating in the conversation")
    args = parser.parse_args()
    if args.min < 2:
        raise ValueError("Error: The value of min must be at least 2. (The default is 2)")
    if args.n < 1:
        raise ValueError("Error: The value of n must be at least 1. (The default is 1)")
    if args.min > args.max:
        raise ValueError("Error: The argument max must be always greater than min.")
    num_personas = random.randint(args.min, args.max)

    selected_personas = random.sample(personas, num_personas)
    start_time = time.time()
    for i in tqdm(range(1, args.n + 1), desc="Generating Text Dialogues"):
        generate_and_save_conversation(i, './dialogues', selected_personas)
    end_time = time.time()
    total_time = end_time - start_time
    print("Generating Text Dialogues : DONE")
    print(f"Total time taken for generating dialogues: {total_time:.2f} seconds")
    print("Cleaning Json files...")
    remove_empty_json_files('./dialogues')
    clean_and_filter_json_files('./dialogues', personas)
    print("Cleaning Json files : DONE")
    print("Generating Unique voices...")
    generate_unique_voices(selected_personas)
    print("Generating Unique voices : DONE")
    os.makedirs(args.o, exist_ok=True)
    print("Generating Dialogue Audios...")
    start_time = time.time()
    os.makedirs('./dialogue_audios', exist_ok=True)
    generate_dialogue_audio('./dialogues', './unique_voices', './dialogue_audios')
    end_time = time.time()
    itotal_time = end_time - start_time
    print("Generating Dialogue Audios : DONE")
    print(f"Total time taken for generating dialogue audios : {total_time:.2f} seconds")
    concatenate_audios('./dialogue_audios', args.o)
    annotate_data('./dialogue_audios')
    df = pd.read_csv('annotations.csv')
    df['filename'] = df['filename'].apply(clean_filename)
    df.to_csv('annotations.csv', index=False)
    print('Annotations saved to annotations.csv')
    print(f'Synthetic conversation audios saved to folder {args.o}')
    print("Generating Synthetic Conversations Audios: DONE")
if __name__ == "__main__":
    main()

