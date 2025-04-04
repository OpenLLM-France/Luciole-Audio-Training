import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def convert_parquet_messages(parquet_file):
    """
    Convert a list of messages with NumPy arrays in 'content' into the desired format.
    """
    table = pq.read_table(parquet_file)
    messages_df = table.to_pandas()

    output_data = []  # List to store all structured message groups

    for _, row in messages_df.iterrows():
        messages = row.iloc[0]  # Extract the column containing the message list (adjust if necessary)
        
        if isinstance(messages, np.ndarray):  # Ensure it's a NumPy array
            structured_messages = []  # Stores system, user, and assistant messages together
            
            for message in messages:
                if isinstance(message, dict):  # Ensure it's a dictionary
                    role = message.get("role")
                    content_list = list(message.get("content", []))

                    if role == "system":
                        system_prompt = next(
                            (item.get("text") for item in content_list if item.get("type") == "text"), None
                        )
                        structured_messages.append({
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}]
                        })

                    elif role == "user":
                        audio_item = next((item for item in content_list if item.get("type") == "audio"), None)
                        if audio_item:
                            if audio_item.get("array") is not None:
                                audio_array = audio_item.get("array")
                                sampling_rate = audio_item.get("sampling_rate")
                                audio_path = audio_item.get("path")
                            elif audio_item.get('type') == 'audio':
                                audio = audio_item.get("audio")
                                audio_data = audio.get("array")
                                
                                if isinstance(audio_data, list):  
                                    # Convert only if it's a valid list
                                    audio_array = np.array(audio_data, dtype=np.float32)
                                elif isinstance(audio_data, np.ndarray):
                                    audio_array = audio_data
                                else:
                                    audio_array = None
                                    
                                sampling_rate = audio.get("sampling_rate")
                                audio_path = audio.get("path")
                                audio_path = os.path.basename(audio_path)
                        else:
                            audio_data = None
                            sampling_rate = None        
                                
                        text_item = next((item for item in content_list if item.get("type") == "text"), None)
                        structured_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "array": audio_array,
                                    "path":audio_path,
                                    "sampling_rate": sampling_rate,
                                },
                                {"type": "text", "text": text_item.get("text") if text_item else None}
                            ]
                        })

                    elif role == "assistant":
                        output_text = next(
                            (item.get("text") for item in content_list if item.get("type") == "text"), None
                        )
                        structured_messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": output_text}]
                        })
            
            if structured_messages:
                output_data.append(structured_messages)  # Append the grouped messages together

    return output_data

if __name__ == "__main__":
    import pprint
    path = "path/to/file.parquet"
    converted = convert_parquet_messages(path)
    pprint.pprint(converted[1])