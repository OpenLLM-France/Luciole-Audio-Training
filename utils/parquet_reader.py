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
                        text_item = next((item for item in content_list if item.get("type") == "text"), None)
                        structured_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "array": audio_item.get("array") if audio_item else None,
                                    "path": audio_item.get("path") if audio_item else None,
                                    "sampling_rate": audio_item.get("sampling_rate") if audio_item else None,
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

    return output_data  # List of structured message lists

if __name__ == "__main__":
    import pprint
    path = "path/to/file.parquet"
    converted = convert_parquet_messages(path)
    pprint.pprint(converted[1])