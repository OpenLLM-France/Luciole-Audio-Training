import datasets
import random
import os
import pandas
import slugify
import glob

from luciole_audio.data.tts.tts_coqui import text_to_speech
from luciole_audio.data.audio import save_audio



def question_answering_data_iterator(debug_folder=None, max_question_len=300):

    REPOS = [

        # (
        #     ("CohereLabs/aya_collection", "aya_dataset"),
        #     "inputs",
        #     "targets",
        #     lambda x: x["language"] == "fra",
        # ),
        # (
        #     ("parquet", {"data_files": "comparia.parquet"}),
        #     "question",
        #     "answer",
        #     None,
        # ),

        # Downloaded from https://github.com/bofenghuang/vigogne/blob/main/data/instruct/alpaca_data_cleaned_fr_52k.jsonl
        (
            ("json", {"data_files": "alpaca_data_cleaned_fr_52k.jsonl"}),
            lambda x: x["instruction"] if not x["input"] else x["instruction"] + " " + x["input"],
            "output",
            None,
        ),


        # (
        #     ("jpacifico/French-Alpaca-dataset-Instruct-55K", "default"),
        #     lambda x: x["instruction"] if not x["input"] else x["instruction"] + " " + x["input"],
        #     "output",
        #     None,
        # ),



        # (
        #     ("PaDaS-Lab/webfaq", "fra", {"split": "default"}),
        #     "question",
        #     "answer",
        #     None,
        # ),

        # (
        #     ("RDTvlokip/InfiniQA", "default"),
        #     "question",
        #     "answer",
        #     None,
        # ),
        # (
        #     ("CohereLabs/aya_collection_language_split", "french"),
        #     "inputs",
        #     "targets",
        #     None,
        # ),
        # (
        #     ("json", {"data_files": glob.glob("vigogne/data/instruct/*_fr_*.jsonl")}),
        #     lambda x: x["instruction"] + " : " + x["input"] if "input" in x else x["instruction"],
        #     "output",
        #     None,
        # ),
    ]

    for repo, question_key, answer_key, filtering in REPOS:
        kwargs = {"split": "train", "streaming": True}
        if isinstance(repo, (tuple, list)) and isinstance(repo[-1], dict):
            kwargs.update(repo[-1])
            repo = repo[:-1]
        if isinstance(repo, str):
            repo = (repo)
        ds = datasets.load_dataset(*repo, **kwargs)

        yield repo[0].replace("json", "Vigogne")

        irow = 0
        for row in ds:
            if filtering and not filtering(row):
                continue
            if isinstance(question_key, str):
                question = row[question_key]
            else:
                question = question_key(row)
            if len(question) > max_question_len:
                continue
            if "_" in question:
                continue
            answer = row[answer_key]
            irow += 1
            print(f"Processing {repo}: {question} -> {answer}")
            yield make_data_instruct(
                question,
                answer,
                debug_folder=os.path.join(debug_folder, Slugify(repo[0])) if (debug_folder and irow < 10) else None
            )
            # if irow == 10:
            #     break # NOCOMMIT

def make_data_instruct(question, answer, debug_folder=None, sampling_rate=16_000):
    audio_instruction = text_to_speech(question, sampling_rate=sampling_rate, add_noise=True)
    
    audio_data = {"type": "audio", "array": audio_instruction, "sampling_rate": sampling_rate}

    if debug_folder:
        filename = f"{Slugify(question)[:50]}"
        # Dump audio for manual inspection (debug, check, ...)
        os.makedirs(debug_folder, exist_ok=True)
        audio_filename = os.path.join(debug_folder, filename + ".wav")
        save_audio(audio_filename, audio_instruction, sampling_rate=sampling_rate)


    return [
        {"role": "user", "content": [audio_data]},
        {"role": "assistant","content": [{"type": "text", "text": answer}]}
    ]

def Slugify(s):
    return slugify.slugify(s).capitalize()

def string_to_integer(s: str) -> int:
    return abs(hash(s))

def main_dump_parquet():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="out", help="Output folder")
    parser.add_argument("--max_docs", default=None, type=int, help="Maximum number of documents")
    parser.add_argument("--num_docs_per_parquet", default=200, type=int, help="Number of documents per parquet")
    parser.add_argument("--debug_folder", default=None, help="Debug folder")
    args = parser.parse_args()

    global messages, first_idx, last_idx, dataset_name, explicit_subname
    messages = []
    first_idx = last_idx = 0
    dataset_name = "data"
    explicit_subname = False
    def _flush():
        global messages, first_idx, last_idx, dataset_name, explicit_subname
        if messages:
            last_idx += len(messages)
            output_filename = os.path.join(
                args.output,
                f"QuestionAnswering--{dataset_name}--{first_idx:06d}-{last_idx:06d}.parquet"
            )
            print(f"Dumping {output_filename} ...")
            os.makedirs(args.output, exist_ok=True)
            pandas.DataFrame({"messages": messages}).to_parquet(output_filename)
            first_idx = last_idx
            messages = []

    dataset_name = None
    for i, data in enumerate(
        question_answering_data_iterator(
            debug_folder=args.debug_folder
        )):
        if args.max_docs and i >= args.max_docs:
            break
        if isinstance(data, str):
            explicit_subname = True
            new_dataset_name = data.replace("/", "--")
            if new_dataset_name != dataset_name:
                _flush()
                dataset_name = new_dataset_name
                # Reset indices
                first_idx = last_idx = 0
                # Pseudo-deterministic randomness for each dataset
                print(f"Resetting random seed for dataset {dataset_name}")
                random.seed(string_to_integer(dataset_name))
            continue
        messages.append(data)
        if len(messages) >= args.num_docs_per_parquet:
            _flush()
    _flush()


if __name__ == "__main__":
    main_dump_parquet()
