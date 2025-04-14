"""
This script is adapted from
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""

import os

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import glob
import json
import numpy as np
import argparse
import os
import requests

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig



def vllm_generate(message, model, model_url):
    """
    Simple function to chat with a reasoning-enabled model.

    Args:
        message (str): The message to send to the model
        model_url (str): The base URL of the model server
    """

    # Prepare the chat message
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 200,
    }

    # Make the API call
    response = requests.post(
        f"{model_url}/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload
    )


    if response.status_code == 200:
        result = response.json()
        reasoning = result["choices"][0]["message"].get("reasoning_content", "")
        content = result["choices"][0]["message"].get("content", "")

        # print(reasoning)
        # print(content)

        # if reasoning:
        #     print(f"Reasoning: {reasoning}")
        # print(f"Content: {content}")
        return content
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(
            self,
            args,
            needle="\n\nRemember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n",
            haystack_dir="PaulGrahamEssays",
            retrieval_question="what is the best thing to do in San Francisco?\n\nAnswer: The best thing to do in San Francisco is",
            results_version=1,
            context_lengths_min=1000,
            context_lengths_max=1048000,
            context_lengths_num_intervals=40,
            context_lengths=None,
            document_depth_percent_min=0,
            document_depth_percent_max=100,
            document_depth_percent_intervals=10,
            document_depth_percents=None,
            document_depth_percent_interval_type="linear",
            model_provider="LLaMa",
            model_name="",
            model_name_suffix=None,
            num_concurrent_requests=1,
            save_results=True,
            save_contexts=True,
            final_context_length_buffer=200,
            seconds_to_sleep_between_completions=None,
            print_ongoing_status=True,
            attn_load_dir=None,
            sparsity=0.5,
            simulation_length=50,
    ):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.args = args
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        if "/" in model_name:
            self.model_version = model_name.split("/")[-1]
        else:
            self.model_version = model_name
        if model_name_suffix is not None:
            self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if (
                    context_lengths_min is None
                    or context_lengths_max is None
                    or context_lengths_num_intervals is None
            ):
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied."
                )
            else:
                self.context_lengths = np.round(
                    np.linspace(
                        context_lengths_min,
                        context_lengths_max,
                        num=context_lengths_num_intervals,
                        endpoint=True,
                    )
                ).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if (
                    document_depth_percent_min is None
                    or document_depth_percent_max is None
                    or document_depth_percent_intervals is None
            ):
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied."
                )
            else:
                if document_depth_percent_interval_type == "linear":
                    self.document_depth_percents = np.round(
                        np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            num=document_depth_percent_intervals,
                            endpoint=True,
                        )
                    ).astype(int)
                elif document_depth_percent_interval_type == "sigmoid":
                    self.document_depth_percents = [
                        self.logistic(x)
                        for x in np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            document_depth_percent_intervals,
                        )
                    ]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals"
            )

        self.model_name = model_name

        # 采用llama的tokenizer
        self.enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = "results/" + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    context_length_met = result["context_length"] == context_length
                    depth_percent_met = result["depth_percent"] == depth_percent
                    version_met = result.get("version", 1) == self.results_version
                    model_met = result["model"] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if (
                            context_length_met
                            and depth_percent_met
                            and version_met
                            and model_met
                    ):
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            tokens_new_context = tokens_context[:insertion_point]

            print(f"Insertion at {insertion_point} / {len(tokens_context)}")
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        print("Final length context tokens", len(tokens_new_context))
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, "r") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context)

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length], skip_special_tokens=True)

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")


def evaluate(question,
             answer,
             model_predict,
             model,
             model_url):
    count = 0
    err = None
    while True:
        try:
            prompt = """给你一个问题、答案以及模型预测的结果，请对模型预测的结果进行打分，打分规则如下。
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        请直接输出分数。
        问题：{}
        答案：{}
        模型预测的结果：{}
        分数："""

            score = vllm_generate(prompt.format(question, answer, model_predict), model, model_url)
            print(score)
            return score
        except Exception as e:
            count += 1
            if count == 10:
                err = e
                break
        return "发生错误：" + str(err)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--s_len", metavar="N", type=int, default=0, help="a number")
    parser.add_argument("-e", "--e_len", metavar="N", type=int, default=128000, help="a number")
    parser.add_argument("--model_path", type=str, default=None, help="path to model")
    parser.add_argument("--model_name", type=str, default=None, help="name of model")
    parser.add_argument("--vllm_url", type=str, default=None, help="url for vllm")
    parser.add_argument(
        "--model_name_suffix", type=str, default=None, help="name of model"
    )
    parser.add_argument(
        "--model_provider", type=str, default="LLaMA", help="which model to use"
    )
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--simulation_length", type=int, default=50)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=40)
    parser.add_argument("--document_depth_percent_intervals", type=int, default=10)
    parser.add_argument("--context_lengths_min", type=int, default=1000)
    parser.add_argument("--context_lengths_max", type=int, default=1048000)
    parser.add_argument("--document_depth_percent_min", type=int, default=0)
    parser.add_argument("--document_depth_percent_max", type=int, default=100)

    parser.add_argument("--prefilling_chunk_size", type=int, default=None)

    parser.add_argument("--sparsity", type=float, default=0.5)

    parser.add_argument(
        "--method",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # model_name = "/nfs/FM/gongoubo/checkpoints/Qwen2___5-7B-Instruct-1M"

    args.context_lengths_max = args.e_len

    ht = LLMNeedleHaystackTester(
        args=args,
        model_name=args.model_name,
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_contexts=True,
        save_results=True,
        attn_load_dir=args.attn_load_dir,
        sparsity=args.sparsity,
        simulation_length=args.simulation_length,
        context_lengths_min=args.context_lengths_min,
        context_lengths_max=args.context_lengths_max,
        context_lengths_num_intervals=args.context_lengths_num_intervals,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        document_depth_percent_min=args.document_depth_percent_min,
        document_depth_percent_max=args.document_depth_percent_max,
    )

    # 文本的长度
    print(ht.context_lengths)
    # 插入的percent的位置
    print(ht.document_depth_percents)

    res = []

    # Save the context to file for retesting
    if not os.path.exists('results'):
        os.makedirs('results')

    errors = open('errors.json', 'w')
    error_count = 0
    for context_length in tqdm(ht.context_lengths, total=len(ht.context_lengths)):
        if context_length < args.s_len or context_length > args.e_len:
            continue
        for depth_percent in tqdm(ht.document_depth_percents, total=len(ht.document_depth_percents)):
            context_file_location = args.model_name.split("/")[-1].replace(".",
                                                             "_") + f"_len_{context_length}_depth_{int(depth_percent)}"

            if os.path.exists(f'results/{context_file_location}_results.json'):
                continue
            con = ht.generate_context(context_length, depth_percent)

            model_predict = vllm_generate(
                con + "\n" + "what is the best thing to do in San Francisco?\n\nAnswer: The best thing to do in San Francisco is",
                args.model_name,
                args.vllm_url,
            )

            if "发生错误" in model_predict:
                errors.write(json.dumps(f'results/{context_file_location}_results.json', ensure_ascii=False) + "\n")
                error_count += 1
                continue
            else:

                from rouge import Rouge

                rouger = Rouge()
                hypothesis = model_predict.replace("*", "").replace(".", "")
                reference = "eat a sandwich and sit in Dolores Park on a sunny day"
                print(hypothesis)
                print(reference)
                score = rouger.get_scores(hypothesis.lower(), reference.lower(),)[0]["rouge-l"]["r"]
                print(score)

                # score = evaluate("what is the best thing to do in San Francisco?",
                #                  "eat a sandwich and sit in Dolores Park on a sunny day.",
                #                  model_predict,
                #                  args.model_name,
                #                  args.vllm_url
                #                  )
                # print(score)
                # if "发生错误" in score:
                #     errors.write(json.dumps(f'results/{context_file_location}_results.json', ensure_ascii=False) + "\n")
                #     error_count += 1
                #     continue

            results = {
                'model': args.model_name,
                'context_length': int(context_length),
                'depth_percent': float(depth_percent),
                'version': "v1.0",
                'needle': "\n\nRemember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n",
                'model_response': model_predict,
                'score': score.replace("分数：", "") if isinstance(score ,str) else score,
            }


            # Save the result to file for retesting
            with open(f'results/{context_file_location}_results.json', 'w', buffering=-1) as f:
                json.dump(results, f, ensure_ascii=False)

    errors.close()
    print("总共有错误：", error_count)
