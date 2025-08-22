import argparse
from gradio_client import Client, handle_file
import os
import json

parse = argparse.ArgumentParser(description="test llm demo server.")

server_ip = "192.168.33.44" #"192.168.33.44"
username = "intellif"
password = "edge10"
port = 7867

def test_demo(url_path, prompt):
    client = Client(url_path, auth=(username, password))
    client.predict(api_name="/welcome")     
    result = client.predict(
        image=handle_file("./example1.jpg"),
        _chatbot=[],
        api_name="/upload_img"        
    )
    print(result)
    result = client.predict(
		_question=prompt,
		_chat_bot=[],
		params_form="Sampling",
		num_beams=3,
		repetition_penalty=1,
		top_p=0.001,
		top_k=1,
		temperature=0,
		api_name="/respond"
    )
    print(result)
    return True
        
prompt = "图片中有什么"
url_paths = ["http://192.168.33.44:7862/",
                 "http://192.168.33.44:7863/",
                 "http://192.168.33.44:7864/",
                 "http://192.168.33.44:7865/",
                 "http://192.168.33.44:7866/",
                 "http://192.168.33.44:7867/",
                 "http://192.168.33.44:7868/",
                 "http://192.168.33.44:7869/",
                 "http://192.168.33.44:7870/",
                 "http://192.168.33.44:7871/",
                 ]
for url_path in url_paths:
    try:
        result = test_demo(url_path, prompt)
    except:
        print(f'failed to connect to {url_path} !!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # if result == False:
    #     print(f"test failured!!!")
