import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import random
import time
from typing import Optional
import logging
import numpy as np 
import heapq
import ollama
import re
from src.utils import resize_image
from src.const import *
END_POINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_KEY = "sk-79813f085233441f8b86bac4cb411786"

client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)
times = 0
all_time = 0
def get_ollama_response(sys_prompt, contents):
    max_tries = 2
    retry_count = 0
    formated_content = format_content_llama_img(sys_prompt, contents)

    message = ""
    for c in formated_content:
        message += c['content']
        if 'images' in c.keys():
            message += f"[{c['images'][0][:10]}...]"
    #logging.info(message)
    a = time.time()
    try:
        response = ollama.chat(
            model='gpt-oss:120b',
            messages=formated_content,
            options={
                "temperature": 0.7,
            }
        )
        #logging.info(f"Raw Response: {response.message.content}")
        b = time.time()
        global times, all_time
        times += 1
        all_time += (b - a)
        #logging.info(f"Average time per response: {all_time / times:.2f} seconds")
        return response.message.content
    except Exception as e:
        print("Error: ", e)
    return None


def format_content_llama_img(sys_prompt, contents):
    formated_content = []
    formated_content.append({"role": "system", "content": sys_prompt})
    
    for c in contents:
        if len(c) == 2:
            formated_content.append(
                {
                    "role": "user",
                    "content": c[0],
                    "images": [c[1]],
                }
            )
        else:
            formated_content.append({"role": "user", "content": c[0]})
    return formated_content

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formated_content

# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="qwen-vl-max",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                presence_penalty=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None

def call_openai_api_sg(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="qwen3-235b-a22b-instruct-2507",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                presence_penalty=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None

def save_image(image, save_path):
    """Save an image from various formats (base64, PIL, numpy) to a file"""
    img_data = base64.b64decode(image)
    img = Image.open(BytesIO(img_data))
    img.save(save_path)

def encode_tensor2base64(img, min_size=16):
    if min_size is not None:
        if (type(img) == np.ndarray):
            img = Image.fromarray(img)
        width, height = img.size
        if min(width, height) < min_size:
            scale = min_size / min(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal

def get_edge_relationship(prompt):
    sys_prompt = f"You are an agent in an indoor scene who can observe the environment and explore to find a target object. You must choose an Image or an Object as the answer, in order to find the specified target object."
    response =  call_openai_api_sg(sys_prompt, prompt)
    try:
        response = response.strip()
        response = re.sub(r'\n+', '\\n', response)   # Remove consecutive blank lines
        response = re.sub(r'\n$', '', response)      # Remove trailing blank line
        response = re.sub(r'^\n', '', response)      # Remove leading blank line
        response = response.lower()
        
    except:
        response = None
    return response

def get_step_info(step, verbose=False, use_ollama = False, use_room_filter = False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information (egocentric and frontier)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))
            

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # 2.3 get objects
    objs = step['objects']
    edges = step['edges']
    images = step['all_imgs']
    prompt_h = step['prompt_h']
    prompt_w = step['prompt_w']
    image_to_edges = step['image_to_edges']
    selected_obj_id = prefiltering_objs(
        question,
        objs,
        edges,
        step["top_k_categories"],
        image_goal,
        verbose=verbose,
        use_ollama=use_ollama,
        use_room_filter = use_room_filter
    )
    
    selected_objs, selected_edges, processed_images = get_prefiltering_sg(edges, objs, images, selected_obj_id, image_to_edges, prompt_h, prompt_w)

    return (
        question,
        image_goal,
        egocentric_imgs,
        selected_objs,
        selected_edges,
        processed_images,
        frontier_imgs,
    )


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    selected_objs,
    selected_edges,
    processed_images,
    egocentric_view = False,
    use_snapshot_class = True,
    image_goal = None,
    history_decision = None,
    room_label = False,
    task_type = None,
):
    if history_decision is not None:
        sys_prompt = f"You are an agent in an indoor scene who can observe the environment and explore to find a target object. You must choose an Image, a Frontier (for further exploration), or directly select an Object as the answer, in order to find the specified target object within {history_decision['max_step']} steps.\n"
    else:
        sys_prompt = "You are an agent in an indoor scene who can observe the environment and explore to find a target object. You must choose an Image, a Frontier (for further exploration), or directly select an Object as the answer, in order to find the specified target object.\n"
    content = []
    text = "Scene Graph Definitions:\n"
    text += """    Objects Attribution: Each object's name, ID, position, and location.
    Relationship Attribution: Relationships between objects, often with a reference image.
    Image List: Supplementary images from the nodes of scenegraph .
    Frontier: Unexplored regions that may provide new information.
    History Decision: All previous choices you made (avoid repeating them).
"""
    text += f"Here is the Question you need to solve: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))
    text = """Instructions:
    Step 1: Examine the Objects Attribution section. If the target object is explicitly listed (by name or ID), and fits the question, select it immediately as the answer.
    Step 2: If the target object is not explicitly in Objects Attribution, check the Relationship Attribution section. Use relationships and referenced images in the Image List to help you identify the target.
    Step 3: If you still cannot identify the object, select a relevant Frontier to explore and gather more information.
    Step 4: Provide your answer in one of the following formats:
        'Object i': If the object is found in Objects Attribution.
        'Image i, j': If the object is likely to exist in image i, and j is the required object category. Ensure you include the category name.
        'Frontier i': If the object is not found and further exploration is necessary.
Additional Notes:
    1. Try not to select any object, image that is in the "History Decision" list unless you are very confident.
    2. The detected class name and located room may be inaccurate, use the images to verify.
    3. Only provide the required answer (with optional brief reasoning in a new line).
    4. If there are no frontier to choose, you must choose the most likely Object or Image even without enough information.
Here is the example for the answer:
    Object 73
    I chose Object 73 because it is labeled as a refrigerator, misidentified as office room but images confirm it's in the kitchen.
or
    Image 3, refrigerator
    The image shows the refrigerator next to a kitchen cabinet, which matches the question.
or
    Frontier 0
    The object cannot be determined from the current information, so I will explore Frontier 0, which show direction to kitchen.
"""
    content.append((text,))


    # 4 format 3D scene graph images
        # ------------------format an 3D scene graph-------------------------
    text = "The 3D scene graph is given as this format :\n"
    text += "Objects Attribution:\n    Object Name: ID, Position(X,Y,Z)"
    if room_label:
        text += ", Located Room"
    text += "\n"
    text += "Relationship Attribution:\n    ID of Object1, ID of Object2, Relationship (a Image ID, you need to utilize the images to judge)\n"
    text += "Image List:\n    Image ID: Image\n"
    content.append((text,))
    
    text = "The followings the concrete content of the task.\n"
    text += "The followings is the 3D scene graph you can observe.\nObjects Attribution:\n"
    for id in selected_objs.keys():
        obj = selected_objs[id]
        text += f"    {obj['class_name']}: {obj['id']}, ({obj['bbox'].center[0]:.2f}, {obj['bbox'].center[2]:.2f}, {obj['bbox'].center[1]:.2f})"
        if room_label:
            text += f", {obj['room_label']}"
        text += "\n"
    if len(selected_objs) == 0:
        text += "    No Objects in the 3D scene graph.\n"
    text += "Relationship Attribution:\n"

    image_map = {}
    image_map_reverse = {}
    for img_key in processed_images.keys():
        idx = len(image_map)
        image_map[img_key] = idx
        image_map_reverse[idx] = img_key
    for node in selected_edges.keys():
        edge = selected_edges[node]
        text += f"    {node[0]}, {node[1]}, ["
        Flag = False
        for node_img in edge.rel_img:
            if node_img in processed_images.keys():
                if Flag:
                    text += ", "
                text += f"Image {image_map[node_img]}"
                Flag = True
        text += f"]\n"
    # 3 here is the egocentric views
    
    if len(selected_edges) == 0:
        text += "    No Relationship in the 3D scene graph.\n"
    text += "Image List:\n"
    content.append((text,))
    idx = -1
    for idx in range(len(image_map)):
        img_id = image_map_reverse[idx]
        content.append((f"    Image {idx} ", processed_images[img_id]))
        content.append(("\n",))
    if egocentric_view:
        for i in range(len(egocentric_imgs)):
            content.append((f"    Image {idx + i + 1} ", egocentric_imgs[i]))
            content.append(("\n",))
    
    
    # 5 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available.\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"    Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))
    if history_decision is not None:
        text = f"The followings are all the previous Decisions that you made: (now step is {history_decision['cnt_step']}/{history_decision['max_step']}). Choosing those incorrect objects or images again is prohibited:\n"
        have_decision = False
        for step in history_decision.keys():
            if type(history_decision[step]) is not dict or 'target_type' not in history_decision[step]:
                continue
            text += f"    History Decision type in step {step} : "
            have_decision = True
            if history_decision[step]['target_type'] == "object":
                text += f" Choosing Object {history_decision[step]['max_point_choice']} as answer, but not correct.\n"
            elif history_decision[step]['target_type'] == "image":
                ID = history_decision[step]['max_point_choice']
                if ID in image_map:
                    maped_ID = image_map[ID]
                else:
                    maped_ID = "unknown"
                text += f"Choosing Object in Image {maped_ID} as answer, but not correct.\n"
            elif history_decision[step]['target_type'] == "frontier":
                text += f"Choosing a Frontier to explore.\n"
        if not have_decision:
            text += "    No previous decisions made.\n"
            

        content.append((text,))
    # 6 here is the format of the answer
    text = "Answer: \n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))
    
    return sys_prompt, content, image_map_reverse


def format_obj_prompt(
    question,
    egocentric_imgs,
    selected_objs,
    selected_edges,
    processed_images,
    egocentric_view = False,
    image_goal = None,
    history_decision = None,
    room_label = False,
    task_type = None,
):
    if history_decision is not None:
        sys_prompt = f"You are an agent in an indoor scene who can observe the environment and explore to find a target object. You must choose an Image or an Object as the answer, in order to find the specified target object within {history_decision['max_step']} steps.\n"
    else:
        sys_prompt = "You are an agent in an indoor scene who can observe the environment and explore to find a target object. You must choose an Image or an Object as the answer, in order to find the specified target object.\n"
    content = []
    text = "Scene Graph Definitions:\n"
    text += """    Objects Attribution: Each object's name, ID, position, and location.
    Relationship Attribution: Relationships between objects.
    Image List: Supplementary images from surroundings of agent views.
    Frontier: Unexplored regions that may provide new information.
    History Decision: All previous choices you made (avoid repeating them).
"""
    text += f"Here is the Question you need to solve: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))
    text = """Instructions:
    Step 1: Examine the Objects Attribution section. If the target object is explicitly listed (by name or ID), and fits the question, select it immediately as the answer.
    Step 2: If the target object is not explicitly in Objects Attribution, check the Relationship Attribution section. Use relationships and referenced images in the Image List to help you identify the target.
    Step 3: If you still cannot identify the object, select to further explore and gather more information.
    Step 4: Provide your answer in one of the following formats:
        'Object i': If the object is found in Objects Attribution.
        'Image i, j': If the object is likely to exist in image i, and j is the required object category. Ensure you include the category name.
        'Continue Exploration': If the object is not found and further exploration is necessary.
Additional Notes:
    1. Try not to select any object, image that is in the "History Decision" list unless you are very confident.
    2. The detected class name and located room in the Objects Attribution may be inaccurate, use the images to verify.
    3. Only provide the required answer (with optional brief reasoning in a new line).
Here is the example for the answer:
Object 73
I chose Object 73 because it is labeled as a refrigerator, misidentified as office room but images confirm it's in the kitchen.
or
Image 3, refrigerator
    The image shows the refrigerator next to a kitchen cabinet, which matches the question.
or
Continue Exploration
The object cannot be determined from the current information, so I need to further explore.
"""
    content.append((text,))


    # 4 format 3D scene graph images
        # ------------------format an 3D scene graph-------------------------
    text = "The 3D scene graph is given as this format :\n"
    text += "Objects Attribution:\n    Object Name: ID, Position(X,Y,Z)"
    if room_label:
        text += ", Located Room"
    text += "\n"
    text += "Relationship Attribution:\n    ID of Object1, ID of Object2, Relationship (a Image ID, you need to utilize the images to judge)\n"
    text += "Image List:\n    Image ID: Image\n"
    content.append((text,))
    
    text = "The followings the concrete content of the task.\n"
    text += "The followings is the 3D scene graph you can observe.\nObjects Attribution:\n"
    for id in selected_objs.keys():
        obj = selected_objs[id]
        text += f"    {obj['class_name']}: {obj['id']}, ({obj['bbox'].center[0]:.2f}, {obj['bbox'].center[2]:.2f}, {obj['bbox'].center[1]:.2f})"
        if room_label:
            text += f", {obj['room_label']}"
        text += "\n"
    if len(selected_objs) == 0:
        text += "    No Objects in the 3D scene graph.\n"
    text += "Relationship Attribution:\n"

    image_map = {}
    image_map_reverse = {}

    for node in selected_edges.keys():
        edge = selected_edges[node]
        text += f"    {node[0]}, {node[1]}, [{edge.caption}]\n"
    # 3 here is the egocentric views
    
    if len(selected_edges) == 0:
        text += "    No Relationship in the 3D scene graph.\n"
    text += "The following are surrounding observations about the egocentric view of the agent :\n"
    text += "Please note that the class name and Located room may not be accurate due to the limitation of the detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    idx = -1

    if egocentric_view:
        for i in range(len(egocentric_imgs)):
            content.append((f"    Image {idx + i + 1} ", egocentric_imgs[i]))
            content.append(("\n",))
    
    
    if history_decision is not None:
        text = f"The followings are all the previous Decisions that you made: (now step is {history_decision['cnt_step']}/{history_decision['max_step']}). Choosing those incorrect objects or images again is prohibited:\n"
        have_decision = False
        for step in history_decision.keys():
            if type(history_decision[step]) is not dict or 'target_type' not in history_decision[step]:
                continue
            text += f"    History Decision type in step {step} : "
            have_decision = True
            if history_decision[step]['target_type'] == "object":
                text += f" Choosing Object {history_decision[step]['max_point_choice']} as answer, but not correct.\n"
            elif history_decision[step]['target_type'] == "image":
                ID = history_decision[step]['max_point_choice']
                if ID in image_map:
                    maped_ID = image_map[ID]
                else:
                    maped_ID = "unknown"
                text += f"Choosing Object in Image {maped_ID} as answer, but not correct.\n"
            elif history_decision[step]['target_type'] == "frontier":
                text += f"Choosing a Frontier to explore.\n"
        if not have_decision:
            text += "    No previous decisions made.\n"
            

        content.append((text,))
    # 6 here is the format of the answer
    text = "Answer: \n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))
    
    return sys_prompt, content, image_map_reverse


def format_exploreonly_prompt(
    question,
    frontier_imgs,
    image_goal = None
):
    content = []
    sys_prompt = "You are an agent tasked with finding a target object in an indoor scene. Your mission is to choose the most promising frontier for further exploration to locate the specified target object.\n"
    text = f"Here is the Question you need to solve: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))
    
    # 5 here is the frontier images
    text = "The Frontiers that you can explore:\n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available.\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"    Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    text = "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    text += """The example for the answer:
Frontier 0
I chose Frontier 0 for exploration, which show direction to kitchens where refrigerators are more likely to appear"""
    content.append((text,))
    
    return sys_prompt, content


def format_end_prompt(
    question,
    egocentric_imgs,
    image_goal=None,
):

    # base_dir = "/data0/hsun/7-15/3D-Mem/vis"
    # timestamp = int(time.time.time() * 1000)  # Milliseconds for uniqueness
    # folder_name = f"check_{question[-40:]}_{timestamp}"
    # folder_path = os.path.join(base_dir, folder_name)
    # os.makedirs(folder_path, exist_ok=True)
    
    # # Save goal image if provided
    # if image_goal is not None:
    #     goal_path = os.path.join(folder_path, "goal_image.png")
    #     save_image(image_goal, goal_path)
    
    # # Save surrounding images
    # for i in range(len(egocentric_imgs) - 1, -1, -1):
    #     surround_path = os.path.join(folder_path, f"surround_{i+1}.png")
    #     save_image(egocentric_imgs[i], surround_path)

    # log_file = os.path.join(folder_path, "log.txt")
    # with open(log_file, 'w') as f:
    #     f.write(f"Here is the Question: {question}\n")

    sys_prompt = "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. You are tasked with indoor navigation, and you are required to choose a Image or a Frontier to explore, or directly select an Object, finally find the target object required in the question.\n"

    content = []
    text = "Definitions:\n"
    text += (
    "Now that you have arrived near the previously selected answer, please observe your surroundings in the last five steps and confirm whether you have really reached the object required by the Question (close enough, less than 1m).\n"
    )
    # text += (
    # "NOTE: Due to limited viewpoints, you must relax your standards for preliminary checking to avoid rejection of a successful reach.\n"
    # )
    text += f"Here is the Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = (
        f"The following are surrounding observations about the egocentric view of the agent: \n"
    )
    content.append((text, ))
    for idx, img in enumerate(egocentric_imgs):
        content.append((f"    Surrounding observation {idx + 1} ", img))
        content.append(("\n",))
    # 6 here is the format of the answer
    text = "Answer: (Yes or No)\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))
    
    return sys_prompt, content

def format_prefiltering_prompt(question, scene_graph, top_k=10, image_goal=None, room_label = False):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    
    prompt = """To efficiently solve the problem,  you should identify key objects that are most helpful for guiding exploration toward the target.
Please follow these strict instructions:
1. Read and understand the full 3D scene graph. Each object includes its id, class, room, and nearby objects (i.e., its neighbors in the graph).
2. Rank objects by how helpful they are for locating the target, based on:
  Semantic relevance to the target; Co-occurrence with the target in typical environments; Presence in the same room as the target.
3. Choose only the most informative and strategically diverse objects for exploration. To maximize coverage: Avoid choosing objects that are directly connected (i.e., neighbors) in the scene graph."""
    content.append((prompt,))
    # ------------------format an 3D scene graph-------------------------
    prompt = "Here is is the format for input 3D scene graph:\n"
    prompt += "Object ID: Class"
    if room_label:
        prompt += ", Located room"
    prompt += ", nearby objects ID\n"
    # prompt += "Here is an example of input 3D scene graph:\n"
    # prompt += "1: tv, (1.4642, 1.7104, 1.0089), living_room, [2, 3]\n"
    # prompt += "2: speaker, (3.3615,1.4266,-1.1037), living_room, [1, 3]\n"
    # prompt += "3: sofa, (0.48304,1.9136,2.5941), living_room, [1, 2]\n"
    content.append((prompt,))

    # # ------------------Task to solve----------------------------
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects in 3D scene graph:\n"
    prompt += "Question: \nWhat can I use to watch my favorite shows and movies?\n"
    if not room_label:
        prompt += (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        prompt += "1: tv, (1.46, 1.71, 1.00), [2, 3]\n"
        prompt += "2: speaker, (3.36, 1.42, -1.10), [1, 3]\n"
        prompt += "3: sofa, (0.48, 1.91, 2.59), [1, 2]\n"
        prompt += "4: bed, (2.42, 1.04, 3.89), [5]\n"
        prompt += "5: lamp, (1.15, 1.66, 1.37), [4]\n"
        prompt += "6: box, (0.30, 1.36, 0.41), [7]\n"
        prompt += "7: cabinet, (2.01, 1.52, 2.08), [6]\n"
        prompt += "Answer:\n1\n5\n"
    else:
        prompt += (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        prompt += "1: tv, living room, [2, 3]\n"
        prompt += "2: speaker, living room, [1, 3]\n"
        prompt += "3: sofa, living room, [1, 2]\n"
        prompt += "4: bed, bedroom, [5]\n"
        prompt += "5: lamp, bedroom, [4]\n"
        prompt += "6: box, kitchen, [7]\n"
        prompt += "7: cabinet, kitchen, [6]\n"
        prompt += "Answer:\n1\n5\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful key objects in order."
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is the 3D scene graph based on the above input format\n"
    )
    for id in scene_graph.keys():
        obj = scene_graph[id]
        prompt += f"{obj['id']}: {obj['class']}"
        if room_label:
            prompt +=  f", {obj['room']}"
        prompt +=  f", [{', '.join(map(str, obj['related_objects_id']))}]\n"
    if len(scene_graph) == 0:
        prompt += "    No items in the 3D scene graph.\n"
    prompt += f"Do not print any object that are not included in the 3D scene graph or include any additional information other than the ID in your response:\n"
    prompt += "Answer (One object per line, no more than 20 lines): \n"
    
    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_objs(question, obj_infos, top_k=10, image_goal=None, use_room_filter = False):
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, obj_infos, top_k=top_k, image_goal=image_goal, room_label=use_room_filter
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image [{c[1][:10]}...]"
    
    response =  call_openai_api(prefiltering_sys, prefiltering_content)
    logging.info(message)
    logging.info(response)
    if response is None:
        return []
    # parse the response and return the top_k objects
    obj_id_set = set(obj_infos.keys())
    selected_objs = response.strip().split("\n")
    selected_objs = [int(id.strip()) for id in selected_objs if id.strip().isdigit()]
    selected_objs = [id for id in selected_objs if id in obj_id_set]
    selected_objs = selected_objs[:top_k]
    # if len(selected_objs) > top_k:
    #     selected_objs = random.sample(selected_objs, top_k)
    return selected_objs


def prefiltering_objs(
    question,
    objs,
    edges,
    top_k=10,
    image_goal=None,
    verbose=False,
    use_ollama=False,
    use_room_filter = False,
):
    obj_infos = {}
    for obj_id in objs.keys():
        obj_infos[obj_id] = {
            "id": obj_id,
            "pos": objs[obj_id]["bbox"].center,
            "class": objs[obj_id]["class_name"],
            "room": objs[obj_id]["room_label"],
            "related_objects_id": [],
        }
    for node in edges.keys():
        obj_infos[node[0]]["related_objects_id"].append(node[1])
    selected_objs = get_prefiltering_objs(
        question, obj_infos, top_k, image_goal, use_room_filter
    )
    if verbose:
        logging.info(f"Prefiltering selected objects: {selected_objs}")

   
    
    return selected_objs

def get_prefiltering_sg(edges, objs, images, selected_obj_id, image_to_edges, prompt_h, prompt_w):
    selected_objs = {obj_id: objs[obj_id] for obj_id in selected_obj_id}
    connected_objs = {}
    connected_objs.update(selected_objs)
    for node in edges.keys():
        if node[0] in selected_objs and node[1] not in selected_objs:
            connected_objs.update({node[1] : objs[node[1]]})
        elif node[1] in selected_objs and node[0] not in selected_objs:
            connected_objs.update({node[0] : objs[node[0]]})

    processed_images = {}
    selected_edges = {}

    if len(selected_objs) == 0:
        logging.info("No selected objects after prefiltering, returning empty dicts")
        return {}, {}, {}
    for a_obj_id in sorted(list(selected_objs.keys())):
        for b_obj_id in sorted(list(connected_objs.keys())):
            if (a_obj_id, b_obj_id) in edges and (b_obj_id, a_obj_id) not in selected_edges:
                selected_edges[(a_obj_id, b_obj_id)] = edges[(a_obj_id, b_obj_id)]
    selected_image_to_edges = {}
    for img in image_to_edges.keys():
        selected_image_to_edges[img] = list(set(image_to_edges[img]) & set(selected_edges.keys()))
    uncovered = {e: True for e in list(selected_edges.keys())}
    uncovered_cnt = len(uncovered)
    gain = {img: len(edges) for img, edges in selected_image_to_edges.items()}
    order = {img: i for i, img in enumerate(sorted(selected_image_to_edges.keys(), key=lambda x: str(x)))}
    heap = [(-gain[img], order[img], img) for img in selected_image_to_edges]
    heapq.heapify(heap)
    while uncovered_cnt > 0 and heap:
        neg_g, _, img = heapq.heappop(heap)
        g = -neg_g
        if g != gain.get(img, 0):
            continue
        if g <= 0:
            logging.info("Error in Greedy Image Allocation!!!")
            break

        image = images[img]
        resized_rgb = resize_image(
            image, prompt_h, prompt_w
        )
        processed_images[img] = encode_tensor2base64(resized_rgb)
        for e in list(selected_image_to_edges[img]):
            if uncovered[e]:
                uncovered[e] = False
                uncovered_cnt -= 1

                for other in selected_edges[e].rel_img:
                    if other == img:
                        continue
                    if gain.get(other, 0) > 0:
                        gain[other] -= 1

                        heapq.heappush(heap, (-gain[other], order[other], other))

    return connected_objs, selected_edges, processed_images

def explore_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    task_type = step["task_type"]
    use_room_filter = cfg.use_room_filter
    (
        question,
        image_goal,
        egocentric_imgs,
        selected_objs,
        selected_edges,
        processed_images,
        frontier_imgs
    ) = get_step_info(step, verbose, cfg.use_ollama, cfg.use_room_filter)
    history_decision = None
    if 'his_decision' in step:
        history_decision  = step['his_decision']
    sys_prompt, content, image_map_reverse = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        selected_objs,
        selected_edges,
        processed_images,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        history_decision = history_decision,
        room_label=use_room_filter,
        task_type = task_type
    )
    
    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        response = call_openai_api(sys_prompt, content)
        if response is None:
            logging.info("call_openai_api returns None, retrying")
            continue
        

        rarwresponse = response
        response = response.strip()
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()
        try:
            choice_type, choice_id = response.split(",")[0].strip().split(" ")
        except Exception as e:
            logging.info(f"Error in splitting response: {rarwresponse}")
            print(e)
            continue

        response_valid = False
        # if (
        #     choice_type == "image"
        #     and choice_id.isdigit()
        #     and 0 <= int(choice_id) < len(image_map_reverse)
        # ):
        #     response_valid = True
        if (
            choice_type == "image"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(image_map_reverse) + len(egocentric_imgs)
        ):
            try:
                object_class = (
                    response.split(",")[1].strip()
                )
            except Exception as e:
                logging.info(f"Error in splitting response: {rarwresponse}")
                print(e)
                continue
            response_valid = True
        elif (
            choice_type == "object"
            and choice_id.isdigit()
            and int(choice_id) in selected_objs
        ):
            response_valid = True
        elif (
            choice_type == "frontier"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(frontier_imgs)
        ):
            response_valid = True
        else:
            logging.info(f"Error choice_type: {rarwresponse}")
        if response_valid:
            final_response = response
            final_reason = reason
            break

    return (
        final_response,
        image_map_reverse,
        final_reason,
        len(image_map_reverse),
    )


def explore_step_by_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    task_type = step["task_type"]
    use_room_filter = cfg.use_room_filter
    (
        question,
        image_goal,
        egocentric_imgs,
        selected_objs,
        selected_edges,
        processed_images,
        frontier_imgs
    ) = get_step_info(step, verbose, cfg.use_ollama, cfg.use_room_filter)
    history_decision = None
    if 'his_decision' in step:
        history_decision  = step['his_decision']
    sys_prompt, content, image_map_reverse = format_obj_prompt(
        question,
        egocentric_imgs,
        selected_objs,
        selected_edges,
        processed_images,
        egocentric_view=step.get("use_egocentric_views", False),
        image_goal=image_goal,
        history_decision = history_decision,
        room_label = use_room_filter,
        task_type = task_type
    )
    
    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        response = call_openai_api(sys_prompt, content)
        if response is None:
            logging.info("call_openai_api returns None, retrying")
            continue
        

        rarwresponse = response
        response = response.strip()
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()
        try:
            choice_type, choice_id = response.split(",")[0].strip().split(" ")
        except Exception as e:
            logging.info(f"Error in splitting response: {rarwresponse}")
            print(e)
            continue

        response_valid = False
        # if (
        #     choice_type == "image"
        #     and choice_id.isdigit()
        #     and 0 <= int(choice_id) < len(image_map_reverse)
        # ):
        #     response_valid = True
        if (
            choice_type == "image"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(image_map_reverse) + len(egocentric_imgs)
        ):
            try:
                object_class = (
                    response.split(",")[1].strip()
                )
            except Exception as e:
                logging.info(f"Error in splitting response: {rarwresponse}")
                print(e)
                continue
            response_valid = True
        elif (
            choice_type == "object"
            and choice_id.isdigit()
            and int(choice_id) in selected_objs
        ):
            response_valid = True
        elif (
            choice_type == "continue"
            and choice_id == "exploration"
        ):
            logging.info(f"Response: [continue exploration]\nReason: [{reason}]")
            if len(frontier_imgs) == 1:
                final_response = "frontier 0"
                final_reason = "Only one frontier."
                response_valid = True
                break
            
            retry_bound_explore = 3
            
            while(retry_bound_explore > 0):
                retry_bound_explore -= 1
                sys_prompt, content = format_exploreonly_prompt(
                    question,
                    frontier_imgs,
                    image_goal=image_goal,
                )
                if verbose:
                    logging.info(f"Input prompt:")
                    message = sys_prompt
                    for c in content:
                        message += c[0]
                        if len(c) == 2:
                            message += f"[{c[1][:10]}...]"
                    logging.info(message)
                response = call_openai_api(sys_prompt, content)
                if response is None:
                    logging.info("call_openai_api returns None, retrying")
                    continue
                
                
                rarwresponse = response
                response = response.strip()
                if "\n" in response:
                    response = response.split("\n")
                    response, reason = response[0], response[-1]
                else:
                    reason = ""
                response = response.lower()
                try:
                    choice_type, choice_id = response.split(",")[0].strip().split(" ")
                except Exception as e:
                    logging.info(f"Error in splitting response: {rarwresponse}")
                    print(e)
                    continue
                if (
                    choice_type == "frontier"
                    and choice_id.isdigit()
                    and 0 <= int(choice_id) < len(frontier_imgs)
                ):
                    response_valid = True
                    break
                else:
                    logging.info(f"Error choice_type: {choice_type, choice_id, rarwresponse}")
            if response_valid:
                final_response = response
                final_reason = reason
                break
        else:
            logging.info(f"Error choice_type: {rarwresponse}")

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return (
        final_response,
        image_map_reverse,
        final_reason,
        len(image_map_reverse),
    )

def end_step(step, verbose=False):
    question, image_goal = format_question(step)
    egocentric_imgs = step["egocentric_views"]
    egocentric_imgs_encode = []
    for egocentric_view in egocentric_imgs:
        egocentric_imgs_encode.append(encode_tensor2base64(egocentric_view))
    sys_prompt, content = format_end_prompt(
        question,
        egocentric_imgs_encode,
        image_goal=image_goal,
    )
    
    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        response = call_openai_api(sys_prompt, content)
        if response is None:
            print("call_openai_api returns None, retrying")
            continue

        response = response.strip()
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()

        response_valid = False
        if response in ["yes", "no"]:
            response_valid = True
        elif response.startswith("yes"):
            response = "yes"
            response_valid = True
        elif response.startswith("no"):
            response = "no"
            response_valid = True

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return (
        final_response,
        final_reason,
    )