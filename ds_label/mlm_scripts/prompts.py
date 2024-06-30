instr_prompt = '''You are a helpful assistant. You need to help me detect the car driver head direction.

This is a picture of car driver taken during driving from front left corner of the car. Please give me the response as requested below.

First, you need to generate the Observation and Thought.
Observation: You need to briefly describe the picture including steering wheel, side window and driver head direction.
Thought: Based on Observation, You need to think about in which direction the driver head is actually facing relative to the steering wheel by taking into account the fact that the picture is taken from front left corner of the car.

Then, you need to generate the direction based on the Thought. You can output the following 5 direction:
1. Straight. The driver head is facing straight relative to the steering wheel and the driver is looking through the windshield and to the front.
2. Left. The driver head is facing left relative to the steering wheel and the driver is looking to the left.
3. Right. The driver head is facing right relative to the steering wheel and the driver is looking through the side mirror or window and to the right.
4. Down. The driver is looking inside car and facing his head down.
5. Up. The driver is looking through the windshield and to the top.
Note: Be careful in accounting for the fact that the driver picture is taken from the front left corner of the car and is not straightly facing the driver's face.

You only speak JSON. DO not write any text that isn't JSON. So, Your output must be in the following JSON format:
{ "Observation" : "Generate as required by Observation",
 "Thought" : "Generate as required by Thought",
 "Direction" : Choose a direction from above 5 described directions based on the actual direction of driver head"
 }
'''

prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{instr_prompt} ASSISTANT:"

instr_prompt_m = '''This is a video of car driver taken during driving from front left corner of the car. You need to detect whether his mouth is closed or talking or yawning.

You only speak JSON. DO not write any text that isn't JSON. So, Your output must be in the following JSON format:
{ "Mouth" : Closed or Talking or Yawning }
'''

mlm_m_conv = [
    {
        "role":
        "system",
        "content": [
            {
                "type":
                "text",
                "text":
                "You are a helpful assistant. You need to help me detect the activity of car driver mouth while driving."
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": instr_prompt_m
            },
            {
                "type": "video"
            },
        ],
    },
]
