project_name = "Emotion Analysis of Friends TV Show Dialogs"

project_details = "analyzing basic emotions of spoken dialogs. The dialogs are collected from Friends TV scripts."

label_dict = {
    "fear": "A feeling of apprehension or dread in response to a perceived threat or danger. It can range from mild anxiety to intense terror.",
    "disgust": "A feeling of revulsion or aversion, often triggered by something perceived as unpleasant, unsanitary, or morally offensive.",
    "excited": "A state of heightened arousal and positive anticipation. It often involves feelings of enthusiasm, eagerness, and energy.",
    "anger": "A feeling of intense displeasure or hostility, often triggered by a perceived wrong or injustice. It can manifest as irritation, frustration, rage, or fury.",
    "surprise": "A brief emotional state in response to an unexpected event. It can be positive, negative, or neutral, depending on the nature of the surprise.",
    "sadness": "A feeling of sorrow, grief, or disappointment. It can range from mild melancholy to intense despair.",
    "joy": "A feeling of happiness, contentment, or pleasure. It can manifest as amusement or love.",
    "neutral": "A state of emotional balance or equilibrium, where no particular emotion is dominant.",
}

system_prompt_template = """
You are a professional annotator specialized in annotating spoken dialogs with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "The speaker of this utterance is expressing fear.",
        "category": "fear"
    },
    {
        "text_id": 2,
        "category": "joy"
    },
    {
        "text_id": 3,
        "reason": "This speaker seems to be disgusted by the situation.",
        "category": "disgust"
    },
    ...
]
"""

system_prompt_with_guidelines_template = """
You are a professional annotator specialized in annotating spoken dialogs with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Annotation Guidelines:
<annotation_guidelines>

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "The speaker of this utterance is expressing fear.",
        "category": "fear"
    },
    {
        "text_id": 2,
        "category": "joy"
    },
    {
        "text_id": 3,
        "reason": "This speaker seems to be disgusted by the situation.",
        "category": "disgust"
    },
    ...
]
"""

user_prompt_template = """
Please annotate each utterance of the following dialog.

Dialog:
{document}

Remember to annotate every provided utterance. You MUST use the categories provided in the Annotation Guidelines!
"""

user_prompt_with_guidelines_template = """
Please annotate each utterance of the following dialog.

Annotation Guidelines:
<annotation_guidelines>

Dialog:
{document}

Remember to annotate every provided utterance. You MUST use the categories provided in the Annotation Guidelines!
"""
