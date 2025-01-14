project_name = "Persuasion Techniques in News Articles"

project_details = "identifying persuasion techniques in news articles about widely discussed topics such as COVID-19, climate change, abortion, migration, etc."

label_dict = {
    "Name Calling/Labeling": "Name-Calling uses biased or emotionally charged terms (either positive or negative) to create a specific impression of a person or group,  influencing the audience's perception without providing factual support. This technique differs from Loaded Language in that it focuses solely on labeling the subject rather than using persuasive language throughout the entire argument.",
    "Guilt by Association": "Guilt by Association aims to discredit a person, group, or idea by linking it to something else that the audience strongly dislikes. This creates a negative impression without directly addressing the merits of the original subject, making it a form of Casting Doubt.",
    "Doubt": "Casting Doubt seeks to discredit a person, idea, or action by attacking their credibility or past failures instead of engaging with their actual argument. This technique relies on undermining trust and confidence to sway the audience's opinion.",
    "Appeal to Hypocrisy": "Appeal to Hypocrisy, also known as Tu Quoque, attempts to discredit an argument by pointing out the speaker's hypocrisy or inconsistency in their beliefs or actions. This technique shifts the focus away from the argument itself and onto the speaker's perceived character flaws.",
    "Questioning the Reputation": "Questioning the Reputation casts doubt on a person, group, or idea by questioning their credibility, integrity, or trustworthiness. This technique aims to undermine the subject's reputation and credibility, making it harder for the audience to trust or support them.",
    "Flag Waving": "Flag-Waving appeals to patriotism or national pride to distract from the merits of an argument or justify questionable actions. By associating the subject with positive national symbols or values, this technique seeks to evoke an emotional response rather than engage with the issue at hand.",
    "Appeal to Authority": "Appeal to Authority relies on citing an expert or respected figure to support an argument, appealing to the audience's trust in the authority's knowledge or credibility. This technique can be persuasive when the authority is relevant to the subject matter and their expertise is well-established.",
    "Appeal to Popularity": "Appeal to Popularity, also known as Bandwagon, argues that an idea or action is valid or desirable because it is popular or widely accepted. This technique leverages social proof to sway the audience's opinion, even if the majority view is not based on facts or logic.",
    "Appeal to Values": "Appeal to Values appeals to the audience's moral or ethical beliefs to justify an argument or action. By framing the subject in terms of shared values or principles, this technique seeks to create a sense of common ground and build rapport with the audience.",
    "Appeal to Fear/Prejudice": "Appeal to Fear and Prejudice uses emotional manipulation to persuade the audience by playing on their fears, biases, or stereotypes. This technique aims to evoke a strong emotional response rather than engage with the merits of the argument, often leading to hasty or irrational decisions.",
    "Straw Man": "Straw Man distorts or misrepresents an opponent's argument to make it easier to attack or refute. By creating a weaker version of the original argument, this technique allows the speaker to appear victorious without engaging with the actual points raised by their opponent.",
    "Red Herring": "Red Herring introduces irrelevant or misleading information to divert attention from the main issue or argument. This technique aims to confuse or distract the audience, making it harder for them to focus on the key points being discussed.",
    "Whataboutism": "Whataboutism deflects criticism or scrutiny by pointing out the flaws or wrongdoings of others, rather than addressing the issue at hand. This technique seeks to shift the focus away from the original topic and onto a different subject, often leading to a stalemate in the conversation.",
    "Causal Oversimplification": "Causal Oversimplification reduces a complex issue to a single cause-and-effect relationship, ignoring other contributing factors or nuances. By oversimplifying the explanation, this technique can mislead the audience and prevent them from understanding the full context of the issue.",
    "False Dilemma": "False Dilemma, also known as Black-and-White Thinking, presents a limited number of options as the only possible choices, ignoring other alternatives or nuances. By framing the issue as a binary decision, this technique oversimplifies complex problems and limits the audience's understanding of the situation.",
    "Consequential Oversimplification": "Consequential Oversimplification downplays or ignores the potential consequences or implications of an action, idea, or decision, making it seem less risky or harmful than it actually is. By simplifying the outcome, this technique can mislead the audience and prevent them from fully understanding the risks involved.",
    "Slogans": "Slogans use catchy or memorable phrases to promote an idea, product, or cause, often without providing substantive information or evidence. This technique relies on repetition and simplicity to reinforce a message and make it more memorable to the audience.",
    "Conversation Killer": "Conversation Killer shuts down discussion or debate by dismissing opposing viewpoints or refusing to engage with counterarguments. This technique aims to silence dissenting voices and maintain control over the conversation, preventing meaningful dialogue or exchange of ideas.",
    "Appeal to Time": "Appeal to Time argues that an idea or action is valid or necessary because it is timely or urgent. By framing the subject in terms of the current context or situation, this technique seeks to create a sense of urgency or importance, motivating the audience to take action or support the argument.",
    "Loaded Language": "Loaded Language uses emotionally charged words or phrases to evoke a strong reaction from the audience, influencing their perception of the subject without providing factual support. This technique can create a biased or one-sided view of the issue, appealing to the audience's emotions rather than their reason.",
    "Obfuscation/Vagueness/Confusion": "Obfuscation, Vagueness, and Confusion introduce ambiguity or complexity to obscure the main issue or argument, making it harder for the audience to understand or evaluate the information presented. This technique aims to create doubt or uncertainty, preventing the audience from reaching a clear conclusion.",
    "Exaggeration/Minimisation": "Exaggeration and Minimisation distort the significance or impact of an event, idea, or argument to make it seem more or less important than it actually is. By manipulating the scale or scope of the subject, this technique can mislead the audience and influence their perception of the issue.",
    "Repetition": "Repetition uses repeated words, phrases, or ideas to reinforce a message and make it more memorable to the audience. This technique relies on the principle of familiarity to create a sense of trust or authority, even if the content itself lacks substance or evidence.",
    "O": "O represents the absence of any of the above techniques in the sentence. This label should be used when none of the other labels apply or when the sentence does not contain any persuasive techniques.",
}

system_prompt_template = """
You are a professional annotator specialized in identifying persuasion techniqes in news articles with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Annotation Guidelines:
<annotation_guidelines>

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "This sentence uses emotionally charged terms.",
        "category": "Name Calling/Labeling"
    },
    {
        "text_id": 2,
        "category": "O"
    },
    {
        "text_id": 3,
        "category": "O"
    },
    {
        "text_id": 4,
        "reason": "This sentence repeats the same idea over and over.",
        "category": "Repetition"
    },
    ...
]
"""

user_prompt_template = """
Please annotate each sentence of the following document.

Document:
{document}

Remember to annotate every provided sentence. You MUST use the categories provided in the Annotation Guidelines!
"""