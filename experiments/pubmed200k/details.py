project_name = "Rethorical Roles Analysis of PubMed Abstracts"

project_details = "analyzing the rhetorical roles of sentences in abstracts of randomized controlled trials. The abstracts are collected from PubMed."

label_dict = {
    "background": "Provides context or previous knowledge relevant to the research topic. Think of it as setting the stage for the study.",
    "methods": "Describes the procedures and techniques used in the research. This includes the study design, data collection, and analysis methods.",
    "objective": "States the main goal or purpose of the research. What question is this work trying to answer?",
    "results": "Presents the findings or outcomes of the research. This often includes statistical data, tables, and figures.",
    "conclusions": "Summarizes the key findings of the research and draw inferences from those findings. They provide closure to the abstract, summarizing the overall contribution of the research.",
}

system_prompt_template = """
You are a professional annotator specialized in annotating sentences of a document with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "The sentence provides context for the research.",
        "category": "background"
    },
    {
        "text_id": 2,
        "category": "methods"
    },
    {
        "text_id": 3,
        "reason": "The sentence presents the research findings.",
        "category": "results"
    },
    ...
]
"""

system_prompt_with_guidelines_template = """
You are a professional annotator specialized in annotating sentences of a document with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Annotation Guidelines:
<annotation_guidelines>

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "The sentence provides context for the research.",
        "category": "background"
    },
    {
        "text_id": 2,
        "category": "methods"
    },
    {
        "text_id": 3,
        "reason": "The sentence presents the research findings.",
        "category": "results"
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

user_prompt_with_guidelines_template = """
Please annotate each sentence of the following document.

Annotation Guidelines:
<annotation_guidelines>

Document:
{document}

Remember to annotate every provided sentence. You MUST use the categories provided in the Annotation Guidelines!
"""
