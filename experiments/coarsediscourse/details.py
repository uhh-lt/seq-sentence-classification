project_name = "Discourse Act Analysis of Reddit Discussions"

project_details = "analyzing the discourse acts of Reddit threads. Posts and comments are categorized by coarse discourse acts."

label_dict = {
    "question": "A comment with a question or a request seeking some form of feedback, help, or other kinds of responses. While the comment may contain a question mark, it is not required. For instance, it might be posed in the form of a statement but still soliciting a response. Also, not everything that has a question mark is automatically a QUESTION. For instance, rhetorical questions are not seeking a response. Relation: This comment might be the first in a thread and have no relation to another comment. Or, it could be a clarifying or followup QUESTION linking to any prior comment.",
    "answer": "A comment that is responding to a QUESTION by answering the question or fulfilling the request. There can be more than one ANSWER responding to a QUESTION. Relation: An ANSWER is always linked to a QUESTION.",
    "announcement": "A comment that is presenting some new information to the community, such as a piece of news, a link to something, a story, an opinion, a review, or insight. Relation: This comment has no relation to a prior comment and is always the initial post in a thread.",
    "agreement": "A comment that is expressing agreement with some information presented in a prior comment. It can be agreeing with a point made, providing supporting evidence, providing a positive example or experience, or confirming or acknowledging a point made. Relation: This comment is always linked to a prior comment to which it is agreeing.",
    "appreciation": "A comment that is expressing thanks, appreciation, excitement, or praise in response to another comment. In contrast to AGREEMENT, it is not evaluating the merits of the points brought up. Comments of this category are more interpersonal as opposed to informational. Relation: This comment is always linked to a prior comment for which it is expressing appreciation.",
    "disagreement": "A comment that is correcting, criticizing, contradicting, or objecting to a point made in a prior comment. It can also be providing evidence to support its disagreement, such as an example or contrary anecdote.Relation: This comment is always linked to a prior comment to which it is disagreeing.",
    "negative reaction": "A comment that is expressing a negative reaction to a previous comment, such as attacking or mocking the commenter, or expressing emotions like disgust, derision, or anger, to the contents of the prior comment. This comment is not discussing the merits of the points made in a prior comment or trying to correct them. Relation: This comment is always linked to a prior comment to which it is negatively reacting.",
    "elaboration": "A comment that is adding additional information on to another comment. Oftentimes, one can imagine it simply appended to the end of the comment it elaborates on. One can elaborate on many kinds of comments, for instance, a questionasker elaborating on their question to provide more context, or someone elaborating on an answer to add more information. Relation: This comment is always linked to a prior comment upon which it is elaborating.",
    "humor": "This comment is primarily a joke, a piece of sarcasm, or a pun intended to get a laugh or be silly but not trying to add information. If a comment is sarcastic but using sarcasm to make a point or provide feedback, then it may belong in a different category. Relation: At times, this comment links to another comment but other times it may not be responding to anything.",
    "other": "A comment that does not fit any of the previous definitions.",
}

system_prompt_template = """
You are a professional annotator specialized in annotating post and comments of reddit threads with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "This post is asking a question.",
        "category": "question"
    },
    {
        "text_id": 2,
        "category": "agreement"
    },
    {
        "text_id": 3,
        "reason": "This comment expresses a negative attitude.",
        "category": "negative reaction"
    },
    ...
]
"""

system_prompt_with_guidelines_template = """
You are a professional annotator specialized in annotating post and comments of reddit threads with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Annotation Guidelines:
<annotation_guidelines>

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "This post is asking a question.",
        "category": "question"
    },
    {
        "text_id": 2,
        "category": "agreement"
    },
    {
        "text_id": 3,
        "reason": "This comment expresses a negative attitude.",
        "category": "negative reaction"
    },
    ...
]
"""

user_prompt_template = """
Please annotate each post/comment of the following thread.

Thread:
{document}

Remember to annotate every provided post/comment. You MUST use the categories provided in the Annotation Guidelines!
"""

user_prompt_with_guidelines_template = """
Please annotate each post/comment of the following thread.

Annotation Guidelines:
<annotation_guidelines>

Thread:
{document}

Remember to annotate every provided post/comment. You MUST use the categories provided in the Annotation Guidelines!
"""
