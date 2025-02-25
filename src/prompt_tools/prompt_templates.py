gqa_templates = {
    "preprompt": {
        "v1": """
You are now a question parser. Your task is to translate a question into a functional program.
The available operations are: select, relate, common, verify, choose, filter, query, same, different, and, or, exist.
"""
    },
    "template": {
        "v1": """
Here are examples of question and corresponding programs:
{examples}

{question}
"""
    },
}

asp_templates = {
    "preprompt": {
        "v1": """
You are now a question parser. Your task is to translate a question into an Answer Set Program. The available operations are: scene, end, select, relate, query, verify_rel, choose_attr, choose_rel, exist, all_same, all_different, two_same, two_same, and, or, unique, negate.
""",
        "v2": """
Imagine you are a translator between human language and a programming language called Answer Set Programming (ASP). Your task is to convert questions into ASP code.
""",
        "v3": """
You are now a question parser. Your task is to translate a question into an Answer Set Program.
""",
        "v4": """
As an AI, you have the ability to understand human language and translate it into Answer Set Programming (ASP). Your task is to convert the given question into an ASP program.
""",
        "v5": """
You are an AI language translator. Your job is to take a question and translate it into an Answer Set Program.
""",
        "v6": """
You are an AI that can understand human language and convert it into Answer Set Programming (ASP). Your task is to translate the given question into an ASP program.
""",
        "v7": """
Imagine you are an AI that can understand human language and translate it into a programming language called Answer Set Programming (ASP). Your task is to convert the given question into an ASP program.
""",
        "v8": """
You are an AI that can translate human language into Answer Set Programming (ASP). Your task is to convert the given question into an ASP program.
""",
        "v9": """
As an AI, you have the ability to understand human language and translate it into a programming language called Answer Set Programming (ASP). Your task is to convert the given question into an ASP program.
""",
        "v10": """
You are an AI that can understand human language and convert it into a programming language called Answer Set Programming (ASP). Your task is to translate the given question into an ASP program.
""",
    },
    "template": {
        "v1": """
Here are examples of question and corresponding programs:
{examples}

{question}
""",
        "v2": """
To help you understand the task, here are some examples of questions and their corresponding ASP programs:
{examples}

Now, try to translate the following question into an ASP program:
{question}
""",
        "v3": """
Consider the following examples of questions and their corresponding ASP programs:
{examples}

Based on these examples, translate the following question into an ASP program:
{question}
""",
        "v4": """
Here are some examples of questions and their corresponding ASP programs:
{examples}

Can you translate the following question into an ASP program?
{question}
""",
        "v5": """
Look at these examples of questions and their corresponding ASP programs:
{examples}

Translate the following question into an ASP program:
{question}
""",
        "v6": """
Here are some examples of questions and their corresponding ASP programs to help you understand the task:
{examples}

Now, translate the following question into an ASP program:
{question}
""",
        "v7": """
Consider these examples of questions and their corresponding ASP programs:
{examples}

Translate the following question into an ASP program:
{question}
""",
        "v8": """
Here are some examples of questions and their corresponding ASP programs:
{examples}

Can you translate the following question into an ASP program?
{question}
""",
        "v9": """
Look at these examples of questions and their corresponding ASP programs:
{examples}

Translate the following question into an ASP program:
{question}
""",
        "v10": """
Here are some examples of questions and their corresponding ASP programs to help you understand the task:
{examples}

Now, translate the following question into an ASP program:
{question}
""",
    },
}
