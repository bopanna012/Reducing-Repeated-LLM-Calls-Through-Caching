import ollama

def ask_llm(question, context):

    prompt = f'''
    Context:
    {context}

    Question:
    {question}
    '''

    response = ollama.chat(
        model='llama3',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response['message']['content']