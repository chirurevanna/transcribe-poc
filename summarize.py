from openai import OpenAI
import os


client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def summarize_text(text, custom_prompt):
    response = client.chat.completions.create(


        messages=[
            {
                "role": "user",
                "content": f"{custom_prompt}\n\n{text}",
            }
        ],
        model="gpt-3.5-turbo"
    )
    return response


def main():
    file_path = 'output/translated_text.txt'
    text = read_file(file_path)
    custom_prompt = "Imagine you are a professional evaluator of interviews. You receive the following text as a transcript of the interview and are asked to summarize it. You do not invent any additional information. There may be errors in the transcript of the interview, correct them if possible and try to understand the text anyway. There are adverts in the transcript of the podcast, please ignore them.:"

    summary = summarize_text(text, custom_prompt)
    print("Summary:")
    print(summary)


if __name__ == "__main__":
    main()
