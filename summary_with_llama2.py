from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
import transformers
from transformers import AutoTokenizer
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate
from langchain_community.llms import CTransformers


import os

file_path = "model/llama-2-7b-chat.Q8_0.gguf"
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="model/llama-2-7b-chat.Q8_0.gguf",
    temperature=0.7,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2000
)

template = """
            Imagine you are a professional evaluator of interviews. You receive the following text as a transcript of the
            interview and are asked to summarize it. You do not invent any additional information. There may be errors
            in the transcript of the interview, correct them if possible and try to understand the text anyway. There are
            adverts in the transcript of the podcast, please ignore them.
              {text}
              SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = prompt | llm


text = ''' 

To start, I would like to talk to you, Jens, about the production that you have here, because that is really something special when you are more classically in the field of mass production and then get to know a mechanical engineer.
Then we will experience what it means to have such a small series production.
What would you say is the biggest difference? The biggest difference? I think the biggest difference is in the area of processes that are still small..
areas are often much longer, long running times, but also paired with high manufacturing costs due to small start-ups.
We don't build a thousand machines a year.
Our largest single-piece numbers are 2,000 to 3,000 components of the same nature and a large range of suppliers, also characterized by small-scale production.
And most of the time, the complexity of the product and the enormous variety of products.
Can you give us an insight into the size of the machines, what kind of variety of products you have here? Yes, we build machines for the craft and we build machines for the industry.
So from the carpenter around the corner with five or six employees to the large-scale industry, IKEA suppliers.
That's the spectrum we cover in plate split saws and, of course, the complexity of the product itself..
with a Horrborder and an automated warehouse.
So not a single machine, but a complete solution that you produce here in the factory.
Yes, exactly.
'''

print(llm_chain.invoke(text))
