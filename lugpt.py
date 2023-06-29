import os
import re

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus


class QueryHandler:
    def __init__(self, openai_api_key, milvus_api_key):
        load_dotenv()  # take environment variables from .env.
        self.openai_api_key = openai_api_key
        self.milvus_api_key = milvus_api_key

        connection_args = {
            "uri": "https://in03-5052868020ac71b.api.gcp-us-west1.zillizcloud.com",
            "user": "vaclav@pechtor.ch",
            "token": milvus_api_key,
            "secure": True,
        }

        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.milvus = Milvus(
            embedding_function=self.embeddings,
            collection_name="LuGPT",
            connection_args=connection_args,
        )
        self.chat_history = []
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True
                                               )
        
        self.qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0),
                                                        self.milvus.as_retriever(),
                                                        memory=self.memory
                                                        )

    def transform_source_to_url(self, source_value: str) -> str:
        match = re.search(r"([a-z]+\.[a-z]+\.[a-z]+)", source_value)
        if match:
            url_start_index = match.start()
        else:
            raise ValueError("Cannot find a URL in the source value.")
        url_part = source_value[url_start_index:]
        url_part = url_part.replace("__", "/")
        url_part = os.path.splitext(url_part)[0]
        final_url = "https://" + url_part
        return final_url

    def get_answer(self, query, language="German"):
        if language == "German":
            model_type = "gpt-3.5-turbo-16k-0613"
        else:
            model_type = "gpt-4-0613"

        prompt_template = f"""You are an assistant that answers questions about the Kanton Luzern,
         based on given information. Only use the information that was provided below.
         Use the following pieces of context to answer the question at the end.
         If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {{context}}

        Question: {{question}}
        Answer in {language}:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        result = self.qa({"question": PROMPT, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        return result