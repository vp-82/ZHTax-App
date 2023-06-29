import logging
import os
import re

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the logging format
)


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

        prompt_template = """Angesichts der folgenden Konversation und einer anschliessenden Frage, formulieren Sie die Nachfrage so um, dass sie als eigenständige Frage gestellt werden kann.
            Alle Ausgaben muessen in Deutsch sein.
            Wenn Du die Antwort nicht kennst, sage einfach, dass Du es nicht weisst, versuche nicht, eine Antwort zu erfinden.

            Chatverlauf:
            {chat_history}
            Nachfrage: {question}
            Eigenständige Frage:

            Zum Beispiel: 
            Chatverlauf: 'Ich habe gestern einen Film gesehen.' 'Oh, welchen Film haben Sie gesehen?' 'Ich habe Titanic gesehen.'
            Nachfrage: 'War es traurig?'
            Eigenständige Frage: 'War der Film Titanic, den Sie gesehen haben, traurig?'
            """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question"]
        )

        
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k-0613')
        question_generator = LLMChain(llm=llm,
                                    prompt=PROMPT,
                                    )
        doc_chain = load_qa_with_sources_chain(
                                            llm,
                                            chain_type="map_reduce"
                                            )

        self.chain = ConversationalRetrievalChain(
            retriever=self.milvus.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
        )

    def process_output(self, output):
        # Check if 'SOURCES: \n' is in the output
        if 'SOURCES:' in output['answer']:
            # Split the answer into the main text and the sources
            answer, raw_sources = output['answer'].split('SOURCES:', 1)

            # Split the raw sources into a list of sources, and remove any leading or trailing whitespaces
            raw_sources_list = [source.strip() for source in raw_sources.split('- ') if source.strip()]

            # Process each source to turn it back into a valid URL
            sources = []
            for raw_source in raw_sources_list:
                if raw_source:  # Ignore empty strings
                    # Extract the relevant part of the path and replace '__' with '/'
                    valid_url = 'https://' + raw_source.split('/')[-1].replace('__', '/').rstrip('.txt')
                    sources.append(valid_url)
        else:
            # If there are no sources, return the answer as is and an empty list for sources
            answer = output['answer']
            sources = []

        # Join the sources list into a single string with each source separated by a whitespace
        sources = ' '.join(sources)

        return answer, sources


    def get_answer(self, query):

        result = self.chain({"question": query, "chat_history": self.chat_history})

        return result