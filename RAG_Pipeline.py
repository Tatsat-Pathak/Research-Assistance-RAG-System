import re
import os
import json
import chromadb
from openai import OpenAI
import glob as g
import pymupdf4llm as p
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./upload_pdfs_chroma_db"

def clean_paper(paper : str):
    paper = re.sub(r"^>.*\n","", paper,flags = re.MULTILINE)
    paper = re.sub(r"^\*\*==>.*<==\*\*"," ", paper,flags = re.MULTILINE)
    paper = re.sub(r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*", " ", paper, flags = re.DOTALL | re.IGNORECASE)
    return paper

def paper_triming(paper_section_heading : list):
    abstract = re.compile(r"abstract|\bIntroduction", flags = re.IGNORECASE)
    conclusion = re.compile(r"\bconclu[a-z]*\b", flags = re.IGNORECASE)
    before_references = re.compile(r"references\b", flags = re.IGNORECASE)
    
    start_flag = False
    end_flag = False
    num_of_headings = len(paper_section_heading)
    start_index = 0
    end_index = num_of_headings 
    for index, heading in enumerate(paper_section_heading):
        header = heading["heading"]
        if not start_flag:
            if abstract.search(header):
                start_index = index
                start_flag = True

        elif not end_flag:
            if conclusion.search(header):
                end_index = index + 1 if (index + 1) < num_of_headings else index 
                end_flag = True
            elif before_references.search(header):
                end_index = index
                end_flag = True
    paper_section_heading = paper_section_heading[start_index:end_index + 1]
    return paper_section_heading

def getting_content_between_headers(paper : str, paper_section_heading : list):
    size = len(paper_section_heading)
    for i in range(size):
        if i < (size - 1):
            j = i + 1
            header = paper_section_heading[i]['heading']
            header_index = paper_section_heading[i]['end_index'] + 1
            next_header = paper_section_heading[j]['start_index']
            text = [sentence.strip().replace("\n", "") + "." for sentence in paper[header_index:next_header].split('.') if sentence.strip() != ""]
            section_size = len(' '.join(text))
            if text:
                paper_section_heading[i]['content'] = text
                paper_section_heading[i]['content size'] = section_size
    return paper_section_heading[:-1]

def parse_document(file_path):

    file_name = os.path.basename(file_path)
    document = p.to_markdown(file_path, header = False, footer = False, use_ocr = False)
    document = clean_paper(document)
    paper_section_heading = [
        {
            "heading" : match.group(1).strip(),
            "start_index" : match.start(),
            "end_index" : match.end()
        }
        for match in re.finditer(r"^#+\s?[\*_]*([I{1,3}VXa-z0-9:\-. ]+)[\*_]*", document, re.MULTILINE | re.IGNORECASE) if match.group(1).strip() != ''
    ]
    paper_section_heading = paper_triming(paper_section_heading)
    paper_section_heading = getting_content_between_headers(document, paper_section_heading)

    return file_name, document, paper_section_heading

def chunks_generation(papers_section_heading : dict, chunk_size : int = 500):
    paper_chunks = {}

    for title, contents in papers_section_heading.items():
        chunks = []
        ids = []
        s = 0
        for section in contents:
            i = 0
            # list_of_chunk = []

            if "content" in section.keys():
                heading = section['heading']
                chunk = f"{heading}\n\n"
                list_size = len(section['content'])
                for index, sentence in enumerate(section['content']):
                    last_index = index + 1
                    sentence_size = len(sentence)
                    current_size = len(chunk)
                        
                    if current_size + sentence_size < chunk_size and chunk:
                        current_size += sentence_size
                        chunk += sentence + " "

                    else:
                        chunks.append(chunk)
                        ids.append(f"{title}_{heading}_{s}_chunk_{i}")
                        chunk = f"{heading}\n\n" + section['content'][index - 1] + " " + sentence + " "
                        i += 1

                    if last_index == list_size and chunk:
                        chunks.append(chunk)
                        ids.append(f"{title}_{heading}_{s}_chunk_{i}")
                        i += 1

                s += 1 
                # chunks.append(list_of_chunk)
                # print(chunks)

        paper_chunks[title] = {
            'chunks' : chunks,
            'ids' : ids
        }

    return paper_chunks

def combining_chunks(papers_chunks_dic : dict):
    all_chunks = []
    all_ids = []
    for title in papers_chunks_dic.keys():
        chunks = papers_chunks_dic[title]['chunks']
        ids = papers_chunks_dic[title]['ids']
        for i in range(len(chunks)):
            all_chunks.append(chunks[i])
            all_ids.append(ids[i])
    
    return all_chunks, all_ids

def load_embedded_model(model_name : str = "BAAI/bge-base-en-v1.5"):
    embedding_model = SentenceTransformer(model_name)
    return embedding_model

def create_embedding_text(embedding_model : object, chunks : list, size : int = 64):
    embedding_text = embedding_model.encode(
    chunks,
    batch_size = size,
    normalize_embeddings = True
    )
    return embedding_text, embedding_text.shape

def initialize_vector_database():
    os.makedirs(CHROMA_PATH, exist_ok = True)

    client = chromadb.PersistentClient(path = CHROMA_PATH)

    return client

def initialize_collections(client : object):
    collection = client.create_collection("chroma_db_documents")

    return collection

def get_collection(client : object, collection_name : str = "chroma_db_documents"):
    collection = client.get_collection(collection_name)

    return collection

def add_in_collection(collection, ids : list, chunks : list, embedded_text):
    collection.add(
        ids = ids,
        documents = chunks,
        embedding = embedded_text.tolist()
    )

def query_builder(user_query : str):
    context = ""
    sources = []
    query_embedding = embedding_model.encode(
        user_query,
        normalize_embeddings = True
    )

    result = collection.query(
        query_embeddings = query_embedding.tolist(),
        n_results = 10,
    )
    
    ids = result['ids'][0]
    chunks = result['documents'][0]
    num_of_chunks_retrieved = len(ids)
    
    for i in range(num_of_chunks_retrieved):
        source = ids[i].split(".pdf")[0]
        if source not in sources:
            sources.append(source)

        context += f"""
Paper   : {source}
Section : {chunks[i]}

-------"""
    
    prompt = f"""
You are an expert research assistance helping researchers and students. 
You job is to give precise and clean answer user question clearly and accurately using only the context provided.
if the answer is not in the context, say sorry can't answer it based on given context.
You may ask clarifying or counter-questions to deepen understanding.


User question : 
{user_query}


Context:
{context}
"""
    return prompt, sources, ids

def answer_generator(prompt : str):
    
    load_dotenv()

    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPEN_ROUTER_API_KEY'),
)

    try:

        response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
            ],
        extra_body={"reasoning": {"enabled": True}}
    )
        message = response.choices[0].message

        return {
            "success": True,
            "content": message.content,
            "reasoning": message.reasoning,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "content": None,
            "reasoning": None,
            "error": str(e)
        }