from fastapi import FastAPI,HTTPException
import pinecone
import dotenv
import openai
from pydantic import BaseModel

apiKeys = dotenv.dotenv_values('.env')
openai.api_key = apiKeys['openai_api_key']

app = FastAPI()

@app.get('/')
def index():
    return {'hello world'}


class SearchResult(BaseModel):
    content: str


@app.post('/')
def search_docs(userIput: SearchResult):
    content = userIput.content
    pinecone.init(api_key= apiKeys['pinecone'], environment='gcp-starter')
    index = pinecone.Index('bioproduct-index')

    try:
        xq = openai.Embedding.create(input=content, engine="text-embedding-ada-002")['data'][0]['embedding']
        res = index.query([xq], top_k=1, include_metadata=True)
        return res['matches'][0]['metadata']['text']
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
