import chainlit as cl
from src.pipelines.naive_rag import NaiveRAGPipeline

@cl.on_chat_start
async def start():
    # Instantiate the NaiveRAGPipeline and store it in user session
    rag_pipeline = NaiveRAGPipeline()
    cl.user_session.set("rag_pipeline", rag_pipeline)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the pipeline from the user session
    rag_pipeline = cl.user_session.get("rag_pipeline")
    
    # Run the query against the pipeline
    res = rag_pipeline.query(message.content)
    
    if res and "answer" in res:
        answer = res["answer"]
    else:
        answer = "Error: Failed to process query."
        
    # Send the response back to the user
    await cl.Message(content=answer).send()
